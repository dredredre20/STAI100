import json
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from resume_processing.pipeline import load_resume_text, run_resume_intake_pipeline
from session_store.persistence import create_session, save_resume_profile, save_diff_result
from gap_diff.diff_engine import run_gap_diff
from chatbot import run_orchestrator

from guardrails.input_guardrail import check_input
from session_store.db_setup import init_db

from config import MODEL
from llmops.llmops_logger import log_request


app = FastAPI(title="STAI100 Resume Intake API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    init_db()


def process_uploaded_resume(uploaded_file: Path | str, target_role: str | None = None) -> dict[str, Any]:
    temp_path = Path(uploaded_file)
    if not temp_path.exists():
        raise FileNotFoundError(f"Uploaded file not found: {temp_path}")
    resume_text = load_resume_text(str(temp_path))
    return run_resume_intake_pipeline(
        resume_text,
        verbose=False,
        interactive=False,
        target_role_override=target_role,
    )


def persist_completed_profile(result: dict[str, Any]) -> str:
    profile = result["validated_profile"]
    try:
        session_id = create_session()
        resume_profile_id = save_resume_profile(session_id, profile)
        diff_result = run_gap_diff(profile.get("skills", []), profile.get("target_role"))
        save_diff_result(session_id, resume_profile_id, profile.get("target_role"), diff_result)
        return session_id
    except Exception:
        traceback.print_exc()   # TEMP DEBUG — shows full traceback in uvicorn terminal
        raise


@app.post("/process")
def process_resume(
    file: UploadFile = File(...),
    target_role: str | None = Form(default=None),
):
    """Non-streaming endpoint — kept for simple/synchronous callers (e.g.
    scripts, testing) that just want the final result in one response."""
    start_ts = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "resume.pdf").suffix or ".pdf", delete=False) as tmp:
            tmp.write(file.file.read())
            temp_path = Path(tmp.name)
        result = process_uploaded_resume(temp_path, target_role=target_role)
        temp_path.unlink(missing_ok=True)

        if result.get("is_complete"):
            result["session_id"] = persist_completed_profile(result)

        latency_ms = (time.perf_counter() - start_ts) * 1000
        log_request(
            model=MODEL,
            prompt=f"[resume upload: {file.filename}, target_role={target_role}]",
            completion=json.dumps(result.get("validated_profile") or result.get("clarification_question") or ""),
            latency_ms=latency_ms,
            guardrail_fired=False,
            extra={"interface": "api", "endpoint": "/process"},
        )
        return result
    except Exception as exc:  # pragma: no cover - defensive API handling
        latency_ms = (time.perf_counter() - start_ts) * 1000
        log_request(
            model=MODEL,
            prompt=f"[resume upload: {file.filename}]",
            completion=f"ERROR: {exc}",
            latency_ms=latency_ms,
            guardrail_fired=False,
            extra={"interface": "api", "endpoint": "/process", "error": True},
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _sse_stage_generator(temp_path: Path, target_role: str | None) -> AsyncGenerator[str, None]:
    """Yields SSE events for each pipeline stage, then a final event with the
    full structured result. Streams stage progress, not tokens, since this
    pipeline produces structured data rather than prose until the very end."""
    start_ts = time.perf_counter()
    stages = [
        "Reading resume...",
        "Redacting personal information...",
        "Extracting profile details...",
        "Checking for missing information...",
    ]
    for stage_label in stages:
        yield f"data: {json.dumps({'type': 'stage', 'label': stage_label})}\n\n"

    try:
        resume_text = load_resume_text(str(temp_path))
        result = run_resume_intake_pipeline(
            resume_text,
            verbose=False,
            interactive=False,
            target_role_override=target_role,
        )

        if result.get("is_complete"):
            result["session_id"] = persist_completed_profile(result)

        latency_ms = (time.perf_counter() - start_ts) * 1000
        log_request(
            model=MODEL,
            prompt=f"[resume upload (stream), target_role={target_role}]",
            completion=json.dumps(result.get("validated_profile") or result.get("clarification_question") or ""),
            latency_ms=latency_ms,
            guardrail_fired=False,
            extra={"interface": "api", "endpoint": "/process/stream"},
        )
        yield f"data: {json.dumps({'type': 'result', 'result': result})}\n\n"
    except Exception as e:
        latency_ms = (time.perf_counter() - start_ts) * 1000
        log_request(
            model=MODEL,
            prompt=f"[resume upload (stream), target_role={target_role}]",
            completion=f"ERROR: {e}",
            latency_ms=latency_ms,
            guardrail_fired=False,
            extra={"interface": "api", "endpoint": "/process/stream", "error": True},
        )
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    finally:
        temp_path.unlink(missing_ok=True)
        yield "data: [DONE]\n\n"


@app.post("/process/stream")
async def process_resume_stream(
    file: UploadFile = File(...),
    target_role: str | None = Form(default=None),
):
    """Streaming version of /process — sends stage-progress events over SSE
    so the frontend can show live status instead of a single blocking wait."""
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "resume.pdf").suffix or ".pdf", delete=False) as tmp:
        tmp.write(file.file.read())
        temp_path = Path(tmp.name)

    return StreamingResponse(
        _sse_stage_generator(temp_path, target_role),
        media_type="text/event-stream",
    )


class ChatRequest(BaseModel):
    message: str
    session_id: str
    resume_skills: list[str]
    target_role: str


@app.post("/chat")
def chat(body: ChatRequest) -> dict:
    start_ts = time.perf_counter()

    is_safe, blocked_message = check_input(body.message)  # input guardrail
    if not is_safe:
        latency_ms = (time.perf_counter() - start_ts) * 1000
        log_request(
            model=MODEL,
            prompt=body.message,
            completion=blocked_message,
            latency_ms=latency_ms,
            guardrail_fired=True,
            extra={"interface": "api", "endpoint": "/chat", "session_id": body.session_id},
        )
        return {"answer": blocked_message}

    try:
        answer = run_orchestrator(
            user_message=body.message,
            session_id=body.session_id,
            resume_skills=body.resume_skills,
            target_role=body.target_role,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - start_ts) * 1000
        log_request(
            model=MODEL,
            prompt=body.message,
            completion=answer,
            latency_ms=latency_ms,
            guardrail_fired=False,
            extra={"interface": "api", "endpoint": "/chat", "session_id": body.session_id},
        )
        return {"answer": answer}
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_ts) * 1000
        log_request(
            model=MODEL,
            prompt=body.message,
            completion=f"ERROR: {exc}",
            latency_ms=latency_ms,
            guardrail_fired=False,
            extra={"interface": "api", "endpoint": "/chat", "session_id": body.session_id, "error": True},
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}