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
from memory.db_memory import create_session, save_resume_profile
from react.react_agent import run_agent

from guardrails.input_guardrail import check_input
from memory.db_setup import init_db

from config import MODEL, MLFLOW_ENABLED

if MLFLOW_ENABLED:
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
    """Initialize the database when the FastAPI app starts."""
    init_db()


def process_uploaded_resume(uploaded_file: Path | str, target_role: str | None = None) -> dict[str, Any]:
    """Process an uploaded resume file through the intake pipeline.

    Args:
        uploaded_file: Path or string pointing to the uploaded PDF file.
        target_role: Optional target role override to apply during extraction.

    Returns:
        A structured dictionary containing the extracted profile and validation status.
    """
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
    """Persist a validated resume profile to the database and return a new session ID.

    Args:
        result: The structured resume-processing result from the intake pipeline.

    Returns:
        The newly created session ID for the saved profile.
    """
    profile = result["validated_profile"]
    try:
        session_id = create_session()
        save_resume_profile(session_id, profile)
        return session_id
    except Exception:
        traceback.print_exc()   # TEMP DEBUG — shows full traceback in uvicorn terminal
        raise


@app.post("/process")
def process_resume(
    file: UploadFile = File(...),
    target_role: str | None = Form(default=None),
):
    """Process an uploaded resume and return the final structured result.

    This is the non-streaming endpoint used by simple clients that want the complete result in
    a single response.

    Args:
        file: The uploaded resume file.
        target_role: Optional target-role override supplied by the client.

    Returns:
        A dictionary describing the extraction result and any validation state.
    """

    start_ts = time.perf_counter()

    try:
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "resume.pdf").suffix or ".pdf", delete=False) as tmp:
            tmp.write(file.file.read())
            temp_path = Path(tmp.name)
        result = process_uploaded_resume(temp_path, target_role=target_role)
        temp_path.unlink(missing_ok=True)

        if result.get("is_complete"):
            result["session_id"] = persist_completed_profile(result)

        if MLFLOW_ENABLED:
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

        if MLFLOW_ENABLED:
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
    """Yield SSE events for each resume-processing stage and the final result.

    Args:
        temp_path: Path to the temporary uploaded resume file.
        target_role: Optional target-role override passed to the processing pipeline.

    Yields:
        Server-sent events describing progress updates and the final structured output.
    """
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

        if MLFLOW_ENABLED:
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

        if MLFLOW_ENABLED:
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
    """Stream resume-processing progress and results over Server-Sent Events.

    This endpoint is used by the frontend to show live status updates while the backend runs the
    resume intake pipeline.

    Args:
        file: The uploaded resume file.
        target_role: Optional target-role override supplied by the client.

    Returns:
        A StreamingResponse that emits stage updates and the final processing result.
    """
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
    """Handle user chat requests and return the advisor's response.

    The endpoint first applies the input guardrail, then runs the agent loop to generate a
    reply based on the resume context and conversation history.

    Args:
        body: Request payload containing the message, session ID, and resume context.

    Returns:
        A dictionary containing the generated answer.
    """

    start_ts = time.perf_counter()

    is_safe, blocked_message = check_input(body.message) # input guardrail

    if not is_safe: # if message is blocked by guardrail, return the blocked message and log the request
        if MLFLOW_ENABLED:
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
        answer = run_agent(
            user_message=body.message,
            session_id=body.session_id,
            resume_skills=body.resume_skills,
            target_role=body.target_role,
            verbose=False,
        )

        if MLFLOW_ENABLED:
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

        if MLFLOW_ENABLED:
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