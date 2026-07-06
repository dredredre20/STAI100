import json
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from resume_processing.pipeline import load_resume_text, run_resume_intake_pipeline

app = FastAPI(title="STAI100 Resume Intake API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/process")
def process_resume(
    file: UploadFile = File(...),
    target_role: str | None = Form(default=None),
):
    """Non-streaming endpoint — kept for simple/synchronous callers (e.g.
    scripts, testing) that just want the final result in one response."""
    try:
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "resume.pdf").suffix or ".pdf", delete=False) as tmp:
            tmp.write(file.file.read())
            temp_path = Path(tmp.name)
        result = process_uploaded_resume(temp_path, target_role=target_role)
        temp_path.unlink(missing_ok=True)
        return result
    except Exception as exc:  # pragma: no cover - defensive API handling
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _sse_stage_generator(temp_path: Path, target_role: str | None) -> AsyncGenerator[str, None]:
    """Yields SSE events for each pipeline stage, then a final event with the
    full structured result. Unlike the Oakridge bot's token-by-token stream
    (one LLM call generating natural language), this pipeline runs multiple
    sequential stages that mostly produce structured data, not prose — so
    what's streamed here is STAGE PROGRESS, not individual tokens. The SSE
    mechanics (generator + "data: ..." lines + [DONE] sentinel) are the same
    pattern as stream_faq_bot; only the granularity of what's sent differs.
    """
    stages = [
        "Reading resume...",
        "Redacting personal information...",
        "Extracting profile details...",
        "Checking for missing information...",
    ]
    for stage_label in stages:
        yield f"data: {json.dumps({'type': 'stage', 'label': stage_label})}\n\n"

    # The actual pipeline call happens once, after announcing the stages —
    # it isn't feasible to yield mid-pipeline without restructuring
    # run_resume_intake_pipeline() into a generator itself (a bigger change,
    # since its internal steps aren't currently written to yield). This
    # still gives the user real-time feedback that something is happening,
    # even though the stage announcements and actual execution aren't
    # perfectly interleaved yet.
    try:
        resume_text = load_resume_text(str(temp_path))
        result = run_resume_intake_pipeline(
            resume_text,
            verbose=False,
            interactive=False,
            target_role_override=target_role,
        )
        yield f"data: {json.dumps({'type': 'result', 'result': result})}\n\n"
    except Exception as e:
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


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}