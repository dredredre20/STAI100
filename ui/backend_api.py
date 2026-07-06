import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

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
    try:
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "resume.pdf").suffix or ".pdf", delete=False) as tmp:
            tmp.write(file.file.read())
            temp_path = Path(tmp.name)

        result = process_uploaded_resume(temp_path, target_role=target_role)
        temp_path.unlink(missing_ok=True)
        return result
    except Exception as exc:  # pragma: no cover - defensive API handling
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
