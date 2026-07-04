# Running the Resume Parser

From the repo root, run:

```bash
python -m resume_processing.pipeline path/to/resume.pdf
```

Example:
```bash
python -m resume_processing.pipeline sample_resume_juan_santos.pdf
```

Make sure Ollama is running first (`ollama list` to check).

If it asks a follow-up question about your target role, just type your answer (data scientist or cloud engineer) and hit enter.