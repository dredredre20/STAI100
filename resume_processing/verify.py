import re
from llm_utils import complete
from config import MODEL

# prompt to verify whether uploaded file is resume/cv
RESUME_VERIFICATION_PROMPT = (
    "You are a resume-document classifier. The user uploaded text extracted "
    "from a PDF. Determine whether the text is from a real resume or CV "
    "that contains job history, education, skills, and career details.\n\n"
    "If the text is a resume or CV, answer EXACTLY: YES\n"
    "If the text is not a resume/CV (for example, an article, email, poem, invoice, "
    "random notes, or blank page), answer EXACTLY: NO\n\n"
    "Respond with ONLY YES or NO."
)


def _normalize_response(response: str) -> str:
    return (response or "").strip().upper()


def verify_resume_text(resume_text: str, model: str = MODEL) -> bool:
    """Return True only if the uploaded text appears to be a resume/CV."""
    if not resume_text or len(resume_text.strip()) < 50:
        return False
    
    lower_text = resume_text.lower()

    # heuristic check for common resume keywords to avoid unnecessary LLM calls
    keywords = [
        "experience",
        "skills",
        "education",
        "certifications",
        "projects",
        "professional",
        "work history",
        "objective",
        "summary",
    ]
    if not any(keyword in lower_text for keyword in keywords):
        return False

    response = complete(
        [
            {"role": "system", "content": RESUME_VERIFICATION_PROMPT},
            {"role": "user", "content": resume_text},
        ],
        model,
    )
    answer = _normalize_response(response)
    match = re.search(r"\b(YES|NO)\b", answer)
    return bool(match and match.group(1) == "YES")
