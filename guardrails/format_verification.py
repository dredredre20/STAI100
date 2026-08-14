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
    """Normalize an LLM response to a consistent uppercase form."""
    return (response or "").strip().upper()


def verify_resume_text(resume_text: str, model: str = MODEL) -> bool:
    """Check whether uploaded text appears to be a resume or CV content.

    The function first uses a lightweight keyword heuristic to reject clearly non-resume content,
    then asks the LLM to confirm the classification when the text looks plausible.

    Args:
        resume_text: Raw text extracted from an uploaded document.
        model: Model identifier used for the verification prompt.

    Returns:
        True when the text appears to be a resume/CV; otherwise False.
    """
    if not resume_text or len(resume_text.strip()) < 50:
        return False

    lower_text = resume_text.lower()

    # Use a cheap heuristic to skip unnecessary LLM calls for clearly unrelated content.
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

    # Ask the LLM to make the final yes/no decision on whether the content is a resume.
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
