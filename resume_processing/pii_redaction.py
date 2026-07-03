import re

def redact_resume_pii(text: str) -> str:
    """Redact personal identifiers from raw resume text, pre-LLM-call (checkpoint 1)."""
    text = re.sub(
        r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
        '[REDACTED_EMAIL]', text
    )

    text = re.sub(
        r'\b(?:\+63[-\s]?|0)9\d{2}[-\.\s]?\d{3,4}[-\.\s]?\d{4}\b',
        '[REDACTED_PHONE]', text
    )

    text = re.sub(
        r'\bhttps?://(www\.)?(linkedin\.com|github\.com)/\S+',
        '[REDACTED_URL]', text
    )

    lines = text.split("\n")
    if lines and re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+){1,3}$', lines[0].strip()):
        lines[0] = '[REDACTED_NAME]'
    text = "\n".join(lines)

    return text