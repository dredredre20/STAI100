import re


def redact_resume_pii(text: str) -> str:
    """Redact common personally identifiable information from raw resume text.

    This is intended as a pre-LLM safety step so sensitive contact details are removed before
    the resume is passed to the extraction model.

    Args:
        text: Raw resume text to sanitize.

    Returns:
        A redacted version of the text with emails, phone numbers, URLs, and a leading name removed.
    """

    text = re.sub(
        r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
        '[REDACTED_EMAIL]', text
    )
    text = re.sub(
        r'(?<!\w)(?:\+63[-\s]?|0)9\d{2}[-\.\s]?\d{3,4}[-\.\s]?\d{4}\b',
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


def redact_field_pii(text: str) -> str:
    """Redact PII from a single already-extracted structured field value.

    This is a second-stage safety net applied after structured output has been produced, so
    any PII that survives the first pass is still removed before storage or downstream use.

    Args:
        text: A single field value such as a skill, certification, or education field.

    Returns:
        The redacted field value.
    """

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
    return text