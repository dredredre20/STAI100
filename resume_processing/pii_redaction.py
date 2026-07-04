import re


def redact_resume_pii(text: str) -> str:
    """Redact personal identifiers from raw resume text, pre-LLM-call (checkpoint 1).
    Assumes `text` is a full document — the first line is checked against a
    name-shaped pattern, since resumes conventionally lead with the candidate's name."""
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


def redact_field_pii(text: str) -> str:
    """Redact PII from a single already-extracted structured field value
    (checkpoint 2 — the independent safety net over ResumeProfile output).

    Deliberately does NOT run the name-line detection from redact_resume_pii().
    That heuristic assumes 'this is the first line of a full document', which
    doesn't hold for isolated field values like job titles, degree names, or
    individual skills — applying it here produces false positives (e.g.
    'Quality Assurance Engineer' or 'Linux CLI' match the same Title Case
    pattern as a real name and would be wrongly redacted).

    Email/phone/URL redaction still applies, since those patterns are
    unambiguous regardless of context (a LinkedIn URL is a LinkedIn URL
    whether it appears in raw text or an isolated field)."""
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