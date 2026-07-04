import re
import json
from .schema import ResumeProfile
from .pii_redaction import redact_field_pii
from llm_utils import complete
from config import MODEL

FIX_PROFILE_PROMPT = (
    "You are a resume-parsing assistant. The extracted fields below failed "
    "schema validation.\n\n"
    "Fix the fields so they satisfy the validation rules. For example:\n"
    "- target_role must be exactly 'data_scientist' or 'cloud_engineering', "
    "or null if genuinely unclear from the resume.\n"
    "- years_of_experience must be a number between 0 and 60.\n"
    "- Do not invent skills or certifications not present in the source text.\n\n"
    "Current fields:\n{current_fields}\n\n"
    "Validation error:\n{error_message}\n\n"
    "Return ONLY a corrected JSON object with the same keys. Do not explain anything."
)


def fix_profile_with_llm(fields: dict, error_msg: str, model: str = MODEL) -> dict:
    prompt = FIX_PROFILE_PROMPT.format(
        current_fields=json.dumps(fields, indent=2),
        error_message=error_msg,
    )
    response = complete(
        [{"role": "system", "content": prompt},
         {"role": "user",   "content": "Return the corrected JSON."}],
        model,
    )
    match = re.search(r'\{.*?\}', response, re.DOTALL)
    if match:
        try:
            fixed = json.loads(match.group())
            return {k: fixed.get(k) if fixed.get(k) is not None else v
                     for k, v in fields.items()}
        except json.JSONDecodeError:
            pass
    return fields


def _redact_pii_from_profile(profile: ResumeProfile) -> ResumeProfile:
    """Checkpoint 2 — independent PII safety net over the *structured output*,
    catches PII the LLM invented or missed on the first (raw-text) pass.

    Uses redact_field_pii(), NOT redact_resume_pii() — the latter's name-line
    detection assumes a full document and produces false positives on isolated
    field values (job titles, degree names, individual skills)."""
    data = profile.model_dump()
    for key in ("current_role_category", "education_level"):
        if isinstance(data.get(key), str):
            data[key] = redact_field_pii(data[key])
    data["skills"] = [redact_field_pii(s) for s in data.get("skills", [])]
    data["certifications"] = [redact_field_pii(c) for c in data.get("certifications", [])]
    return ResumeProfile(**data)


def validate_resume_profile(fields: dict, model: str = MODEL, max_retries: int = 3):
    """
    Try to build a ResumeProfile from fields. On each schema-validation
    failure, feed the exact Pydantic error back to the LLM to correct
    itself, then retry. Returns (ResumeProfile | None, final_error_msg).
    """
    relevant = {k: v for k, v in fields.items() if k in ResumeProfile.model_fields}
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            profile = ResumeProfile(**relevant)
            profile = _redact_pii_from_profile(profile)  # checkpoint 2
            return profile, ""
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                relevant = fix_profile_with_llm(relevant, last_error, model)
    return None, last_error