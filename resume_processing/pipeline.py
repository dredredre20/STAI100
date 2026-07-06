from .pii_redaction import redact_resume_pii
from .extract_fields import extract_resume_fields
from .disambiguation import get_missing_fields, generate_target_role_clarification
from .validate_output import validate_resume_profile
from llm_utils import complete
from config import MODEL
from pypdf import PdfReader


def load_resume_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() for page in reader.pages)


SEP = "-" * 54


def run_resume_intake_pipeline(
    resume_text: str,
    model: str = MODEL,
    verbose: bool = True,
    interactive: bool = True,
    target_role_override: str | None = None,
) -> dict:
    """5-stage resume intake pipeline: parse (caller-provided) → PII redact
    → extract → disambiguate (target_role only) → validate."""

    # [1] PII Redaction — checkpoint 1, pre-LLM ─────────────────────────
    if verbose: print(f"{SEP}\n[1] PII Redaction")
    clean_text = redact_resume_pii(resume_text)
    if verbose: print("    => done")

    # [2] Field Extraction ────────────────────────────────────────────
    if verbose: print("[2] Field Extraction")
    fields = extract_resume_fields(clean_text, model)
    if verbose: print(f"    => {fields}")

    if target_role_override:
        fields["target_role"] = target_role_override

    # [3] Completeness Check ──────────────────────────────────────────
    if verbose: print("[3] Completeness Check")
    missing = get_missing_fields(fields)
    if verbose: print(f"    => Missing required: {missing or 'none'}")

    # [4] Clarification Loop — target_role only, capped at 3 rounds ────
    if verbose: print("[4] Clarification Loop")
    MAX_ROUNDS = 3
    clarification_question = None
    for round_num in range(1, MAX_ROUNDS + 1):
        missing = get_missing_fields(fields)
        if not missing:
            break
        clarification_question = generate_target_role_clarification(fields, model)
        if not interactive:
            if verbose: print(f"    Non-interactive: cannot clarify '{clarification_question}' — stopping")
            break
        print(f"    Bot [{round_num}/{MAX_ROUNDS}]: {clarification_question}")
        user_answer = input("    You: ").strip()
        if user_answer:
            reextracted = extract_resume_fields(user_answer, model)
            if reextracted.get("target_role"):
                fields["target_role"] = reextracted["target_role"]

    missing = get_missing_fields(fields)
    is_complete = not missing
    if verbose:
        print(f"    => {'All required fields filled' if is_complete else f'Still missing: {missing}'}")

    # [5] Output Validation ───────────────────────────────────────────
    if verbose: print("[5] Output Validation")
    validated_profile = None
    validation_error = None
    if is_complete:
        profile, err = validate_resume_profile(fields, model)
        if profile:
            validated_profile = profile.model_dump()
            if verbose: print(f"    => Valid: {validated_profile}")
        else:
            validation_error = err
            is_complete = False
            if verbose: print(f"    => Validation failed: {err}")

    return {
        "input_length": len(resume_text),
        "fields": fields,
        "missing_fields": missing,
        "is_complete": is_complete,
        "clarification_question": clarification_question,
        "validated_profile": validated_profile,
        "validation_error": validation_error,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m resume_processing.pipeline <path_to_resume.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    resume_text = load_resume_text(pdf_path)
    result = run_resume_intake_pipeline(resume_text)

    print(f"\n{SEP}\nFINAL RESULT\n{SEP}")
    print(f"Complete: {result['is_complete']}")
    print(f"Validated profile: {result['validated_profile']}")
    if result['validation_error']:
        print(f"Validation error: {result['validation_error']}")