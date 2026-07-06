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
    → extract → disambiguate (target_role only) → validate.

    interactive: if True (default, CLI usage), the clarification loop uses
        input() to prompt the user in the terminal when target_role is
        missing — this ONLY works when there's an actual terminal attached.
        If False (e.g. called from a web API, where there's no stdin to
        read from), the clarification loop is skipped entirely — input()
        would hang a web request indefinitely waiting for input that can
        never arrive. Instead, if target_role is still missing after
        applying target_role_override, the function returns immediately
        with is_complete=False and needs_clarification=True, so the caller
        (e.g. a FastAPI endpoint) can hand the clarification_question back
        to its own frontend and let the user answer through a normal form
        field, then call this function again with target_role_override set.

    target_role_override: if provided, used to directly fill in a missing
        target_role without going through the clarification loop at all —
        e.g. when the frontend already collected it via a dropdown before
        calling the API, as in the Streamlit sidebar's target_role selector.
    """

    # [1] PII Redaction — checkpoint 1, pre-LLM ─────────────────────────
    if verbose: print(f"{SEP}\n[1] PII Redaction")
    clean_text = redact_resume_pii(resume_text)
    if verbose: print("    => done")

    # [2] Field Extraction ────────────────────────────────────────────
    if verbose: print("[2] Field Extraction")
    fields = extract_resume_fields(clean_text, model)
    if verbose: print(f"    => {fields}")

    # Apply an explicit override before checking completeness, regardless
    # of interactive mode — this lets a caller who already knows the
    # target_role (e.g. from a frontend dropdown) skip clarification
    # entirely, even in interactive/CLI mode.
    if target_role_override:
        fields["target_role"] = target_role_override

    # [3] Completeness Check ──────────────────────────────────────────
    if verbose: print("[3] Completeness Check")
    missing = get_missing_fields(fields)
    if verbose: print(f"    => Missing required: {missing or 'none'}")

    clarification_question = None

    if missing and not interactive:
        # Non-interactive mode (e.g. API): do NOT call input(). Return
        # immediately so the caller can collect the missing field through
        # its own UI and retry with target_role_override set.
        clarification_question = generate_target_role_clarification(fields, model)
        if verbose: print(f"    => Non-interactive mode, cannot prompt. Needs: {missing}")
        return {
            "input_length": len(resume_text),
            "fields": fields,
            "missing_fields": missing,
            "is_complete": False,
            "needs_clarification": True,
            "clarification_question": clarification_question,
            "validated_profile": None,
            "validation_error": None,
        }

    # [4] Clarification Loop — target_role only, capped at 3 rounds ────
    # Only reached in interactive mode with missing fields — safe to call
    # input() here since interactive=True implies a real terminal caller.
    if missing and interactive:
        if verbose: print("[4] Clarification Loop")
        MAX_ROUNDS = 3
        for round_num in range(1, MAX_ROUNDS + 1):
            missing = get_missing_fields(fields)
            if not missing:
                break
            clarification_question = generate_target_role_clarification(fields, model)
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
        "needs_clarification": False,
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
    result = run_resume_intake_pipeline(resume_text)  # interactive=True by default for CLI use

    print(f"\n{SEP}\nFINAL RESULT\n{SEP}")
    print(f"Complete: {result['is_complete']}")
    print(f"Validated profile: {result['validated_profile']}")
    if result['validation_error']:
        print(f"Validation error: {result['validation_error']}")