from guardrails.pii_redaction import redact_resume_pii
from .field_extraction import extract_resume_fields
from .disambiguation import get_missing_fields, generate_target_role_clarification
from .resume_schema import RESUME_FIELD_DEFINITIONS
from .validate_output import validate_resume_profile
from guardrails.format_verification import verify_resume_text
from llm_utils import complete
from config import MODEL
from pypdf import PdfReader


def load_resume_text(pdf_path: str) -> str:
    """Extract and concatenate text from a PDF resume file.

    Args:
        pdf_path: Path to the PDF file to read.

    Returns:
        A single string containing the extracted text from all pages.
    """
    reader = PdfReader(pdf_path)
    page_texts = []
    for page in reader.pages:
        # Some PDF pages may not contain extractable text, so skip empty results.
        text = page.extract_text()
        if text:
            page_texts.append(text)
    return "\n".join(page_texts).strip()


SEP = "-" * 54


def run_resume_intake_pipeline(
    resume_text: str,
    model: str = MODEL,
    verbose: bool = True,
    interactive: bool = True,
    target_role_override: str | None = None,
) -> dict:
    
    """Run the full resume intake workflow from raw text to validated structured output.

    The pipeline performs a series of checks and transformations:
    1. verify the upload appears to be a resume,
    2. redact personally identifiable information,
    3. extract structured fields with an LLM,
    4. check for missing required fields,
    5. ask for clarification when needed,
    6. validate and repair the final structured profile.

    Args:
        resume_text: Raw resume text or extracted PDF content.
        model: Model identifier to use for LLM calls.
        verbose: Whether to print progress information to the console.
        interactive: Whether to prompt the user for clarification in a chat loop.
        target_role_override: Optional override for the target role when known upfront.

    Returns:
        A dictionary containing the processed fields, missing-field state, validation output,
        and any clarification information.
    """
    # [1] Resume verification — reject non-resume uploads early.
    if verbose: print(f"{SEP}\n[1] Resume verification")
    if not verify_resume_text(resume_text, model):
        if verbose: print("    => upload rejected: not a resume")
        return {
            "input_length": len(resume_text),
            "fields": {name: None for name in RESUME_FIELD_DEFINITIONS},
            "missing_fields": list(RESUME_FIELD_DEFINITIONS.keys()),
            "is_complete": False,
            "needs_clarification": False,
            "clarification_question": None,
            "validated_profile": None,
            "validation_error": "Uploaded document does not appear to be a resume/CV.",
        }

    # [2] PII Redaction — checkpoint 1, pre-LLM ─────────────────────────
    if verbose: print(f"{SEP}\n[2] PII Redaction")
    clean_text = redact_resume_pii(resume_text)
    if verbose: print("    => done")

    # [3] Field Extraction ────────────────────────────────────────────
    if verbose: print("[3] Field Extraction")
    fields = extract_resume_fields(clean_text, model)
    if verbose: print(f"    => {fields}")

    if target_role_override:
        fields["target_role"] = target_role_override

    # [3] Completeness Check ──────────────────────────────────────────
    if verbose: print("[3] Completeness Check")
    missing = get_missing_fields(fields)
    if verbose: print(f"    => Missing required: {missing or 'none'}")

    clarification_question = None

    if missing and not interactive:
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

    # [4] Output Validation ───────────────────────────────────────────
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