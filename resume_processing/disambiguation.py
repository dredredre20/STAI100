import json
from .schema import RESUME_FIELD_DEFINITIONS
from llm_utils import complete
from config import MODEL

def get_missing_fields(filled_fields: dict) -> list:
    """Only target_role blocks completion — everything else is optional context."""
    return [
        name for name, info in RESUME_FIELD_DEFINITIONS.items()
        if info.get("required", False) and filled_fields.get(name) is None
    ]


CLARIFICATION_PROMPT = (
    "You are a helpful career-readiness assistant.\n\n"
    "The user uploaded a resume. Information already extracted: {known_fields}\n"
    "You need to ask about: target_role — which career transition they're "
    "aiming for (data scientist or cloud engineering).\n\n"
    "Write ONE short, natural, friendly question to collect this. "
    "Do not use technical terms like 'field' or 'slot'. "
    "Respond with ONLY the question."
)

# function to generate a clarification question for the user to specify their target role
def generate_target_role_clarification(filled_fields: dict, model: str = MODEL) -> str:
    known = {k: v for k, v in filled_fields.items() if v is not None and k != "target_role"}
    prompt = CLARIFICATION_PROMPT.format(
        known_fields=json.dumps(known) if known else "nothing yet"
    )
    return complete(
        [{"role": "system", "content": prompt},
         {"role": "user",   "content": "Please ask me."}],
        model
    ).strip()