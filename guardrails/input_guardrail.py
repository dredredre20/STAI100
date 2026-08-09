import re
import json
import ollama
from config import MODEL, OLLAMA_BASE_URL

# ── Prompt injection detection ───────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (all )?(previous|prior|above) instructions",
    r"forget (everything|all)( you)? (were|was|are|is) told",
    r"forget (everything|all) (you|i) (said|told)",
    r"new instructions?:",
    r"reveal your (prompt|instructions|system prompt)",
    r"^you are now\b",  
    r"pretend (you are|to be) (an? )?(unrestricted|jailbroken|uncensored)",
    r"\bDAN\b",  # common jailbreak persona name
]

_INJECTION_REGEX = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def detect_prompt_injection(message: str) -> bool:
    """Check whether a user message attempts prompt injection or instruction override.

    Args:
        message: User text to inspect.

    Returns:
        True when the message appears to contain prompt-injection patterns; otherwise False.
    """
    return bool(_INJECTION_REGEX.search(message))


# ── Off-topic detection ──────────────────────────────────────────────────
TOPIC_CLASSIFIER_PROMPT = """You are a topic classifier for a career-readiness advisor chatbot.
The chatbot ONLY discusses: skill gaps, job readiness, resume content, career progress,
target roles, certifications, and things related to the user profile.

Classify the following user message as either ON_TOPIC or OFF_TOPIC.
Respond with ONLY one word: ON_TOPIC or OFF_TOPIC.

Message: {message}
"""


def is_off_topic(message: str) -> bool:
    """Classify whether a user message is outside the supported career-readiness domain.

    Args:
        message: User text to classify.

    Returns:
        True when the message is judged off-topic; otherwise False.
    """
    client = ollama.Client(host=OLLAMA_BASE_URL)
    prompt = TOPIC_CLASSIFIER_PROMPT.format(message=message)
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    verdict = response["message"]["content"].strip().upper()
    return "OFF_TOPIC" in verdict


# ── Combined check ────────────────────────────────────────────────────────
INJECTION_BLOCKED_MESSAGE = (
    "I can't follow instructions embedded in a message like that. "
    "I'm here to help with your career readiness — feel free to ask about "
    "your skill gaps, progress, or course recommendations."
)

OFF_TOPIC_BLOCKED_MESSAGE = (
    "I'm only able to help with career readiness topics — things like your "
    "skill gaps, target role progress, certifications, or course recommendations. "
    "Could you rephrase your question around one of those?"
)


def check_input(message: str) -> tuple[bool, str | None]:
    """Run the input-safety checks in order and return a safe/blocked decision.

    The prompt-injection check is applied first because it is cheaper and more security-sensitive.
    If either check fails, the caller should block the request and avoid running the ReAct loop.

    Args:
        message: User text to validate.

    Returns:
        A tuple of (is_safe, blocked_reason), where blocked_reason is None when the input is allowed.
    """
    if detect_prompt_injection(message):
        return False, INJECTION_BLOCKED_MESSAGE
    if is_off_topic(message):
        return False, OFF_TOPIC_BLOCKED_MESSAGE
    return True, None