# guardrails/input_guardrails.py
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
    # Returns True if the message matches a known injection pattern.
    return bool(_INJECTION_REGEX.search(message))


# ── Off-topic detection ──────────────────────────────────────────────────
TOPIC_CLASSIFIER_PROMPT = """You are a strict topic classifier for a career-readiness advisor chatbot.
The chatbot ONLY discusses: skill gaps, job readiness, resume content, career progress,
target roles (data scientist or cloud engineering), certifications, and course/learning
recommendations related to those roles.

Classify the following user message as either ON_TOPIC or OFF_TOPIC.
Respond with ONLY one word: ON_TOPIC or OFF_TOPIC.

Message: {message}
"""


def is_off_topic(message: str) -> bool:
    """Returns True if the message is classified as unrelated to the
    chatbot's career-readiness domain."""
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
    """Runs both guardrail checks in order (injection first — cheaper and
    catches the more security-sensitive case first). Returns (is_safe, blocked_reason).
    If is_safe is False, the caller should return blocked_reason directly to
    the user WITHOUT ever invoking the ReAct loop."""
    if detect_prompt_injection(message):
        return False, INJECTION_BLOCKED_MESSAGE
    if is_off_topic(message):
        return False, OFF_TOPIC_BLOCKED_MESSAGE
    return True, None