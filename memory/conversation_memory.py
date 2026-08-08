import ollama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from config import MODEL, OLLAMA_BASE_URL

# Configuration
KEEP_LAST = 4  # most recent messages (2 turns) kept verbatim; older gets summarized

# Global session store
_session_histories: dict[str, list] = {}


def _plain_llm_call(prompt: str) -> str:
    """Free-text (non-JSON) completion, used only for summarizing history."""
    client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].strip()


def summarize_old_history(messages: list, keep_last: int = KEEP_LAST) -> list:
    """
    Compact the message list progressively so it never grows unboundedly.
    Returns at most one SystemMessage (the rolling summary) followed by the
    most recent verbatim turns.
    """
    if not messages or len(messages) <= keep_last:
        return messages

    older = messages[:-keep_last]
    recent = messages[-keep_last:]

    existing_summary = next((m.content for m in older if isinstance(m, SystemMessage)), "")
    summarizable = [m for m in older if not isinstance(m, SystemMessage)]

    if not summarizable:
        return [SystemMessage(content=existing_summary)] + recent if existing_summary else recent

    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in summarizable
    )

    summary_prompt = (
        "Progressively update the summary of the conversation between a user and a "
        "career-readiness advisor bot.\n"
        f"Current summary: {existing_summary or 'None'}\n\n"
        f"New turns to incorporate:\n{history_text}\n\n"
        "Generate a concise, updated summary capturing the user's target role, "
        "any skill gaps or readiness scores discussed, and what advice was given."
    )

    try:
        summary = _plain_llm_call(summary_prompt)
    except Exception as e:
        print(f"[MEMORY WARNING] Summarization failed, falling back to recent turns: {e}")
        return recent

    return [SystemMessage(content=summary)] + recent


def format_chat_history(messages: list) -> str:
    """Render the rolling history (summary + recent verbatim turns) as plain
    text suitable for injection into the system prompt."""
    if not messages:
        return "No previous conversation this session."

    lines = []
    for m in messages:
        if isinstance(m, SystemMessage):
            lines.append(f"[Summary of earlier conversation]\n{m.content}")
        elif isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content}")
    return "\n".join(lines)


def get_formatted_session_history(session_id: str) -> tuple[list, str]:
    """Retrieves, summarizes, and formats the active session history for the prompt."""
    session_history = _session_histories.get(session_id, [])
    session_history = summarize_old_history(session_history)
    conversation_history_text = format_chat_history(session_history)
    return session_history, conversation_history_text


def save_session_turn(session_id: str, session_history: list, user_message: str, final_answer: str):
    """Appends the latest interaction turn to session history and updates storage."""
    session_history.append(HumanMessage(content=user_message))
    session_history.append(AIMessage(content=final_answer))
    _session_histories[session_id] = summarize_old_history(session_history)