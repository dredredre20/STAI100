import json
import ollama
from config import MODEL, OLLAMA_BASE_URL
from gap_diff.diff_engine import run_gap_diff
from session_store.persistence import get_progress_history, get_latest_resume_profile
from resource_retrieval.retrieval import search_courses

TOOL_DESCRIPTIONS = """
Available Tools:
- get_skill_gap[resume_skills, target_role] : Compares the user's skills against
  aggregated job posting requirements for their target role. Returns matched/missing
  required and preferred skills, plus a readiness_score (0-100). Use this when the
  user asks what skills they need, what they're missing, or how ready they are.
- get_progress_history[] : Returns this session's saved readiness scores from
  past resume checks, most-recent first (each with a readiness_score, timestamp,
  and missing_required/missing_preferred counts). Use this for anything about the
  user's history or progress over time (e.g. "have I improved?", "what was my
  score before?"), NOT for a fresh skill-gap comparison happening right now
  (use get_skill_gap for that).
- search_courses[query] : Semantically searches a course catalog for learning resources
  relevant to a skill or topic. Use this when the user asks how to close a skill gap,
  wants course/training recommendations, or asks "how do I learn X." Automatically
  restricted to courses for the user's target_role.
- get_user_profile[] : Returns this session's most recently saved resume
  profile — skills, certifications, education level, years of experience,
  and target_role. Use this for general questions about the user's
  background (e.g. "what skills do I have listed?", "what's my education?",
  "what certifications did I upload?"), NOT for readiness scores or progress
  over time (use get_progress_history for that) and NOT for a fresh gap
  comparison against job postings (use get_skill_gap for that).
"""

SYSTEM_PROMPT_TEMPLATE = """Role: You are a career-readiness advisor agent.
You help users understand their skill gaps for a target role and track their progress.
{tool_descriptions}
User context:
- session_id: {session_id}
- resume_skills: {resume_skills}
- target_role: {target_role}

CONVERSATION HISTORY (rolling summary + recent turns from earlier in this session):
{conversation_history}
---

You must respond with ONLY a valid JSON object. Do not wrap it in markdown code blocks.
Do not include any explanatory text before or after the JSON.

Every response has this shape:
{{"thought": "...", "action": {{"tool_name": "...", "parameters": {{...}}}} or null, "final_answer": "..." or null}}

Exactly one of "action" or "final_answer" must be non-null. The other must be null.

Use the "thought" field to actually reason step by step before deciding, covering:
1. What is the user asking for, specifically? Use CONVERSATION HISTORY above to
   understand follow-ups like "what about the preferred ones?" or "has that improved?"
2. What do I already know from User context, CONVERSATION HISTORY, or prior
   Observations in this turn? Is it enough to answer, or is something missing?
3. If something is missing, which single tool fills that gap, and what parameters
   does it need? (Only call a tool if you genuinely need its output — don't call a
   tool you already have the answer from, including from CONVERSATION HISTORY.)
4. If I already have enough information, skip the tool: set "action" to null and
   write the final answer instead.
Keep "thought" as short as it can be while still covering the above — a few
sentences is normally enough, don't pad it.

Conventions:
- Use get_skill_gap's resume_skills/target_role from the User context above unless
  the user's question implies a different target_role.
- After a tool Observation is added to the conversation, re-do this same reasoning:
  decide if the observation is now enough to answer (go to final_answer) or if
  another tool call is needed (only if truly necessary — avoid loops).
- Be concise and conversational in final_answer — this is a chat interface, not a report.
- NEVER output raw JSON, dict syntax, or tool observation data structures in
  final_answer. Always translate tool results into plain natural-language sentences
  a person would actually say out loud. For example, instead of
  {{"missing_required": ["ML"], "readiness_score": 60}}, write something like:
  "You're missing Machine Learning experience, and your readiness score is 60 out
  of 100." The same applies to course results from search_courses — describe them
  in a sentence or short list of course names, don't paste the raw dict.
"""


from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

_session_histories: dict[str, list] = {}

KEEP_LAST = 4  # most recent messages (2 turns) kept verbatim; older gets summarized


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

    older  = messages[:-keep_last]
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


def call_llm(messages: list) -> str:
    client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.chat(model=MODEL, messages=messages, format="json")
    return response['message']['content'].strip()


def parse_json(text: str) -> dict:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def run_tool(action: dict, session_id: str, resume_skills: list[str], target_role: str) -> str:
    tool_name = action.get("tool_name", "")
    params = action.get("parameters", {})

    if tool_name == "get_skill_gap":
        role = params.get("target_role", target_role)
        result = run_gap_diff(resume_skills, role)
        return json.dumps({
            "target_role": result.target_role,
            "readiness_score": result.readiness_score,
            "missing_required": [m.skill for m in result.missing_required[:15]],
            "missing_preferred": [m.skill for m in result.missing_preferred[:10]],
            "matched_required_count": len(result.matched_required),
            "note": "missing lists truncated to top items by frequency" if len(result.missing_required) > 15 else None,
        })

    elif tool_name == "get_progress_history":
        try:
            rows = get_progress_history(session_id)
            if not rows:
                return json.dumps({"history": [], "note": "No past sessions found for this session_id."})
            trimmed = [
                {
                    "created_at": row.get("created_at"),
                    "readiness_score": row.get("readiness_score"),
                    "missing_required_count": len(json.loads(row.get("missing_required") or "[]")),
                    "missing_preferred_count": len(json.loads(row.get("missing_preferred") or "[]")),
                }
                for row in rows
            ]
            return json.dumps({"history": trimmed})
        except Exception as e:
            return f"ERROR: get_progress_history failed: {e}"

    elif tool_name == "search_courses":
        query = params.get("query", "")
        if not query:
            return "ERROR: Missing 'query' parameter for search_courses."
        try:
            results = search_courses(query, target_role=target_role, top_k=5)
            if not results:
                return json.dumps({"courses": [], "note": "No matching courses found."})
            return json.dumps({"courses": results})
        except FileNotFoundError as e:
            return f"ERROR: Course search index not available yet ({e})."
        except Exception as e:
            return f"ERROR: search_courses failed: {e}"
        
    elif tool_name == "get_user_profile":
        try:
            profile = get_latest_resume_profile(session_id)
            if not profile:
                return json.dumps({"note": "No resume profile found for this session."})
            return json.dumps({
                "target_role": profile.get("target_role"),
                "current_role_category": profile.get("current_role_category"),
                "years_of_experience": profile.get("years_of_experience"),
                "skills": json.loads(profile.get("skills") or "[]"),
                "certifications": json.loads(profile.get("certifications") or "[]"),
                "education_level": profile.get("education_level"),
            })
        except Exception as e:
            return f"ERROR: get_user_profile failed: {e}"

    else:
        return f"ERROR: Unknown tool '{tool_name}'"
    



def format_final_answer(answer: str) -> str:
    """Safety net — if the LLM echoed a tool's raw JSON as its final_answer
    instead of writing prose, reformat it into readable text."""
    try:
        data = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        return answer

    if not isinstance(data, dict):
        return answer

    parts = []
    if "readiness_score" in data:
        parts.append(f"Your readiness score is {data['readiness_score']}/100.")
    if data.get("missing_required"):
        parts.append(f"Missing required skills: {', '.join(data['missing_required'])}.")
    if data.get("missing_preferred"):
        parts.append(f"Missing preferred skills: {', '.join(data['missing_preferred'])}.")
    if data.get("courses"):
        titles = [c.get("title", "Unknown course") for c in data["courses"]]
        parts.append("Recommended courses: " + ", ".join(titles) + ".")
    if data.get("recommendation"):
        parts.append(data["recommendation"])
    return " ".join(parts) if parts else answer


def run_agent(
    user_message: str,
    session_id: str,
    resume_skills: list[str],
    target_role: str,
    max_turns: int = 10,
    verbose: bool = True,
) -> str:

    session_history = _session_histories.get(session_id, [])
    session_history = summarize_old_history(session_history)
    conversation_history_text = format_chat_history(session_history)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        tool_descriptions=TOOL_DESCRIPTIONS,
        session_id=session_id,
        resume_skills=json.dumps(resume_skills),
        target_role=target_role,
        conversation_history=conversation_history_text,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    final_answer = None
    for turn in range(1, max_turns + 1):
        if verbose:
            print(f"\n------ TURN {turn} ------")
        try:
            raw = call_llm(messages)
            data = parse_json(raw)
        except Exception as e:
            if verbose:
                print(f"[Error] {e}")
            final_answer = "Sorry, I ran into an error trying to answer that."
            break

        messages.append({"role": "assistant", "content": raw})
        if verbose:
            print(f"Thought: {data.get('thought', '')}")

        action = data.get("action")
        if action:
            if verbose:
                print(f"Action: {json.dumps(action)}")
            result_str = run_tool(action, session_id, resume_skills, target_role)
            if verbose:
                print(f"Result: {result_str[:300]}")
            messages.append({"role": "user", "content": f"Observation: {result_str}"})
        else:
            final_answer = format_final_answer(data.get("final_answer", ""))
            break

    if final_answer is None:
        final_answer = "I wasn't able to reach an answer within the allowed number of steps."

    # Persist this turn into cross-call session memory.
    session_history.append(HumanMessage(content=user_message))
    session_history.append(AIMessage(content=final_answer))
    _session_histories[session_id] = summarize_old_history(session_history)

    return final_answer