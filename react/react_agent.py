import json
import ollama
from config import MODEL, OLLAMA_BASE_URL
from gap_diff.diff_engine import run_gap_diff
from memory.db_memory import get_latest_resume_profile
from memory.conversation_memory import (
    get_formatted_session_history,
    save_session_turn
)


TOOL_DESCRIPTIONS = """
Available Tools:
- get_skill_gap[resume_skills, target_role] : Compares the user's skills against
  aggregated job posting requirements for their target role. Returns matched/missing
  required and preferred skills, plus a readiness_score (0-100). Use this when the
  user asks what skills they need, what they're missing, or how ready they are.
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
"""

# function to run a tool based on the action dict returned by the model, return json result as string
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
    


# function to format the final answer from the model, converting raw JSON into readable text
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


# function to run the agent loop, calling the LLM and tools iteratively until a final answer is reached or max turns exceeded
def run_agent(
    user_message: str,
    session_id: str,
    resume_skills: list[str],
    target_role: str,
    max_turns: int = 10,
    verbose: bool = True,
) -> str:

    # Retrieve & format memory 
    session_history, conversation_history_text = get_formatted_session_history(session_id)

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

    # Persist turn back to memory
    save_session_turn(session_id, session_history, user_message, final_answer)

    return final_answer


def call_llm(messages: list) -> str:
    client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.chat(model=MODEL, messages=messages, format="json")
    return response['message']['content'].strip()


def parse_json(text: str) -> dict:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)
