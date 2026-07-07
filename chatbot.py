import json
import ollama
from config import MODEL, OLLAMA_BASE_URL
from gap_diff.diff_engine import run_gap_diff
from session_store.persistence import get_session_history
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
"""

SYSTEM_PROMPT_TEMPLATE = """Role: You are a career-readiness advisor agent operating in a strict state machine.
You help users understand their skill gaps for a target role and track their progress.
You must respond with ONLY a valid JSON object. Do not wrap it in markdown code blocks. Do not include any explanatory text before or after the JSON.
{tool_descriptions}
User context:
- session_id: {session_id}
- resume_skills: {resume_skills}
- target_role: {target_role}

Conventions:
- Only call a tool if you need more information. If you already have the answer, set next_action to null.
- Use get_skill_gap's resume_skills/target_role parameters from the User context above unless the user's
  question implies a different target_role.
- Be concise and conversational in your FINAL_ANSWER — this is a chat interface, not a report.
- NEVER output raw JSON, dict syntax, or tool observation data structures in your final_answer.
  Always translate tool results into plain natural-language sentences a person would actually
  say out loud. For example, instead of {{"missing_required": ["ML"], "readiness_score": 60}},
  write something like: "You're missing Machine Learning experience, and your readiness score
  is 60 out of 100." The same applies to course results from search_courses — describe them
  in a sentence or short list of course names, don't paste the raw dict.

You must output your current phase in the STATE.phase field. Valid phases: UNDERSTAND, PLAN, EXECUTE, CRITIQUE, FINAL_ANSWER.

PHASE 1 — UNDERSTAND:
{{"STATE": {{"phase": "UNDERSTAND", "understanding": "..."}}}}

PHASE 2 — PLAN:
{{"STATE": {{"phase": "PLAN", "plan": "..."}}}}

PHASE 3 — EXECUTE:
{{"STATE": {{"phase": "EXECUTE", "plan": "...", "next_action": {{"tool_name": "<tool>", "parameters": {{...}}}}}}}}

PHASE 4 — CRITIQUE:
{{"STATE": {{"phase": "CRITIQUE", "critique": "..."}}}}

PHASE 5 — FINAL_ANSWER:
{{"STATE": {{"phase": "FINAL_ANSWER", "final_answer": "..."}}}}

RULES:
- Flow: UNDERSTAND -> PLAN -> EXECUTE -> CRITIQUE -> (PLAN again or FINAL_ANSWER).
- Start with UNDERSTAND on the first turn.
- After EXECUTE, the system runs the tool and provides an observation; you then output CRITIQUE.
- After CRITIQUE, output PLAN again if more info is needed, or FINAL_ANSWER if you have enough.
- If you already have enough information to answer, skip next_action and go straight to FINAL_ANSWER.
"""


def build_context(history: list) -> str:
    if not history:
        return "\n\n[CURRENT STATE]\nSTART. Please begin with the UNDERSTAND phase.\n\n[EXECUTION HISTORY]\nNone."
    last = history[-1]
    phase = last.get("phase", "")
    if phase == "EXECUTE" and not last.get("next_action"):
        current = "EXECUTE complete with no action. Please output FINAL_ANSWER."
    else:
        state_map = {
            "UNDERSTAND": "UNDERSTAND complete. Please output PLAN.",
            "PLAN": "PLAN complete. Please output EXECUTE with next_action.",
            "EXECUTE": "EXECUTE complete, tool called. Please output CRITIQUE.",
            "TOOL_RESULT": "Tool execution complete. Please output CRITIQUE.",
            "CRITIQUE": "CRITIQUE complete. Please output PLAN (more work needed) or FINAL_ANSWER.",
            "FINAL_ANSWER": "Task finished.",
        }
        current = state_map.get(phase, f"Unknown phase ({phase}).")
    return f"\n\n[CURRENT STATE]\n{current}\n\n[EXECUTION HISTORY]\n{json.dumps(history, indent=2)}"


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
        # Deterministic — no LLM in this path. Pulls saved diff_results
        # straight from persistence.py rather than generating SQL, since a
        # fixed "give me this session's past scores" query doesn't need
        # free-form NL->SQL generation to answer correctly.
        try:
            rows = get_session_history(session_id)
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
            # Chroma store hasn't been built yet — surface a clear message
            # rather than crashing the whole orchestrator turn.
            return f"ERROR: Course search index not available yet ({e})."
        except Exception as e:
            return f"ERROR: search_courses failed: {e}"

    else:
        return f"ERROR: Unknown tool '{tool_name}'"


def format_final_answer(answer: str) -> str:
    """Safety net — if the LLM echoed a tool's raw JSON as its final_answer
    instead of writing prose (a known failure mode on smaller models like
    llama3.1:8b), reformat it into readable text rather than showing the
    user a JSON blob. Falls through untouched if answer is already plain text."""
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


def run_orchestrator(
    user_message: str,
    session_id: str,
    resume_skills: list[str],
    target_role: str,
    max_turns: int = 10,
    verbose: bool = True,
) -> str:
    
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        tool_descriptions=TOOL_DESCRIPTIONS,
        session_id=session_id,
        resume_skills=json.dumps(resume_skills),
        target_role=target_role,
    )

    history = []
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for turn in range(1, max_turns + 1):
        if verbose:
            print(f"\n------ TURN {turn} ------")
        msgs = [{"role": "system", "content": system_prompt + build_context(history)}] + messages[1:]
        try:
            raw = call_llm(msgs)
            data = parse_json(raw)["STATE"]
        except Exception as e:
            if verbose:
                print(f"[Error] {e}")
            return "Sorry, I ran into an error trying to answer that."

        messages.append({"role": "assistant", "content": raw})
        phase = data.get("phase", "")
        if verbose:
            print(f"Phase: {phase}")

        if phase == "UNDERSTAND":
            history.append({"phase": "UNDERSTAND", "understanding": data.get("understanding", "")})
            messages.append({"role": "user", "content": "Acknowledged. Output PLAN."})

        elif phase == "PLAN":
            history.append({"phase": "PLAN", "plan": data.get("plan", "")})
            messages.append({"role": "user", "content": "Acknowledged. Output EXECUTE with next_action."})

        elif phase == "EXECUTE":
            action = data.get("next_action")
            if action:
                if verbose:
                    print(f"Action: {json.dumps(action)}")
                history.append({"phase": "EXECUTE", "plan": data.get("plan", ""), "next_action": action})
                result_str = run_tool(action, session_id, resume_skills, target_role)
                if verbose:
                    print(f"Result: {result_str[:300]}")
                history.append({"phase": "TOOL_RESULT", "result": result_str})
                messages.append({"role": "user", "content": f"Observation: {result_str}. Output CRITIQUE."})
            else:
                history.append({"phase": "EXECUTE", "plan": data.get("plan", ""), "next_action": None})
                messages.append({"role": "user", "content": "No action needed. Output FINAL_ANSWER."})

        elif phase == "CRITIQUE":
            history.append({"phase": "CRITIQUE", "critique": data.get("critique", "")})
            messages.append({"role": "user", "content": "Acknowledged. Output PLAN or FINAL_ANSWER."})

        elif phase == "FINAL_ANSWER":
            answer = data.get("final_answer", "")
            history.append({"phase": "FINAL_ANSWER", "final_answer": answer})
            return format_final_answer(answer)

        else:
            messages.append({"role": "user", "content": f"Invalid phase '{phase}'."})

    return "I wasn't able to reach an answer within the allowed number of steps."