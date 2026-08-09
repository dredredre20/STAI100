import json
import ollama
from config import MODEL, OLLAMA_BASE_URL
from memory.db_memory import (
    get_latest_resume_profile,
    update_skills
)
from memory.conversation_memory import (
    get_formatted_session_history,
    save_session_turn
)
from ds_integration.ingest_job_postings import job_collection
from ds_integration.job_fit_prediction import predict_job_matches, format_resume_string
from ds_integration.job_search import get_all_postings, search_postings
from ds_integration.skill_gap import get_skill_gap_analysis

# Tool descriptions for the agent to use in its system prompt, each tool has an explanation of its purpose and when to use it.
TOOL_DESCRIPTIONS = """
Available Tools:
- get_top_matches[] : Ranks all stored job postings against the user's resume
  using semantic similarity, returning the top matches readiness tiers 
  (Ready to Apply / Near Match / Long-Term Upskilling).
  Use this when the user asks what jobs they're qualified for, what they
  should apply to, or wants an overview of their best-fit postings.

- get_skill_gap[company, title] : Retrieves a specific job posting (by company
  and/or title — at least one is required) and compares it against the user's
  resume. Returns matched skills and gaps, with the response style adapted to
  fit tier (deal-breaker gaps for Near Match, a 6-12 month roadmap for
  Long-Term Upskilling, etc). Use this when the user names a specific company
  or role and asks what they're missing for it, or asks for a detailed
  breakdown beyond a similarity score.

- search_posting[query] : Semantic search over stored job postings for a
  fuzzy/descriptive query (e.g. "AI engineering roles", "remote data roles").
  Returns matching postings with title/company/link and similarity. Use this
  when the user doesn't name an exact company/title but describes what
  they're looking for, or wants to browse rather than get a specific fit
  analysis.

- update_skills[new_skills, new_certifications] : Adds new skills and/or
  certifications to the user's saved resume profile (e.g. after they mention
  completing a course, earning a certification, or learning a new skill).
  Both parameters are optional lists of strings — pass only what's new, not
  the full existing list. Use this when the user says something like "I just
  got AWS certified" or "I learned Terraform" — NOT for correcting mistakes
  in their profile (that would need a different tool) and not for one-off
  skill questions (use get_user_profile or get_skill_gap for those).

- get_user_profile[] : Returns this session's most recently saved resume
  profile — skills, certifications, education level, years of experience,
  and target_role. Use this for general questions about the user's
  background (e.g. "what skills do I have listed?", "what's my education?",
  "what certifications did I upload?")
"""

# Prompt template for the system message that contains context and instructions for the agent.
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
- After a tool Observation is added to the conversation, re-do this same reasoning:
  decide if the observation is now enough to answer (go to final_answer) or if
  another tool call is needed (only if truly necessary — avoid loops).
- Be concise and conversational in final_answer — this is a chat interface, not a report.
"""


def _get_resume_profile_dict(session_id: str) -> dict:
    """Load the latest saved resume profile for a session in the shape expected by the embedding helpers.

    Args:
        session_id: Unique identifier for the current conversation.

    Returns:
        A dictionary containing a "fields" key with the resume attributes required by
        format_resume_string(), or an empty dictionary when no profile is available.
    """
    profile = get_latest_resume_profile(session_id)
    if not profile:
        return {}
    return {
        "fields": {
            "current_role_category": profile.get("current_role_category"),
            "target_role": profile.get("target_role"),
            "years_of_experience": profile.get("years_of_experience"),
            "skills": json.loads(profile.get("skills") or "[]"),
            "certifications": json.loads(profile.get("certifications") or "[]"),
            "education_level": profile.get("education_level"),
        }
    }


def run_tool(action: dict, session_id: str, resume_skills: list[str], target_role: str) -> str:
    """Execute the requested tool and return its result as a JSON string.

    Args:
        action: Dictionary with "tool_name" and "parameters" keys describing the tool call.
        session_id: Unique identifier for the current conversation.
        resume_skills: List of skills from the saved resume profile.
        target_role: Target role from the saved resume profile.

    Returns:
        A JSON-formatted string containing either the tool output or an error message.
    """

    tool_name = action.get("tool_name", "")
    params = action.get("parameters", {})

    if tool_name == "get_user_profile":
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

    elif tool_name == "update_skills":
        try:
            new_skills = params.get("new_skills", [])
            new_certifications = params.get("new_certifications", [])
            if not new_skills and not new_certifications:
                return json.dumps({"error": "update_skills requires at least one new skill or certification."})

            result = update_skills(
                session_id=session_id,
                new_skills=new_skills,
                new_certifications=new_certifications,
            )
            return json.dumps(result)
        except Exception as e:
            return f"ERROR: update_skills failed: {e}"

    elif tool_name == "get_top_matches":
        try:
            profile_dict = _get_resume_profile_dict(session_id)
            if not profile_dict:
                return json.dumps({"note": "No resume profile found for this session."})
            resume_text = format_resume_string(profile_dict)

            all_postings = get_all_postings()
            if not all_postings:
                return json.dumps({"note": "No job postings are currently stored."})

            results_df = predict_job_matches(resume_text, all_postings)
            top_matches = results_df.head(5)[["title", "company", "link", "tier"]].to_dict(orient="records")
            return json.dumps({"top_matches": top_matches})
        except Exception as e:
            return f"ERROR: get_top_matches failed: {e}"

    elif tool_name == "get_skill_gap":
        try:
            company = params.get("company")
            title = params.get("title")
            if not company and not title:
                return json.dumps({"error": "get_skill_gap requires at least a company or title."})

            profile_dict = _get_resume_profile_dict(session_id)
            if not profile_dict:
                return json.dumps({"note": "No resume profile found for this session."})
            resume_text = format_resume_string(profile_dict)

            result = get_skill_gap_analysis(
                collection=job_collection,
                resume_text=resume_text,
                company=company,
                title=title,
            )
            return json.dumps(result)
        except Exception as e:
            return f"ERROR: get_skill_gap failed: {e}"

    elif tool_name == "search_posting":
        try:
            query = params.get("query", "")
            if not query:
                return json.dumps({"error": "search_posting requires a query."})
            matches = search_postings(query, n_results=5)
            return json.dumps({"results": matches})
        except Exception as e:
            return f"ERROR: search_posting failed: {e}"

    else:
        return f"ERROR: Unknown tool '{tool_name}'"



def format_final_answer(answer: str) -> str:
    """Convert a model-produced JSON payload into a readable text response.

    This acts as a safety net for cases where the LLM returns JSON instead of prose for its
    final answer. The function extracts the most relevant fields and formats them into a compact
    natural-language summary.

    Args:
        answer: Raw final-answer text returned by the model.

    Returns:
        A human-readable string suitable for display to the user.
    """
    try:
        data = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        return answer

    if not isinstance(data, dict):
        return answer

    parts = []
    if "tier" in data:
        parts.append(f"Tier: {data['tier']}.")
    if "similarity_score" in data:
        parts.append(f"Similarity score: {data['similarity_score']}.")
    if data.get("matched_skills"):
        parts.append(f"Matched skills: {', '.join(data['matched_skills'])}.")
    if data.get("deal_breaker_gaps"):
        parts.append(f"Deal-breaker gaps: {', '.join(data['deal_breaker_gaps'])}.")
    if data.get("core_gaps"):
        parts.append(f"Core skills to develop: {', '.join(data['core_gaps'])}.")
    if data.get("roadmap"):
        milestones = [
            f"{m.get('milestone', '')} ({m.get('estimated_months', '?')} mo)"
            for m in data["roadmap"]
        ]
        parts.append("Roadmap: " + "; ".join(milestones) + ".")
    if data.get("top_matches"):
        titles = [
            f"{m.get('title')} at {m.get('company')} ({m.get('tier')})"
            for m in data["top_matches"]
        ]
        parts.append("Top matches: " + ", ".join(titles) + ".")
    if data.get("results"):
        titles = [f"{r.get('title')} at {r.get('company')}" for r in data["results"]]
        parts.append("Found: " + ", ".join(titles) + ".")
    if data.get("summary"):
        parts.append(data["summary"])
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

    """Run the agent loop by alternating between LLM reasoning and tool execution.

    The function builds a system prompt with the available tool descriptions and session context,
    then repeatedly asks the model for a thought/action pair. If a tool is required, it executes
    the tool and feeds the observation back into the conversation. Once the model produces a
    final answer, the result is formatted and persisted to session memory.

    Args:
        user_message: The latest user message to respond to.
        session_id: Unique identifier for the current conversation.
        resume_skills: The list of skills from the saved resume profile.
        target_role: The target role from the saved resume profile.
        max_turns: Maximum number of reasoning turns before stopping.
        verbose: Whether to print progress information during execution.

    Returns:
        The final answer string returned to the user, or an error message if the loop fails.
    """

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
    """Send the current conversation to the configured LLM and return the raw response.

    Args:
        messages: Conversation history represented as a list of role/content dictionaries.

    Returns:
        The model's text response as a string.
    """

    client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.chat(model=MODEL, messages=messages, format="json")
    return response['message']['content'].strip()


def parse_json(text: str) -> dict:
    """Parse a model response that is expected to contain JSON.

    The function strips optional markdown code fences and returns the parsed dictionary.

    Args:
        text: Raw text from the LLM response.

    Returns:
        A dictionary parsed from the JSON payload.
    """

    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)