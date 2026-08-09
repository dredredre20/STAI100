import json
import ollama
from config import MODEL, OLLAMA_BASE_URL
from memory.db_memory import get_latest_resume_profile
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
    """
    param session_id: the unique session ID for this conversation
    return: dict of the latest resume profile for this session, or empty dict if none exists
    Fetches the saved resume profile and wraps it in the shape
    format_resume_string() expects (a 'fields' key).
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


# function to run a tool based on the action dict returned by the model, return json result as string
def run_tool(action: dict, session_id: str, resume_skills: list[str], target_role: str) -> str:
    """
    param action: dict containing "tool_name" and "parameters" keys
    param session_id: the unique session ID for this conversation   
    param resume_skills: the list of skills from the user's resume
    return : JSON string result of the tool execution
    Runs the tool specified in the action dict, returning its output as a JSON string.
    If the tool fails, returns a JSON string with an "error" key describing the failure
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
            top_matches = results_df.head(5).to_dict(orient="records")
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



# function to format the final answer from the model, converting raw JSON into readable text
def format_final_answer(answer: str) -> str:
    """
    param answer: the raw final answer string from the model
    return: human-readable text version of the final answer
    Safety net — if the LLM echoed a tool's raw JSON as its final_answer
    instead of writing prose, reformat it into readable text.
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


# function to run the agent loop, calling the LLM and tools iteratively until a final answer is reached or max turns exceeded
def run_agent(
    user_message: str,
    session_id: str,
    resume_skills: list[str],
    target_role: str,
    max_turns: int = 10,
    verbose: bool = True,
) -> str:

    """
    param user_message: the message from the user to respond to
    param session_id: the unique session ID for this conversation
    param resume_skills: the list of skills from the user's resume
    param target_role: the user's target role for career readiness
    param max_turns: the maximum number of turns to allow before giving up
    param verbose: whether to print debug information during the agent loop
    return: the final answer from the agent, or an error message if it failed

    This function runs the agent loop, calling the LLM and tools iteratively until a final answer is reached or 
    the maximum number of turns is exceeded. It constructs the system prompt with user context and conversation history, 
    then repeatedly calls the LLM to get thoughts and actions, executing tools as needed. The final answer is formatted and returned.
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
    """
    param messages: list of dicts representing the conversation so far
    return : LLM's response as a string
    This function calls the LLM with the given messages and returns its response.
    """

    client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.chat(model=MODEL, messages=messages, format="json")
    return response['message']['content'].strip()


def parse_json(text: str) -> dict:
    """
    param text: raw text from the LLM that is expected to be a JSON object
    return: parsed JSON as a dict
    This function parses the LLM response text, cleaning it and returns the json object.
    """

    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)