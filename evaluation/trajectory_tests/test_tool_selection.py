"""
Trajectory evals — does the agent choose the RIGHT tool and RIGHT parameters
for a given user message? This does NOT judge the final answer quality (see
evaluation/end_to_end/ for that) — it isolates the reasoning/routing step,
since that's the step most likely to silently break with a small local model.

These tests call the LLM (via run_agent's single-turn tool selection), so
they're slower and less deterministic than unit tests — run separately.

Run with: pytest evaluation/trajectory/ -v -s
"""
import pytest
from react.react_agent import call_llm, parse_json, SYSTEM_PROMPT_TEMPLATE, TOOL_DESCRIPTIONS
from metrics import collector

FIXED_RESUME_SKILLS = ["Python", "SQL", "Pandas", "Selenium", "AWS"]
FIXED_TARGET_ROLE = "data_scientist"
FIXED_SESSION_ID = "eval-session-fixed"


def get_first_action(user_message: str) -> dict:
    """Runs a single agent turn and returns the parsed action (or None if
    the model skipped straight to a final_answer)."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        tool_descriptions=TOOL_DESCRIPTIONS,
        session_id=FIXED_SESSION_ID,
        resume_skills=FIXED_RESUME_SKILLS,
        target_role=FIXED_TARGET_ROLE,
        conversation_history="(none)",
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    raw = call_llm(messages)
    data = parse_json(raw)
    return data.get("action")


class TestToolRouting:
    """Each case pairs a realistic user phrasing with the tool call we
    expect it to trigger. Small local models can be inconsistent, so these
    are best read in aggregate (e.g. 8/10 pass) rather than expecting 100%
    on the first run — flag flaky ones for prompt tuning, not code bugs."""

    @pytest.mark.parametrize("user_message,expected_tool", [
        ("what jobs am I qualified for right now?", "get_top_matches"),
        ("which job postings match best with my current skills?", "get_top_matches"),
        ("what am I missing for the Data Scientist role at LSEG?", "get_skill_gap"),
        ("can you check what gaps i need to close before applying to Wells Fargo?", "get_skill_gap"),
        ("find me some AI engineering roles", "search_posting"),
        ("what skills do I have listed on my resume?", "get_user_profile"),
        ("I've learned how to use Databricks and Terraform, can you check my skill gaps with the role again?", "update_skills"), 
        ("I did a course on advanced python programming and did a data analysis project, check my skill gap again for the role", "update_skills"), 
        ("Given our discussions, can you draft me a cover letter that I can use in my applications?", "generate_cover_letter"),
        ("Can you help me write a cover letter for the Data Scientist role at LSEG?", "generate_cover_letter"),
        ("With my updated skills, can you generate me a resume that I can use to apply for the Data Scientist role at LSEG?", "generate_targeted_resume"),
        ("Can you help me create a resume that highlights my skills for the Data Scientist role at LSEG?", "generate_targeted_resume"),
    ])

    def test_correct_tool_selected(self, user_message, expected_tool):
        action = get_first_action(user_message)
        assert action is not None, f"Expected a tool call for '{user_message}', got a direct final_answer instead"
        correct = action["tool_name"] == expected_tool
        collector.record("tool_selection_accuracy", 1.0 if correct else 0.0)
        assert correct, f"For '{user_message}': expected {expected_tool}, got {action['tool_name']}"

    def test_skill_gap_extracts_company_param(self):
        action = get_first_action("what am I missing for the Data Scientist role at LSEG?")
        assert action["tool_name"] == "get_skill_gap"
        params = action.get("parameters", {})
        assert params.get("company") or params.get("title"), (
            "get_skill_gap was called with no company or title — agent will fail to resolve the job posting"
        )
        

    def test_update_skills_extracts_params(self):
        action = get_first_action("I've learned how to use Databricks and Terraform, can you check my skill gaps with the role again?")
        assert action["tool_name"] == "update_skills"
        params = action.get("parameters", {})
        assert params.get("new_skills") or params.get("new_certifications"), (
            "update_skills was called with no new_skills or new_certifications"
        )

    def test_does_not_call_tool_when_answer_already_in_history(self):
        """If the user context/profile already answers the question, the
        agent should skip tool use — this is a common small-model failure
        mode (calling tools redundantly)."""
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=TOOL_DESCRIPTIONS,
            session_id=FIXED_SESSION_ID,
            resume_skills=FIXED_RESUME_SKILLS,
            target_role=FIXED_TARGET_ROLE,
            conversation_history=(
                "User: what's my target role?\n"
                "Assistant: Your target role is data_scientist."
            ),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "wait, what did you say my target role was again?"},
        ]
        raw = call_llm(messages)
        data = parse_json(raw)
        assert data.get("action") is None, (
            "Agent called a tool for info already present in conversation history"
        )

    # Overall accuracy threshold is 80% since this is only a career readiness model and not a life-critical system
    # We can adjust this to a 90% to hold the system to a higher standard it should perform better all the time
    def test_tool_selection_accuracy_meets_threshold(self):
        accuracy = collector.mean("tool_selection_accuracy")
        print(f"\nTool selection accuracy: {accuracy:.1%} ({collector.count('tool_selection_accuracy')} cases)")
        assert accuracy >= 0.8, f"Accuracy {accuracy:.1%} below 80% threshold"
