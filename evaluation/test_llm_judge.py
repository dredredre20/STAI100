"""
End-to-end evals — full run_agent() call, then a JUDGE model (not the same
small local model being tested) scores the final answer against a rubric.

Using a separate, stronger judge model avoids the "model grading its own
homework" problem — gemma4:e4b's own biases/blind spots would just be
replicated if it judged itself.

Requires ANTHROPIC_API_KEY (or your judge provider's key) to be set.
Run with: pytest evaluation/end_to_end/ -v -s
"""
import json
import pytest
import anthropic  # swap for your preferred judge provider's SDK
from react.react_agent import run_agent

JUDGE_MODEL = "claude-sonnet-4-6"  # a model NOT under test, used only to grade

judge_client = anthropic.Anthropic()


def llm_judge(question: str, answer: str, rubric: str) -> dict:
    """Sends the question/answer/rubric to a judge model, returns a
    structured score + reasoning. Keeping this as a separate function
    (not inline in the test) means the judge logic can be reused and
    audited independently of any single test case."""
    prompt = f"""You are grading a career-readiness advisor chatbot's response.

User Question: {question}

Chatbot's Answer: {answer}

Grading Rubric:
{rubric}

Respond with ONLY valid JSON, no markdown fences:
{{"pass": true/false, "score": 0-10, "reasoning": "one or two sentences"}}"""

    response = judge_client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    return json.loads(raw)


# Fixed test fixtures — using a consistent resume/session across cases
# makes it easier to spot when a *specific* case starts failing vs. noise
# from a different resume each run.
FIXED_SESSION_ID = "eval-e2e-session"
FIXED_RESUME_SKILLS = ["Python", "SQL", "Pandas", "Selenium", "AWS", "scikit-learn"]
FIXED_TARGET_ROLE = "data_scientist"


class TestSkillGapResponses:
    """These assume a resume profile + job postings are already seeded for
    FIXED_SESSION_ID — see evaluation/end_to_end/conftest.py for fixture setup."""

    def test_ready_to_apply_response_names_specific_evidence(self):
        answer = run_agent(
            user_message="what am I missing for the Data Scientist role at LSEG?",
            session_id=FIXED_SESSION_ID,
            resume_skills=FIXED_RESUME_SKILLS,
            target_role=FIXED_TARGET_ROLE,
            verbose=False,
        )
        result = llm_judge(
            question="what am I missing for the Data Scientist role at LSEG?",
            answer=answer,
            rubric=(
                "PASS if the response names at least one SPECIFIC skill or requirement from "
                "the job posting and connects it to specific resume evidence (not generic "
                "praise like 'you have great skills'). FAIL if vague or generic."
            ),
        )
        assert result["pass"], f"Judge failed this response: {result['reasoning']}"

    def test_near_match_response_names_deal_breaker(self):
        answer = run_agent(
            user_message="can you check what gaps i need to close before applying to Wells Fargo?",
            session_id=FIXED_SESSION_ID,
            resume_skills=FIXED_RESUME_SKILLS,
            target_role=FIXED_TARGET_ROLE,
            verbose=False,
        )
        result = llm_judge(
            question="what gaps do I need to close for Wells Fargo?",
            answer=answer,
            rubric=(
                "PASS if the response names at least one SPECIFIC missing skill/technology "
                "by name (e.g. 'Terraform', 'Kubernetes') rather than a vague statement like "
                "'you may have some gaps'. FAIL if no specific skill is named."
            ),
        )
        assert result["pass"], f"Judge failed this response: {result['reasoning']}"

    def test_response_does_not_hallucinate_when_no_postings_exist(self):
        """Regression test for the earlier bug where the agent invented a
        plausible-sounding excuse ('please ensure you have posted some
        roles') instead of accurately reporting an empty tool result."""
        answer = run_agent(
            user_message="which job postings match best with my current skills?",
            session_id="eval-empty-collection-session",  # session with NO seeded postings
            resume_skills=FIXED_RESUME_SKILLS,
            target_role=FIXED_TARGET_ROLE,
            verbose=False,
        )
        result = llm_judge(
            question="which job postings match best with my current skills?",
            answer=answer,
            rubric=(
                "PASS if the response accurately states that no job postings are currently "
                "available/loaded, without inventing a reason that blames the user (e.g. "
                "'please post some roles') or fabricating job postings that don't exist. "
                "FAIL if it fabricates postings or invents an unsupported excuse."
            ),
        )
        assert result["pass"], f"Judge failed this response: {result['reasoning']}"


class TestToneAndFormat:
    def test_response_is_conversational_not_raw_json(self):
        """Safety-net regression: format_final_answer() should always
        convert structured tool output into prose, never leak raw JSON."""
        answer = run_agent(
            user_message="what am I missing for the Data Scientist role at LSEG?",
            session_id=FIXED_SESSION_ID,
            resume_skills=FIXED_RESUME_SKILLS,
            target_role=FIXED_TARGET_ROLE,
            verbose=False,
        )
        assert not answer.strip().startswith("{"), "Raw JSON leaked into the final answer"
