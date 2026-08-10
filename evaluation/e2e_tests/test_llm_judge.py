"""
End-to-end evals — full run_agent() call, then a JUDGE model (not the same
small local model being tested) scores the final answer against a rubric.

Using a separate, stronger judge model avoids the "model grading its own
homework" problem — gemma4:e4b's own biases/blind spots would just be
replicated if it judged itself.

Requires an open model key (or your judge provider's key) to be set.
Run with: pytest evaluation/end_to_end/ -v -s
"""
import json
import pytest
from react.react_agent import run_agent
from metrics import collector
from session_helpers import FIXED_SESSION_ID, FIXED_RESUME_SKILLS, FIXED_TARGET_ROLE, llm, reset_fixed_session


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
{{"pass": true/false, "score": 0-5, "reasoning": "one or two sentences"}}"""

    response = llm.invoke(prompt)
    raw = response.content.strip()
    return json.loads(raw)

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
        collector.record("judge_score", result["score"])
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
        collector.record("judge_score", result["score"])
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
        collector.record("judge_score", result["score"])
        assert result["pass"], f"Judge failed this response: {result['reasoning']}"

    # This will test whether the agent correctly updates its skill gap analysis after the user adds new skills to their profile.
    def test_skill_gap_reflects_recently_updated_skills(self):

        # Initial run to add new skills
        run_agent(
        user_message="I've learned Terraform and Kubernetes, please add that to my profile",
        session_id=FIXED_SESSION_ID,
        resume_skills=FIXED_RESUME_SKILLS,
        target_role=FIXED_TARGET_ROLE,
        verbose=False,
        )
        # Run skill gap analysis again to see if the agent recognizes that Terraform and Kubernetes are no longer missing skills.
        answer = run_agent(
            user_message="what am I missing for the Data Scientist role at LSEG now?",
            session_id=FIXED_SESSION_ID,
            resume_skills=FIXED_RESUME_SKILLS,
            target_role=FIXED_TARGET_ROLE,
            verbose=False,
        )

        # Judge the response to ensure it does not list Terraform and Kubernetes in the list of missing skills.
        result = llm_judge(
            question="what am I missing for the Data Scientist role at LSEG now?",
            answer=answer,
            rubric=(
                "PASS if the response does NOT list Terraform or Kubernetes as missing "
                "skills, since these were just added to the profile. FAIL if it still "
                "treats them as gaps."
            ),
        )
        collector.record("judge_score", result["score"])
        assert result["pass"], f"Judge failed this response: {result['reasoning']}"


   
    def test_average_judge_score_meets_threshold(self):
        """
        This test check the average judge score across all tests cases. 
        We expect that 80% of the test cases will be passing which is why the threshold is set to 4.0.
        """

        avg = collector.mean("judge_score")
        print(f"\nAverage judge score: {avg:.2f}/5 ({collector.count('judge_score')} cases)")
        assert avg >= 4.0, f"Average judge score {avg:.2f} below 4.0 threshold"


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
