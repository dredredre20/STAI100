"""
This test is specifically for evaluating the agent's capacities for handling
career-readiness advising tasks such as gap analysis, faithfulness to job postings, 
and providing actionable guidance. 
"""

import json
import pytest
from react.react_agent import run_agent
from metrics import collector
from session_helpers import FIXED_SESSION_ID, FIXED_RESUME_SKILLS, FIXED_TARGET_ROLE, llm, reset_fixed_session, llm_judge

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


    def test_consistency_in_skill_gap_analysis(self):
        """Ensures that repeated queries about skill gaps yield consistent results,
        and that the agent does not contradict itself or provide conflicting advice."""
        answer1 = run_agent(
            user_message="what am I missing for the Data Scientist role at LSEG?",
            session_id=FIXED_SESSION_ID,
            resume_skills=FIXED_RESUME_SKILLS,
            target_role=FIXED_TARGET_ROLE,
            verbose=False,
        )

        # added an addition query to test consistency even when the user asks 
        # a different query about a different company inbetween the two skill gap queries
        run_agent(
            user_message="What are the requirements for Wells Fargo's senior analytics role?",
            session_id=FIXED_SESSION_ID,
            resume_skills=FIXED_RESUME_SKILLS,
            target_role=FIXED_TARGET_ROLE,
            verbose=False,
        )

        answer2 = run_agent(
            user_message="can you tell me again what gaps I need to close for the Data Scientist role at LSEG?",
            session_id=FIXED_SESSION_ID,
            resume_skills=FIXED_RESUME_SKILLS,
            target_role=FIXED_TARGET_ROLE,
            verbose=False,
        )
        result = llm_judge(
            question="Are the two responses consistent in their skill gap analysis?",
            answer=f"Response 1: {answer1}\nResponse 2: {answer2}",
            rubric=(
                "PASS if both responses provide consistent information regarding the missing skills "
                "for the Data Scientist role at LSEG, without contradictions. "
                "FAIL if there are discrepancies or conflicting advice between the two responses."
            ),
        )
        collector.record("judge_score", result["score"])

        assert answer1.strip() and answer2.strip(), "One of the two responses was empty"
        assert result["pass"], f"Consistency check failed: {result['reasoning']}"

    def test_multi_requirement_for_different_companies(self):
        """
        Tests the agent's ability to handle queries that involved multiple job postings for different companies in one query, 
        and whether it can provide a comprehensive skill gap analysis for each role without omitting any of them.
        """
        answer = run_agent(
            user_message="what am I missing for the Data Scientist role at LSEG? " \
            "Also check for Wells Fargo's senior analytics role and Globe's AI engineering role",
            session_id=FIXED_SESSION_ID,
            resume_skills=FIXED_RESUME_SKILLS,
            target_role=FIXED_TARGET_ROLE,
            verbose=False,
            )

        result = llm_judge(
            question="what am I missing for the Data Scientist role at LSEG? " \
            "Also check for Wells Fargo's senior analytics role and Globe's AI engineering role",
            answer=answer,
            rubric=(
                "PASS if the response provides a clear and accurate skill gap analysis for all three " \
                "roles mentioned, without omitting any of them. " \
                "FAIL if it only addresses one or two of the roles, or provides inaccurate information."
                ),
            )
        collector.record("judge_score", result["score"])
        assert result["pass"], f"Multi-requirement check failed: {result['reasoning']}"

class TestFaithfulnessAndGrounding:
    def test_cited_requirements_are_grounded_in_job_posting(self):
        """Ensures the agent only claims missing skills/requirements that actually 
        exist in the retrieved job posting, without inventing requirements."""
        answer = run_agent(
            user_message="what specific requirements am I missing for the Data Scientist role at LSEG?",
            session_id=FIXED_SESSION_ID,
            resume_skills=FIXED_RESUME_SKILLS,
            target_role=FIXED_TARGET_ROLE,
            verbose=False,
        )
        result = llm_judge(
            question="what specific requirements am I missing for the Data Scientist role at LSEG?",
            answer=answer,
            rubric=(
                "PASS if all mentioned missing skills or job requirements strictly align with "
                "standard/retrieved requirements for the role and do not hallucinate non-existent "
                "qualifications or fabricate false information about the resume. "
                "FAIL if the answer invents unsupported skill gaps or requirements."
            ),
        )
        collector.record("judge_score", result["score"])
        assert result["pass"], f"Faithfulness check failed: {result['reasoning']}"

class TestActionabilityAndGuidance:
    def test_recommends_concrete_next_steps_for_gaps(self):
        """Verifies that advice goes beyond identifying gaps to giving concrete, 
        actionable learning paths or project ideas."""
        answer = run_agent(
            user_message="how can I close my skill gaps for the Data Scientist role at LSEG?",
            session_id=FIXED_SESSION_ID,
            resume_skills=FIXED_RESUME_SKILLS,
            target_role=FIXED_TARGET_ROLE,
            verbose=False,
        )
        result = llm_judge(
            question="how can I close my skill gaps for the Data Scientist role at LSEG?",
            answer=answer,
            rubric=(
                "PASS if the response provides specific, actionable recommendations (e.g., building "
                "a specific type of project, studying specific libraries, or taking target courses) "
                "to bridge the gaps. FAIL if it merely lists the missing skills without actionable guidance."
            ),
        )
        collector.record("judge_score", result["score"])
        assert result["pass"], f"Actionability check failed: {result['reasoning']}"

# Test for checking the output of the agent if it is human-readable and not structured output
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
