"""
This tests the agent's capacities for handling ambiguity and to ensure that the output of the model
is within the scope of this project which is career readiness advising and preparation of materials 
for the job application.
"""
import json
import pytest
from react.react_agent import run_agent
from metrics import collector
from session_helpers import FIXED_SESSION_ID, FIXED_RESUME_SKILLS, FIXED_TARGET_ROLE, llm, reset_fixed_session, llm_judge

class TestAmbiguityAndScopeHandling:
    def test_handle_ambiguous_user_queries_gracefully(self):
        """Ensures the agent can handle vague or ambiguous queries by asking clarifying questions
        or providing a range of possible interpretations, rather than making assumptions."""
        answer = run_agent(
            user_message="what now?",
            session_id=FIXED_SESSION_ID,
            resume_skills=FIXED_RESUME_SKILLS,
            target_role=FIXED_TARGET_ROLE,
            verbose=False,
        )
        result = llm_judge(
            question="what now?",
            answer=answer,
            rubric=(
                "PASS if the response acknowledges the ambiguity of the query and either asks for "
                "clarification or provides multiple reasonable interpretations. FAIL if it makes "
                "unsupported assumptions or gives a single prescriptive answer without context."
            ),
        )
        collector.record("judge_score", result["score"])
        assert result["pass"], f"Ambiguity handling check failed: {result['reasoning']}"

    def test_scope_handling_for_broad_queries(self):
        """Ensures the agent can handle broad queries by breaking them down into manageable parts
        or providing a structured approach, rather than giving an overly generic response."""
        answer = run_agent(
            user_message= "Should I become a data scientist?",
            session_id=FIXED_SESSION_ID,
            resume_skills=FIXED_RESUME_SKILLS,
            target_role=FIXED_TARGET_ROLE,
            verbose=False,
        )
        result = llm_judge(
            question="Should I become a data scientist?",
            answer=answer,
            rubric=(
                "PASS if the response breaks down the broad query into actionable steps, provides "
                "a structured learning path, or suggests specific milestones and also references "
                "the user's current skills in the resume."
                "FAIL if it gives an overly generic or vague answer without guidance."
            ),
        )
        collector.record("judge_score", result["score"])
        assert result["pass"], f"Scope handling check failed: {result['reasoning']}"

    def test_unrelated_career_queries(self):
        """Ensures the agent can handle queries that are unrelated to the user's career path
        by providing relevant advice or redirecting appropriately."""
        answer = run_agent(
            user_message="what are some good hobbies for a data scientist?",
            session_id=FIXED_SESSION_ID,
            resume_skills=FIXED_RESUME_SKILLS,
            target_role=FIXED_TARGET_ROLE,
            verbose=False,
        )
        result = llm_judge(
            question="what are some good hobbies for a data scientist?",
            answer=answer,
            rubric=(
                "PASS if the response provides relevant hobby suggestions that align with the "
                "interests of a data scientist or explains how hobbies can complement their career. "
                "FAIL if it gives irrelevant or unhelpful advice."
            ),
        )
        collector.record("judge_score", result["score"])
        assert result["pass"], f"Unrelated query handling check failed: {result['reasoning']}"

class TestTheAgentsConstraintHandling:
    def test_emotional_user_queries(self):
        """
        Ensures that the agent remains focused on career readiness advising and not emotional counseling, 
        even when the user expresses personal dissatisfaction or emotional distress.
        """

        answer = run_agent(
            user_message="Please be honest, am I good enough for this role? I " \
            "feel like super unqualified and not happy about my life.",
            session_id=FIXED_SESSION_ID,
            resume_skills=FIXED_RESUME_SKILLS,
            target_role=FIXED_TARGET_ROLE,
            verbose=False,
        )
        result = llm_judge(
            question="Please be honest, am I good enough for this role? I " \
            "feel like super unqualified and not happy about my life.",
            answer=answer,
            rubric=(
                "PASS if the response remains professional and provides constructive "
                "feedback or encouragement without making personal judgments. "
                "FAIL if it gives unprofessional, dismissive, or overly personal advice."
            ),
        )
        collector.record("judge_score", result["score"])
        assert result["pass"], f"Emotional query handling check failed: {result['reasoning']}"


    def test_irrelevant_or_off_topic_queries(self):
        """
        Ensures that the agent can handle queries that are completely off-topic or irrelevant to career readiness advising, 
        by politely redirecting the conversation back to relevant topics.
        """
        answer = run_agent(
            user_message="What's the weather in Manila today?",
            session_id=FIXED_SESSION_ID,
            resume_skills=FIXED_RESUME_SKILLS,
            target_role=FIXED_TARGET_ROLE,
            verbose=False,
        )
        result = llm_judge(
            question="What's the weather in Manila today?",
            answer=answer,
            rubric=(
                "PASS if the response acknowledges the off-topic nature of the query and "
                "redirects the conversation back to career readiness advising. "
                "FAIL if it provides irrelevant information or fails to guide the user back to the intended scope."
            ),
        )
        collector.record("judge_score", result["score"])
        assert result["pass"], f"Irrelevant query handling check failed: {result['reasoning']}"



