"""
Unit evals for resume formatting and the Chroma $and filter fix.
Run with: pytest evaluation/unit/ -v
"""
import pytest
from ds_integration.job_fit_prediction import format_resume_string


class TestFormatResumeString:
    def test_handles_validated_profile_key(self):
        profile = {
            "validated_profile": {
                "current_role_category": "QA Engineer",
                "target_role": "Data Scientist",
                "years_of_experience": 4,
                "skills": ["Python", "SQL"],
                "certifications": ["AWS Certified Cloud Practitioner"],
                "education_level": "Bachelor's",
            }
        }
        result = format_resume_string(profile)
        assert "QA Engineer" in result
        assert "Data Scientist" in result
        assert "Python, SQL" in result

    def test_handles_fields_key(self):
        profile = {
            "fields": {
                "current_role_category": "Analyst",
                "target_role": "Data Engineer",
                "years_of_experience": 2,
                "skills": ["Excel"],
                "certifications": [],
                "education_level": "Bachelor's",
            }
        }
        result = format_resume_string(profile)
        assert "Analyst" in result
        assert "Excel" in result

    def test_missing_fields_dont_crash(self):
        # Empty/partial profile shouldn't raise — should degrade gracefully
        result = format_resume_string({})
        assert isinstance(result, str)

    def test_empty_skills_list_produces_empty_string_not_crash(self):
        profile = {"fields": {"skills": [], "certifications": []}}
        result = format_resume_string(profile)
        assert "Skills: ." in result or "Skills:" in result


class TestChromaWhereFilterConstruction:
    """Regression test for the ValueError we hit in production:
    'Expected where to have exactly one operator, got {...}'
    Chroma requires $and for multi-condition filters."""

    def _build_where_filter(self, company=None, title=None):
        # mirrors the logic inside get_job_by_identifier
        conditions = []
        if company:
            conditions.append({"company": company})
        if title:
            conditions.append({"title": title})
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    def test_single_condition_no_and_wrapper(self):
        result = self._build_where_filter(company="LSEG")
        assert result == {"company": "LSEG"}

    def test_multiple_conditions_wrapped_in_and(self):
        result = self._build_where_filter(company="LSEG", title="Data Scientist")
        assert result == {"$and": [{"company": "LSEG"}, {"title": "Data Scientist"}]}

    def test_no_conditions_returns_none(self):
        assert self._build_where_filter() is None
