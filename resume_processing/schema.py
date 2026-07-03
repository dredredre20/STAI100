import re
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, Literal

# ── Field definitions (drives extraction prompt + completeness check) ─────
RESUME_FIELD_DEFINITIONS = {
    "current_role_category": {
        "description": "The person's current job title or field, as stated on the resume",
        "required": False,   # open field — never blocks completion
    },
    "target_role": {
        "description": "Which career transition they're targeting: software_development or cloud_engineering",
        "required": True,    # the ONLY field that gates disambiguation
    },
    "years_of_experience": {
        "description": "Total years of relevant professional experience",
        "required": False,
    },
    "skills": {
        "description": "Technical skills mentioned in the resume, as a list",
        "required": False,
    },
    "certifications": {
        "description": "Certifications listed, if any, as a list",
        "required": False,
    },
    "education_level": {
        "description": "Highest level of education completed",
        "required": False,
    },
}


# ── ResumeProfile — the Pydantic model ─────────────────────────────────────
class ResumeProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")  # rejects any unexpected field, e.g. leaked PII

    current_role_category: Optional[str] = Field(
        default=None,
        description="The person's current job title or field, as stated on the resume"
    )

    target_role: Optional[Literal["software_development", "cloud_engineering"]] = Field(
        default=None,
        description="Which of the two supported career transitions the person is targeting"
    )

    years_of_experience: Optional[float] = Field(
        default=None, description="Total years of relevant professional experience"
    )

    skills: list[str] = Field(
        default_factory=list, description="Technical skills extracted from the resume"
    )

    certifications: list[str] = Field(
        default_factory=list, description="Certifications listed on the resume, if any"
    )

    education_level: Optional[str] = Field(
        default=None, description="Highest level of education completed"
    )

    @field_validator("skills", "certifications")
    @classmethod
    def dedupe_and_clean(cls, v: list[str]) -> list[str]:
        seen, cleaned = set(), []
        for item in v:
            item = item.strip()
            if item and item.lower() not in seen:
                seen.add(item.lower())
                cleaned.append(item)
        return cleaned

    @field_validator("years_of_experience")
    @classmethod
    def validate_years(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or v > 60):
            raise ValueError("years_of_experience must be between 0 and 60")
        return v