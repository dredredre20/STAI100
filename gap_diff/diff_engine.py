# gap_diff/diff_engine.py
from dataclasses import dataclass, field

from gap_diff.aggregate_requirements import aggregate_requirements_for_role, AggregatedRequirements
from gap_diff.skill_matching import best_fuzzy_match

# Weights used to compute the overall readiness score. Required skills count
# more heavily than preferred ones — missing a required skill should hurt the
# score more than missing a preferred one.
REQUIRED_WEIGHT = 0.8
PREFERRED_WEIGHT = 0.2


@dataclass
class MatchedSkill:
    resume_skill: str      # the skill string as it appeared on the resume
    requirement_skill: str  # the canonical requirement skill it matched against
    match_score: float      # fuzzy match score (0-100)
    frequency: int          # how many postings requested this skill
    total_postings: int


@dataclass
class MissingSkill:
    skill: str
    frequency: int          # how many postings requested this skill
    total_postings: int
    frequency_pct: float    # frequency / total_postings * 100, for easy display


@dataclass
class GapDiffResult:
    target_role: str
    total_postings_analyzed: int
    matched_required: list[MatchedSkill] = field(default_factory=list)
    missing_required: list[MissingSkill] = field(default_factory=list)
    matched_preferred: list[MatchedSkill] = field(default_factory=list)
    missing_preferred: list[MissingSkill] = field(default_factory=list)
    readiness_score: float = 0.0   # 0-100, weighted required+preferred match


def _diff_skill_group(
    resume_skills: list[str],
    requirement_group: list,  # list[SkillFrequency]
) -> tuple[list[MatchedSkill], list[MissingSkill]]:
    matched = []
    missing = []

    for req in requirement_group:
        match = best_fuzzy_match(req.skill, resume_skills)
        if match:
            matched_resume_skill, score = match
            matched.append(MatchedSkill(
                resume_skill=matched_resume_skill,
                requirement_skill=req.skill,
                match_score=score,
                frequency=req.count,
                total_postings=req.total_postings,
            ))
        else:
            missing.append(MissingSkill(
                skill=req.skill,
                frequency=req.count,
                total_postings=req.total_postings,
                frequency_pct=round(req.count / req.total_postings * 100, 1),
            ))

    return matched, missing


def _compute_readiness_score(
    matched_required: list, missing_required: list,
    matched_preferred: list, missing_preferred: list,
) -> float:
    total_required = len(matched_required) + len(missing_required)
    total_preferred = len(matched_preferred) + len(missing_preferred)

    required_pct = (len(matched_required) / total_required * 100) if total_required > 0 else 100.0
    preferred_pct = (len(matched_preferred) / total_preferred * 100) if total_preferred > 0 else 100.0

    score = (required_pct * REQUIRED_WEIGHT) + (preferred_pct * PREFERRED_WEIGHT)
    return round(score, 1)


def run_gap_diff(resume_skills: list[str], target_role: str) -> GapDiffResult:
    """Pure deterministic comparison — no LLM call. Loads the aggregated,
    frequency-weighted requirement set for target_role and diffs it against
    the candidate's extracted resume skills using fuzzy string matching."""
    aggregated: AggregatedRequirements = aggregate_requirements_for_role(target_role)

    matched_required, missing_required = _diff_skill_group(resume_skills, aggregated.required)
    matched_preferred, missing_preferred = _diff_skill_group(resume_skills, aggregated.preferred)

    readiness_score = _compute_readiness_score(
        matched_required, missing_required, matched_preferred, missing_preferred
    )

    return GapDiffResult(
        target_role=target_role,
        total_postings_analyzed=aggregated.total_postings,
        matched_required=matched_required,
        missing_required=missing_required,
        matched_preferred=matched_preferred,
        missing_preferred=missing_preferred,
        readiness_score=readiness_score,
    )