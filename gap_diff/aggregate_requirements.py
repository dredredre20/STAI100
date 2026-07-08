# gap_diff/aggregate_requirements.py
import json
from pathlib import Path
from dataclasses import dataclass, field

from gap_diff.skill_matching import normalize_skill, best_fuzzy_match

REQUIREMENTS_DIR = Path(__file__).parent.parent / "postings" / "requirements"


@dataclass
class SkillFrequency:
    skill: str            # canonical display form (first-seen casing)
    count: int            # number of postings this skill appeared in
    total_postings: int   # total postings for this target_role


# dataclass to hold aggregated reqs for target and preferred skills by postings (for comparison)
@dataclass
class AggregatedRequirements:
    target_role: str
    total_postings: int
    required: list[SkillFrequency] = field(default_factory=list)
    preferred: list[SkillFrequency] = field(default_factory=list)


# function to load all postings based on target role, and skills
def _load_postings_for_role(target_role: str, requirements_dir: Path = REQUIREMENTS_DIR) -> list[dict]:
    postings = []
    for filepath in sorted(requirements_dir.glob("*.json")):
        data = json.loads(filepath.read_text(encoding="utf-8"))
        if data.get("target_role") == target_role:
            postings.append(data)
    return postings


# function to aggregate a list of postings into canonical skill frequencies
def _aggregate_skill_list(postings: list[dict], field_name: str) -> list[SkillFrequency]:
    """Merge a given field ('required_skills' or 'preferred_skills') across all
    postings for a role. Fuzzy matching is used here too — different postings
    often phrase the same skill slightly differently (e.g. one posting says
    "PostgreSQL", another says "Postgres") — without merging these, frequency
    counts would be artificially fragmented and under-count real repetition."""
    canonical_skills: list[str] = []       # display-form skill strings, in first-seen order
    counts: dict[str, int] = {}            # canonical_skill -> count

    # Loop through postings and their skills, fuzzy-matching to canonical_skills
    for posting in postings:
        seen_in_this_posting = set()  # avoid double-counting if a posting somehow repeats a skill
        for raw_skill in posting.get(field_name, []):
            match = best_fuzzy_match(raw_skill, canonical_skills)
            if match:
                canonical_skill = match[0]
            else:
                canonical_skill = raw_skill
                canonical_skills.append(canonical_skill)
                counts[canonical_skill] = 0

            norm_canonical = normalize_skill(canonical_skill)
            if norm_canonical not in seen_in_this_posting:
                counts[canonical_skill] += 1
                seen_in_this_posting.add(norm_canonical)

    total = len(postings)
    return [
        SkillFrequency(skill=skill, count=counts[skill], total_postings=total)
        for skill in canonical_skills
    ]


# function to aggregate requirements for a given target role
def aggregate_requirements_for_role(target_role: str, requirements_dir: Path = REQUIREMENTS_DIR) -> AggregatedRequirements:
    postings = _load_postings_for_role(target_role, requirements_dir)
    if not postings:
        raise ValueError(f"No requirement postings found for target_role='{target_role}'")

    required = _aggregate_skill_list(postings, "required_skills")
    preferred = _aggregate_skill_list(postings, "preferred_skills")

    # Sort by frequency descending — most commonly requested skills first.
    # This is what lets the agent later say "here are the top 3 most in-demand
    # gaps" instead of dumping an unordered list.
    required.sort(key=lambda sf: sf.count, reverse=True)
    preferred.sort(key=lambda sf: sf.count, reverse=True)

    return AggregatedRequirements(
        target_role=target_role,
        total_postings=len(postings),
        required=required,
        preferred=preferred,
    )