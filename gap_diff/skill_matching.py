# gap_diff/skill_matching.py
import re
from rapidfuzz import fuzz

# Fuzzy match threshold (0-100). Below this, two skill strings are treated
# as different skills. 85 is fairly strict — tuned to catch near-duplicates
# like "Postgres" vs "PostgreSQL" without collapsing genuinely different
# skills (e.g. "AWS" vs "Azure") into false matches.
MATCH_THRESHOLD = 85


def normalize_skill(skill: str) -> str:
    """Lowercase, strip punctuation/extra whitespace so trivial formatting
    differences ("AWS" vs "aws", "CI/CD" vs "CI-CD") don't count as
    mismatches before fuzzy matching even runs."""
    skill = skill.lower().strip()
    skill = re.sub(r'[^\w\s]', ' ', skill)   # punctuation -> space
    skill = re.sub(r'\s+', ' ', skill).strip()
    return skill


def best_fuzzy_match(target_skill: str, candidate_skills: list[str]) -> tuple[str, float] | None:
    """Find the best fuzzy match for target_skill among candidate_skills.
    Returns (matched_candidate, score) if score >= MATCH_THRESHOLD, else None.
    Matching is done on normalized strings, but the original candidate
    string is returned so downstream display keeps proper casing."""
    target_norm = normalize_skill(target_skill)
    best_match = None
    best_score = 0.0

    for candidate in candidate_skills:
        candidate_norm = normalize_skill(candidate)
        # token_sort_ratio handles word-order differences and partial overlaps
        # better than plain ratio() for multi-word skill phrases.
        score = fuzz.token_sort_ratio(target_norm, candidate_norm)
        if score > best_score:
            best_score = score
            best_match = candidate

    if best_match is not None and best_score >= MATCH_THRESHOLD:
        return best_match, best_score
    return None