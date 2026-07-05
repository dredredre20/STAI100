import re
from rapidfuzz import fuzz

MATCH_THRESHOLD = 85


def normalize_skill(skill: str) -> str:
    """Lowercase, strip punctuation/extra whitespace so trivial formatting
    differences ("AWS" vs "aws", "CI/CD" vs "CI-CD") don't count as
    mismatches before fuzzy matching even runs."""
    skill = skill.lower().strip()
    skill = re.sub(r'[^\w\s]', ' ', skill)
    skill = re.sub(r'\s+', ' ', skill).strip()
    return skill


def best_fuzzy_match(target_skill: str, candidate_skills: list[str]) -> tuple[str, float] | None:
    """Find the best fuzzy match for target_skill among candidate_skills.
    Returns (matched_candidate, score) if score >= MATCH_THRESHOLD, else None.

    Uses the max of two scoring strategies rather than one:
    - token_sort_ratio: handles word-order differences.
    - ratio (plain Levenshtein-based): handles spacing/concatenation
      differences ("Power BI" vs "PowerBI") that token_sort_ratio scores poorly.

    Deliberately does NOT use partial_ratio — it scores substring matches
    too generously (e.g. "Git" would falsely match "Version Control (git)").
    """
    target_norm = normalize_skill(target_skill)
    best_match = None
    best_score = 0.0

    for candidate in candidate_skills:
        candidate_norm = normalize_skill(candidate)
        score = max(
            fuzz.token_sort_ratio(target_norm, candidate_norm),
            fuzz.ratio(target_norm, candidate_norm),
        )
        if score > best_score:
            best_score = score
            best_match = candidate

    if best_match is not None and best_score >= MATCH_THRESHOLD:
        return best_match, best_score
    return None
