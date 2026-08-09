import json
import uuid
from memory.db_setup import get_connection


def create_session() -> str:
    """Generate a new session_id and register it. Session-based (no login) —
    the caller (e.g. the outer ReAct orchestrator) generates one session_id
    per user interaction and passes it through every subsequent call."""
    session_id = str(uuid.uuid4())
    conn = get_connection()
    conn.execute("INSERT INTO sessions (session_id) VALUES (?)", (session_id,))
    conn.commit()
    conn.close()
    return session_id


def save_resume_profile(session_id: str, profile: dict) -> int:
    """Save a ResumeProfile (as a dict, e.g. from ResumeProfile.model_dump())
    for a given session. Returns the new row's id, needed to link a
    subsequent diff_result back to the exact profile it was computed from."""
    conn = get_connection()
    cursor = conn.execute(
        """INSERT INTO resume_profiles
           (session_id, target_role, current_role_category, years_of_experience,
            skills, certifications, education_level)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            profile.get("target_role"),
            profile.get("current_role_category"),
            profile.get("years_of_experience"),
            json.dumps(profile.get("skills", [])),
            json.dumps(profile.get("certifications", [])),
            profile.get("education_level"),
        ),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def update_skills(
    session_id: str,
    new_skills: list[str] | None = None,
    new_certifications: list[str] | None = None,
) -> dict:
    """Appends new skills/certifications to the session's latest resume
    profile and saves it as a NEW profile row (not an in-place update),
    so profile history over time stays intact — matches the existing
    pattern of resume_profiles being append-only, same as diff_results.
    Returns the updated profile dict, or an error note if no profile exists
    yet to update."""
    conn = get_connection()
    row = conn.execute(
        """SELECT * FROM resume_profiles
           WHERE session_id = ?
           ORDER BY created_at DESC LIMIT 1""",
        (session_id,),
    ).fetchone()

    if not row:
        conn.close()
        return {"error": "No existing resume profile found for this session to update."}

    current = dict(row)
    current_skills = json.loads(current.get("skills") or "[]")
    current_certs = json.loads(current.get("certifications") or "[]")

    # Merge, de-duplicate (case-insensitive), preserve original casing of existing entries
    def merge_unique(existing: list[str], additions: list[str]) -> list[str]:
        existing_lower = {s.lower() for s in existing}
        merged = list(existing)
        for item in additions or []:
            if item.lower() not in existing_lower:
                merged.append(item)
                existing_lower.add(item.lower())
        return merged

    updated_skills = merge_unique(current_skills, new_skills or [])
    updated_certs = merge_unique(current_certs, new_certifications or [])

    cursor = conn.execute(
        """INSERT INTO resume_profiles
           (session_id, target_role, current_role_category, years_of_experience,
            skills, certifications, education_level)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            current.get("target_role"),
            current.get("current_role_category"),
            current.get("years_of_experience"),
            json.dumps(updated_skills),
            json.dumps(updated_certs),
            current.get("education_level"),
        ),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()

    return {
        "profile_id": new_id,
        "skills": updated_skills,
        "certifications": updated_certs,
        "added_skills": new_skills or [],
        "added_certifications": new_certifications or [],
    }


def get_latest_resume_profile(session_id: str) -> dict | None:
    """Fetch this session's most recently saved resume profile — skills,
    certifications, education, target_role, years of experience. Use this
    for general questions about the user's background/profile, as opposed
    to get_session_history (readiness scores over time) or get_skill_gap
    (a fresh comparison against job requirements)."""
    conn = get_connection()
    row = conn.execute(
        """SELECT * FROM resume_profiles
           WHERE session_id = ?
           ORDER BY created_at DESC LIMIT 1""",
        (session_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None