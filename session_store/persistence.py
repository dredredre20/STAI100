import json
import uuid
from session_store.db_setup import get_connection


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


def save_diff_result(session_id: str, resume_profile_id: int, target_role: str, diff_result) -> int:
    """Save a GapDiffResult (from gap_diff.diff_engine.run_gap_diff) tied to
    the resume_profile it was computed from. Stores just the skill NAMES
    (not full frequency/score detail) since that's what's needed for
    progress comparison across sessions — the full requirement-side detail
    can always be recomputed from corpus_build if needed."""
    conn = get_connection()
    cursor = conn.execute(
        """INSERT INTO diff_results
           (session_id, resume_profile_id, target_role, readiness_score,
            matched_required, missing_required, matched_preferred, missing_preferred)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            resume_profile_id,
            target_role,
            diff_result.readiness_score,
            json.dumps([m.requirement_skill for m in diff_result.matched_required]),
            json.dumps([m.skill for m in diff_result.missing_required]),
            json.dumps([m.requirement_skill for m in diff_result.matched_preferred]),
            json.dumps([m.skill for m in diff_result.missing_preferred]),
        ),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def get_latest_diff_result(session_id: str) -> dict | None:
    """Convenience fetch — most recent diff_result for a session, used by
    the orchestrator to decide whether this is a returning user with
    existing history to compare against."""
    conn = get_connection()
    row = conn.execute(
        """SELECT * FROM diff_results
           WHERE session_id = ?
           ORDER BY created_at DESC LIMIT 1""",
        (session_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_session_history(session_id: str, limit: int = 10) -> list[dict]:
    """Deterministic fetch of all diff_results for a session, most-recent
    first. Used for progress/history questions ('am I improving?', 'how many
    times have I checked?') without routing them through LLM-generated SQL —
    same underlying data get_latest_diff_result reaches, just not limited to
    one row, and with zero free-form query generation involved."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT * FROM diff_results
           WHERE session_id = ?
           ORDER BY created_at DESC LIMIT ?""",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]