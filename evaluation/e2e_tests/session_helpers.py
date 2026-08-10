import pytest
from memory.db_setup import get_connection
from memory.db_memory import save_resume_profile
from memory import conversation_memory
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

# Fixed test fixtures — using a consistent resume/session across cases
# makes it easier to spot when a *specific* case starts failing vs. noise
# from a different resume each run.
FIXED_SESSION_ID = "eval-e2e-session"
FIXED_RESUME_SKILLS = ["Python", "SQL", "Pandas", "Selenium", "AWS", "scikit-learn"]
FIXED_TARGET_ROLE = "data_scientist"

# Delete resume profile and session id so that other tests don't carryover to the next
def _clear_fixed_session():
    conn = get_connection()
    conn.execute("DELETE FROM resume_profiles WHERE session_id = ?", (FIXED_SESSION_ID,))
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (FIXED_SESSION_ID,))
    conn.commit()
    conn.close()
    conversation_memory._session_histories.pop(FIXED_SESSION_ID, None)


@pytest.fixture(autouse=True) # fixture runs automatically for every test in this file
def reset_fixed_session():
    """Resets FIXED_SESSION_ID before every test in this file: wipes any
    resume_profiles rows and conversation history left over from a prior
    test or a prior pytest run, then re-seeds a known baseline profile.
    Without this, test_skill_gap_reflects_recently_updated_skills's added
    skills (Terraform/Kubernetes) persist in the DB forever and silently
    change what the earlier tests see."""
    _clear_fixed_session()

    conn = get_connection()
    conn.execute("INSERT INTO sessions (session_id) VALUES (?)", (FIXED_SESSION_ID,))
    conn.commit()
    conn.close()

    save_resume_profile(FIXED_SESSION_ID, {
        "target_role": FIXED_TARGET_ROLE,
        "skills": FIXED_RESUME_SKILLS,
        "certifications": [],
        "current_role_category": None,
        "years_of_experience": None,
        "education_level": None,
    })

    yield

    _clear_fixed_session()
