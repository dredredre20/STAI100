import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "career_readiness.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS resume_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    target_role TEXT NOT NULL,
    current_role_category TEXT,
    years_of_experience REAL,
    skills TEXT,              -- JSON-encoded list
    certifications TEXT,       -- JSON-encoded list
    education_level TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS diff_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    resume_profile_id INTEGER NOT NULL,
    target_role TEXT NOT NULL,
    readiness_score REAL NOT NULL,
    matched_required TEXT,     -- JSON-encoded list of skill names
    missing_required TEXT,     -- JSON-encoded list of skill names
    matched_preferred TEXT,    -- JSON-encoded list of skill names
    missing_preferred TEXT,    -- JSON-encoded list of skill names
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (resume_profile_id) REFERENCES resume_profiles(id)
);
"""


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the database and tables if they don't already exist. Safe to
    call repeatedly — CREATE TABLE IF NOT EXISTS is idempotent."""
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Standard connection helper — row_factory set so query results come
    back as dict-like Row objects instead of plain tuples, which makes them
    much easier to serialize to JSON for the agent's tool results."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
