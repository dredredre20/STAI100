from pathlib import Path
from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings

COURSES_DIR = Path(__file__).parent / "courses"
CHROMA_PERSIST_DIR = Path(__file__).parent / "chroma_store"
COLLECTION_NAME = "courses"

FOLDER_TO_ROLE = {
    "cloud_engineering": "cloud_engineering",
    "data_scientist": "data_scientist",
}


@dataclass
class Course:
    course_id: str        # e.g. "course1", "course2" — derived from filename stem
    target_role: str      # e.g. "cloud_engineering", "data_scientist"
    title: str
    provider: str
    level: str
    skills_covered: str
    description: str
    text: str             # full raw text, used as the embedded page_content


def _infer_target_role(filepath: Path) -> str:
    # Derive target_role from the immediate parent folder name.
    parent_folder = filepath.parent.name
    if parent_folder not in FOLDER_TO_ROLE:
        raise ValueError(
            f"Unrecognized course folder: '{parent_folder}' (file: {filepath.name}). "
            f"Expected one of: {list(FOLDER_TO_ROLE.keys())}"
        )
    return FOLDER_TO_ROLE[parent_folder]


def _parse_course_text(text: str) -> dict:
    """Parses the fixed 'Title: ... / Provider: ... / Level: ... /
    Skills covered: ... / Description: ...' format used in every course
    .txt file. Assumes each field is on its own line, in this order —
    matches the format you're already using across all 16 files."""
    fields = {"title": "", "provider": "", "level": "", "skills_covered": "", "description": ""}
    key_map = {
        "title": "title",
        "provider": "provider",
        "level": "level",
        "skills covered": "skills_covered",
        "description": "description",
    }
    lines = text.strip().splitlines()
    current_key = None
    for line in lines:
        if ":" in line:
            prefix, _, rest = line.partition(":")
            normalized = prefix.strip().lower()
            if normalized in key_map:
                current_key = key_map[normalized]
                fields[current_key] = rest.strip()
                continue
        # Handles multi-line descriptions (text wrapping onto following lines)
        if current_key:
            fields[current_key] = (fields[current_key] + " " + line.strip()).strip()
    return fields


def load_courses(courses_dir: Path = COURSES_DIR) -> list[Course]:
    """Walks courses/ recursively, reading every .txt file."""
    courses = []
    for filepath in sorted(courses_dir.rglob("*.txt")):
        text = filepath.read_text(encoding="utf-8").strip()
        if not text:
            continue  # skip empty files rather than embedding garbage
        target_role = _infer_target_role(filepath)
        parsed = _parse_course_text(text)
        courses.append(
            Course(
                course_id=filepath.stem,
                target_role=target_role,
                text=text,
                **parsed,
            )
        )
    return courses


def courses_to_documents(courses: list[Course]) -> list[Document]:
    """One Document per course — no chunking. Courses are short enough
    that splitting would fragment the skills-covered/description content
    and hurt retrieval quality for course recommendation."""
    return [
        Document(
            page_content=c.text,
            metadata={
                "course_id": c.course_id,
                "target_role": c.target_role,
                "title": c.title,
                "provider": c.provider,
                "level": c.level,
                "skills_covered": c.skills_covered,
            },
        )
        for c in courses
    ]


def build_chroma_collection(
    documents: list[Document],
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    embeddings = FastEmbedEmbeddings()  
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )
    return vectorstore


if __name__ == "__main__":
    courses = load_courses()
    print(f"Loaded {len(courses)} courses from {COURSES_DIR}")
    for c in courses:
        print(f"  [{c.target_role}] {c.course_id}: {c.title}")

    documents = courses_to_documents(courses)
    vectorstore = build_chroma_collection(documents)
    print(f"\nBuilt Chroma collection '{COLLECTION_NAME}' with {len(documents)} documents")
    print(f"Persisted to: {CHROMA_PERSIST_DIR}")