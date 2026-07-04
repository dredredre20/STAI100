# corpus_build/ingest.py
from pathlib import Path
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings  

JOB_POSTINGS_DIR = Path(__file__).parent / "job_postings"
CHROMA_PERSIST_DIR = Path(__file__).parent / "chroma_store"
COLLECTION_NAME = "job_postings"

FOLDER_TO_ROLE = {
    "cloud_engineering": "cloud_engineering",
    "data_scientist": "data_scientist",
}


@dataclass
class JobPosting:
    posting_id: str        # e.g. "cloud1", "data1" — derived from filename stem
    target_role: str       # e.g. "cloud_engineering", "data_scientist"
    text: str


def _infer_target_role(filepath: Path) -> str:
    """Derive target_role from the immediate parent folder name.
    Raises if unrecognized — fail loudly rather than silently mis-tagging a posting."""
    parent_folder = filepath.parent.name
    if parent_folder not in FOLDER_TO_ROLE:
        raise ValueError(
            f"Unrecognized job posting folder: '{parent_folder}' (file: {filepath.name}). "
            f"Expected one of: {list(FOLDER_TO_ROLE.keys())}"
        )
    return FOLDER_TO_ROLE[parent_folder]


def load_job_postings(postings_dir: Path = JOB_POSTINGS_DIR) -> list[JobPosting]:
    """Walks job_postings/ recursively, reading every .txt file."""
    postings = []
    for filepath in sorted(postings_dir.rglob("*.txt")):
        text = filepath.read_text(encoding="utf-8").strip()
        if not text:
            continue  # skip empty files rather than embedding garbage
        target_role = _infer_target_role(filepath)
        posting_id = filepath.stem
        postings.append(JobPosting(posting_id=posting_id, target_role=target_role, text=text))
    return postings


def postings_to_documents(postings: list[JobPosting]) -> list[Document]:
    """One Document per posting — no chunking. Postings are short enough
    that splitting would fragment requirement lists across chunks and hurt
    retrieval quality for the gap-diff engine."""
    return [
        Document(
            page_content=p.text,
            metadata={"posting_id": p.posting_id, "target_role": p.target_role},
        )
        for p in postings
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