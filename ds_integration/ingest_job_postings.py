import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from config import EMBEDDING_MODEL


bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

from pathlib import Path

CHROMA_DB_PATH = Path(__file__).resolve().parent / "chroma_db"

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

job_collection = chroma_client.get_or_create_collection(
    name="job_postings",
    embedding_function=bge_ef
)


def upsert_job_postings(job_postings: list[dict] | pd.DataFrame) -> int:
    """Embeds job postings' requirements and stores them in Chroma.
    Uses upsert so re-running on the same postings is safe (no duplicates).

    Returns the number of postings ingested.
    """
    if isinstance(job_postings, pd.DataFrame):
        jobs_list = job_postings.to_dict(orient="records") if not job_postings.empty else []
    else:
        jobs_list = job_postings or []

    if not jobs_list:
        return 0

    # Stable unique ID per posting — link is a good candidate since it shouldn't repeat
    ids = [str(job.get("id", job.get("link"))) for job in jobs_list]
    documents = [str(job.get("requirements", "")) for job in jobs_list]
    metadatas = [
        {
            "title": job.get("title", ""),
            "company": job.get("company", ""),
            "link": job.get("link", ""),
        }
        for job in jobs_list
    ]

    job_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(jobs_list)