import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from config import EMBEDDING_MODEL

from pathlib import Path

bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

# Persist the Chroma vector database inside the ds_integration/chroma_db folder.
CHROMA_DB_PATH = Path(__file__).resolve().parent / "chroma_db"

# Connect to the local Chroma instance and create or reuse the job postings collection.
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
job_collection = chroma_client.get_or_create_collection(
    name="job_postings",
    embedding_function=bge_ef,
)


def upsert_job_postings(job_postings: list[dict] | pd.DataFrame) -> int:
    """Embed job posting requirements and store them in Chroma.

    The function converts each posting's requirements into vector embeddings and upserts
    them into the existing Chroma collection. Using upsert makes re-running ingestion
    safe because the same posting IDs overwrite previous entries instead of creating duplicates.

    Args:
        job_postings: Either a list of dictionaries containing job fields or a pandas
            DataFrame whose rows represent job postings.

    Returns:
        The number of job postings successfully ingested into the collection.
    """
    # Normalize the input to a consistent list-of-dicts format.
    if isinstance(job_postings, pd.DataFrame):
        jobs_list = job_postings.to_dict(orient="records") if not job_postings.empty else []
    else:
        jobs_list = job_postings or []

    # Return early when there is nothing to ingest.
    if not jobs_list:
        return 0

    # Use a stable identifier per posting; the link is a sensible default when no explicit ID exists.
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

    # Upsert the embeddings and metadata into the Chroma collection.
    job_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(jobs_list)