import argparse
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from config import EMBEDDING_MODEL

bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

# Persist the Chroma vector database inside the ds_integration/chroma_db folder.
CHROMA_DB_PATH = Path(__file__).resolve().parent / "chroma_db"

# Default to the LinkedIn JSON export stored in the repository.
DEFAULT_JOB_POSTINGS_PATH = (
    Path(__file__).resolve().parent.parent / "job_postings_ph" / "linkedin_jobs_ds.json"
)

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

    documents = [
        f"{job.get('title', '')}. {job.get('requirements', '')}"
        for job in jobs_list
    ]

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


def main():
    parser = argparse.ArgumentParser(description="Ingest job postings into ChromaDB.")
    parser.add_argument(
        "--file",
        type=str,
        default=str(DEFAULT_JOB_POSTINGS_PATH),
        help="Path to the job postings JSON or CSV file.",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"Loading data from {file_path}...")

    if file_path.suffix.lower() == ".json":
        df = pd.read_json(file_path)
    else:
        print("Error: Unsupported file format. Please provide a .csv or .json file.")
        return

    count = upsert_job_postings(df)
    print(f"Successfully ingested {count} job postings into ChromaDB.")


if __name__ == "__main__":
    main()