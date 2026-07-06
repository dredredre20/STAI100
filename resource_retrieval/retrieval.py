from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

CHROMA_PERSIST_DIR = Path(__file__).parent / "chroma_store"
COLLECTION_NAME = "courses"

# Module-level cache — the embedding model and Chroma connection are loaded
# once per process, not once per query. FastEmbedEmbeddings() has real
# startup cost (loads an ONNX model), so re-instantiating it on every
# search_courses() call would be slow and wasteful in a chat loop where
# this might get called several times per conversation.
_vectorstore: Chroma | None = None


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        if not CHROMA_PERSIST_DIR.exists():
            raise FileNotFoundError(
                f"No Chroma store found at {CHROMA_PERSIST_DIR}. "
                f"Run `python resource_retrieval/ingest.py` first to build the index."
            )
        embeddings = FastEmbedEmbeddings()  # same backend used at ingest time
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR),
        )
    return _vectorstore


def search_courses(query: str, target_role: str | None = None, top_k: int = 5) -> list[dict]:
    """Semantic search over the course catalog. Embeds the query and returns
    the top_k most similar courses by cosine distance.

    If target_role is given, restricts results to that role's courses via
    Chroma's metadata filter — useful since the orchestrator already knows
    the user's target_role and shouldn't recommend cloud_engineering courses
    to a data_scientist candidate (or vice versa).

    Returns a list of dicts (not raw LangChain Document objects) so this is
    trivially JSON-serializable for the orchestrator tool observation.
    """
    vectorstore = _get_vectorstore()

    search_kwargs = {}
    if target_role is not None:
        search_kwargs["filter"] = {"target_role": target_role}

    results = vectorstore.similarity_search_with_score(query, k=top_k, **search_kwargs)

    courses = []
    for doc, score in results:
        courses.append({
            "course_id": doc.metadata.get("course_id"),
            "title": doc.metadata.get("title"),
            "provider": doc.metadata.get("provider"),
            "level": doc.metadata.get("level"),
            "skills_covered": doc.metadata.get("skills_covered"),
            "target_role": doc.metadata.get("target_role"),
            # Chroma returns a distance (lower = more similar), not a
            # similarity score — inverted here so callers/LLMs see a more
            # intuitive "higher = better match" number instead.
            "relevance_score": round(1 / (1 + score), 3),
        })
    return courses


if __name__ == "__main__":
    # Quick manual sanity check — run this file directly to confirm
    # retrieval actually returns sensible results before wiring it into
    # the orchestrator.
    test_query = "I need to learn machine learning and deep learning"
    print(f"Query: {test_query!r} (target_role=data_scientist)\n")
    for course in search_courses(test_query, target_role="data_scientist", top_k=3):
        print(f"  [{course['relevance_score']}] {course['title']} ({course['provider']})")