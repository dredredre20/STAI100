from ds_integration.ingest_job_postings import job_collection

def get_all_postings() -> list[dict]:
    """Fetch every stored job posting back out of Chroma (doc + metadata)."""
    result = job_collection.get(include=["documents", "metadatas"])
    postings = []
    for doc, meta in zip(result["documents"], result["metadatas"]):
        postings.append({
            "title": meta.get("title", ""),
            "company": meta.get("company", ""),
            "link": meta.get("link", ""),
            "requirements": doc,
        })
    return postings


def search_postings(query: str, n_results: int = 5) -> list[dict]:
    """Semantic search over stored postings for a fuzzy/descriptive query."""
    results = job_collection.query(query_texts=[query], n_results=n_results)

    if not results["documents"] or not results["documents"][0]:
        return []

    matches = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        matches.append({
            "title": meta.get("title", ""),
            "company": meta.get("company", ""),
            "link": meta.get("link", ""),
            "similarity": round(1 - distance, 4),  # Chroma returns distance; convert to similarity
        })
    return matches