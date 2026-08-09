from ds_integration.ingest_job_postings import job_collection


def get_all_postings() -> list[dict]:
    """Retrieve every stored job posting from the Chroma collection.

    Returns:
        A list of dictionaries containing the posting title, company, link, and stored
        requirements text.
    """
    # Request both the document text and metadata for each stored embedding.
    result = job_collection.get(include=["documents", "metadatas"])
    postings = []
    for doc, meta in zip(result["documents"], result["metadatas"]):
        postings.append(
            {
                "title": meta.get("title", ""),
                "company": meta.get("company", ""),
                "link": meta.get("link", ""),
                "requirements": doc,
            }
        )
    return postings


def search_postings(query: str, n_results: int = 5) -> list[dict]:
    """Perform semantic search over stored postings for a fuzzy or descriptive query.

    Args:
        query: The natural-language search text to compare against the stored job posting
            embeddings.
        n_results: The maximum number of matching postings to return.

    Returns:
        A list of dictionaries containing the best matching postings and their similarity
        scores, where higher values indicate a stronger match.
    """
    # Query Chroma using the provided text and request the top matching results.
    results = job_collection.query(query_texts=[query], n_results=n_results)

    if not results["documents"] or not results["documents"][0]:
        return []

    matches = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # Chroma returns a distance metric, so convert it to a similarity score.
        matches.append(
            {
                "title": meta.get("title", ""),
                "company": meta.get("company", ""),
                "link": meta.get("link", ""),
                "similarity": round(1 - distance, 4),
            }
        )
    return matches