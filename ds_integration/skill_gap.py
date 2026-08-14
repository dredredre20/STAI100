import json
import ollama
from config import MODEL, OLLAMA_BASE_URL
from ds_integration.job_fit_prediction import get_tier, BGE_QUERY_INSTRUCTION


TIER_PROMPTS = {
    "Ready to Apply": """The candidate is a strong match for this role. Highlight the specific
skills, tools, and experience from their resume that directly align with THIS COMPANY's stated
hiring standards in the job requirements — be concrete, naming the exact requirement and the
matching resume evidence (e.g. "Requires 3+ years with cloud infrastructure — your AWS EC2/S3
experience at NexusTech covers this."). Keep tone confident and affirming; this candidate should
walk away knowing exactly why they're ready.

Respond with ONLY valid JSON, no markdown fences, no extra text:
{{"tier": "Ready to Apply", "matched_skills": ["..."], "summary": "..."}}""",

    "Near Match": """The candidate is close but not quite there. Your main job is to flag the
exact "deal-breaker" skill(s) — requirements explicitly marked as required/must-have in the job
description that are missing or unclear in the resume. Name the specific missing skill/technology
directly, framed against what they already have, in this style: "You match [Company]'s backend
stack, but the role strictly requires [missing skill], which isn't on your resume." Do not soften
this into a vague gap list — the goal is one or two precise, named blockers standing between them
and Ready to Apply.

Respond with ONLY valid JSON, no markdown fences, no extra text:
{{"tier": "Near Match", "matched_skills": ["..."], "deal_breaker_gaps": ["..."], "summary": "..."}}""",

    "Long-Term Upskilling": """The candidate is not currently a match for this role, and gap-listing
alone won't help them. Instead, map out a 6-12 month upskilling roadmap: identify the 3-5 CORE
skills (not nice-to-haves) that would move them from unqualified toward this role, in the order
they should be learned, each with a realistic timeframe. Frame this as a plan, not a rejection —
concrete milestones the candidate can actually act on.

Respond with ONLY valid JSON, no markdown fences, no extra text:
{{"tier": "Long-Term Upskilling", "core_gaps": ["..."], "roadmap": [
  {{"milestone": "...", "skills": ["..."], "estimated_months": 0}}
], "summary": "..."}}""",
}

def get_job_by_identifier(collection, company: str = None, title: str = None, job_id: str = None):
    """Retrieve a specific job posting from Chroma by exact ID or metadata filter.

    Args:
        collection: The Chroma collection containing stored job postings.
        company: Optional company name used to filter matching postings.
        title: Optional job title used to filter matching postings.
        job_id: Optional exact identifier for a single posting.

    Returns:
        A dictionary with the posting ID, requirements text, and metadata when a match is found;
        otherwise, None.
    """
    if job_id:
        # Fetch the exact posting by its Chroma ID.
        result = collection.get(ids=[job_id], include=["documents", "metadatas"])
    else:
        # Build a metadata filter from the provided company/title values.
        conditions = []
        if company:
            conditions.append({"company": company})
        if title:
            conditions.append({"title": title})

        if not conditions:
            return None

        where_filter = conditions[0] if len(conditions) == 1 else {"$and": conditions}
        result = collection.get(where=where_filter, include=["documents", "metadatas"])

    if not result["documents"]:
        return None

    return {
        "id": result["ids"][0],
        "requirements_text": result["documents"][0],
        "metadata": result["metadatas"][0],
    }


def find_job_semantic(collection, query_text: str, n_results: int = 1):
    """Find the closest matching job posting using semantic similarity.

    This is used as a fallback when an exact company/title lookup does not produce a match.

    Args:
        collection: The Chroma collection containing stored job postings.
        query_text: Natural-language text used to search for relevant postings.
        n_results: Maximum number of matching postings to return.

    Returns:
        A dictionary with the best matching posting, its requirements text, metadata, and a
        similarity score when results are available; otherwise, None.
    """
    # Prefix the query so the embedding matches the same retrieval style used elsewhere.
    prefixed_query = f"{BGE_QUERY_INSTRUCTION}{query_text}"
    results = collection.query(query_texts=[prefixed_query], n_results=n_results)

    if not results["documents"] or not results["documents"][0]:
        return None

    return {
        "id": results["ids"][0][0],
        "requirements_text": results["documents"][0][0],
        "metadata": results["metadatas"][0][0],
        "similarity_score": round(1 - results["distances"][0][0], 4),
    }


def get_similarity_for_job(collection, resume_text: str, job_id: str, search_pool: int = 50) -> float:
    """Compute the similarity between a resume and a specific job posting.

    Chroma's query API does not support filtering directly to a known ID, so this helper
    queries broadly and extracts the score for the requested posting from the returned results.

    Args:
        collection: The Chroma collection containing stored job postings.
        resume_text: Resume text to compare against the posting embeddings.
        job_id: The target posting ID to locate in the query results.
        search_pool: Number of nearest neighbors to request. Increase this if the collection
            grows beyond the current size.

    Returns:
        A similarity score in the range [0, 1] when the job is found; otherwise 0.0.
    """
    prefixed_resume = f"{BGE_QUERY_INSTRUCTION}{resume_text}"
    result = collection.query(query_texts=[prefixed_resume], n_results=search_pool)

    if not result["ids"] or not result["ids"][0]:
        return 0.0

    for idx, rid in enumerate(result["ids"][0]):
        if rid == job_id:
            return round(1 - result["distances"][0][idx], 4)

    # The requested posting was not within the initial search pool, so retry with a larger query.
    total_count = collection.count()
    if total_count > search_pool:
        wider = collection.query(query_texts=[prefixed_resume], n_results=total_count)
        for idx, rid in enumerate(wider["ids"][0]):
            if rid == job_id:
                return round(1 - wider["distances"][0][idx], 4)

    return 0.0


def analyze_skill_gap(resume_text: str, job_requirements_text: str, similarity_score: float) -> dict:
    """Compare a resume against a specific job using the local LLM.

    The similarity score determines which response template is used so the LLM can produce
    either a strong-match summary, a blocker-focused near-match analysis, or a roadmap for
    long-term upskilling.

    Args:
        resume_text: The candidate resume text to analyze.
        job_requirements_text: The target job posting requirements text.
        similarity_score: Numeric similarity score used to choose the appropriate tier prompt.

    Returns:
        A dictionary containing the LLM-generated analysis payload.
    """
    # Select the prompt style based on the match tier.
    tier = get_tier(similarity_score)
    tier_instruction = TIER_PROMPTS[tier]

    prompt = f"""Resume:
{resume_text}

Job Requirements:
{job_requirements_text}

{tier_instruction}"""

    # Use the configured Ollama client to generate the structured response.
    client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    raw = response["message"]["content"].strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"tier": tier, "summary": "Could not parse skill gap analysis.", "raw_response": raw}

    return result


def get_skill_gap_analysis(
    collection,
    resume_text: str,
    company: str = None,
    title: str = None,
    fuzzy_query: str = None,
) -> dict:
    """Resolve a job posting, compute its similarity to a resume, and generate a skill-gap analysis.

    The function first tries an exact metadata lookup, then falls back to semantic search if
    no direct match is found. Once a job is resolved, it computes the similarity score and
    hands the resume/job text pair to the local LLM for structured analysis.

    Args:
        collection: The Chroma collection containing stored job postings.
        resume_text: The candidate resume text to compare against the job.
        company: Optional company name for exact lookup.
        title: Optional job title for exact lookup.
        fuzzy_query: Optional natural-language fallback query when exact lookup fails.

    Returns:
        A dictionary with the skill-gap analysis payload and associated job metadata.
    """
    job = None
    similarity_score = None

    # Try an exact lookup first when company or title information is available.
    if company or title:
        job = get_job_by_identifier(collection, company=company, title=title)
        if job:
            similarity_score = get_similarity_for_job(collection, resume_text, job["id"])

    # Fall back to semantic retrieval if no exact match was found.
    if job is None:
        query = fuzzy_query or " ".join(filter(None, [title, company]))
        job = find_job_semantic(collection, query)
        if job:
            similarity_score = job.get("similarity_score", 0.0)

    if job is None:
        return {"error": "No matching job posting found."}

    # Generate the analysis and attach the matched job metadata to the response.
    result = analyze_skill_gap(resume_text, job["requirements_text"], similarity_score)
    result["job_metadata"] = job["metadata"]
    return result