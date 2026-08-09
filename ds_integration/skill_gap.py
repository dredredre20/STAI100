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


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def get_job_by_identifier(collection, company: str = None, title: str = None, job_id: str = None):
    """Retrieve a specific job posting's document + metadata from Chroma
    via exact ID or metadata filter (company/title)."""
    if job_id:
        result = collection.get(ids=[job_id], include=["documents", "metadatas"])
    else:
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
    """Semantic fallback when no exact company/title is given.
    Also returns a similarity score straight from this query, so we don't
    need a second round-trip to compute it."""
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
    """Computes resume-vs-specific-job similarity.

    Chroma's query() doesn't support filtering results down to a single known
    id, so instead we query broadly (resume against the whole collection) and
    pick out the distance for the job_id we care about from the results.
    search_pool should be >= total number of stored postings to guarantee the
    target job_id appears in results; bump it if your collection grows past 50.

    Uses the same BGE_QUERY_INSTRUCTION prefix as job_fit_prediction.py's
    predict_job_matches(), so scores/tiers agree between get_top_matches and
    get_skill_gap for the same resume+job pair.
    """
    prefixed_resume = f"{BGE_QUERY_INSTRUCTION}{resume_text}"
    result = collection.query(query_texts=[prefixed_resume], n_results=search_pool)

    if not result["ids"] or not result["ids"][0]:
        return 0.0

    for idx, rid in enumerate(result["ids"][0]):
        if rid == job_id:
            return round(1 - result["distances"][0][idx], 4)

    # job_id wasn't within search_pool results — collection is likely larger
    # than search_pool. Retry once against the full collection size.
    total_count = collection.count()
    if total_count > search_pool:
        wider = collection.query(query_texts=[prefixed_resume], n_results=total_count)
        for idx, rid in enumerate(wider["ids"][0]):
            if rid == job_id:
                return round(1 - wider["distances"][0][idx], 4)

    return 0.0


# ---------------------------------------------------------------------------
# Generation (Ollama)
# ---------------------------------------------------------------------------
def analyze_skill_gap(resume_text: str, job_requirements_text: str, similarity_score: float) -> dict:
    """Compare resume against a specific job's requirements using the local LLM,
    with the response style/tier driven by the similarity score."""
    tier = get_tier(similarity_score)
    tier_instruction = TIER_PROMPTS[tier]

    prompt = f"""Resume:
{resume_text}

Job Requirements:
{job_requirements_text}

{tier_instruction}"""

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


# ---------------------------------------------------------------------------
# This is what react_agent.py's run_tool() calls
# ---------------------------------------------------------------------------
def get_skill_gap_analysis(
    collection,
    resume_text: str,
    company: str = None,
    title: str = None,
    fuzzy_query: str = None,
) -> dict:
    """Resolves a job posting (exact lookup or semantic fallback), computes
    similarity against the resume, then runs tiered skill gap analysis."""
    job = None
    similarity_score = None

    if company or title:
        job = get_job_by_identifier(collection, company=company, title=title)
        if job:
            similarity_score = get_similarity_for_job(collection, resume_text, job["id"])

    if job is None:
        query = fuzzy_query or " ".join(filter(None, [title, company]))
        job = find_job_semantic(collection, query)
        if job:
            similarity_score = job.get("similarity_score", 0.0)

    if job is None:
        return {"error": "No matching job posting found."}

    result = analyze_skill_gap(resume_text, job["requirements_text"], similarity_score)
    result["job_metadata"] = job["metadata"]
    return result