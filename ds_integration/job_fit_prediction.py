import pandas as pd
from sentence_transformers import SentenceTransformer, util
from config import EMBEDDING_MODEL

# Prefix used to guide the embedding model toward a search-oriented representation.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

model = SentenceTransformer(EMBEDDING_MODEL)

# Similarity thresholds used to bucket matches into human-readable tiers.
STRONG_MATCH_THRESHOLD = 0.75
POSSIBLE_MATCH_THRESHOLD = 0.55

# Match label -> tier name mapping.
TIER_LABELS = {
    "Strong Match": "Ready to Apply",
    "Possible Match": "Near Match",
    "Weak Match": "Long-Term Upskilling",
}

def get_match_label(score: float) -> str:
    """Classify a similarity score into an internal match bucket.

    Args:
        score: Cosine similarity score between the resume embedding and a job embedding.

    Returns:
        A human-readable label describing the strength of the match.
    """
    if score >= STRONG_MATCH_THRESHOLD:
        return "Strong Match"
    elif score >= POSSIBLE_MATCH_THRESHOLD:
        return "Possible Match"
    else:
        return "Weak Match"


def get_tier(score: float) -> str:
    """Translate a numeric similarity score into a display-friendly tier name.

    Args:
        score: Cosine similarity score in the range [-1, 1].

    Returns:
        A user-facing label such as "Ready to Apply" or "Near Match".
    """
    return TIER_LABELS[get_match_label(score)]


def predict_job_matches(
    processed_resume: dict | str,
    job_postings: list[dict] | pd.DataFrame,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Compute similarity scores between a resume and a list of job postings.

    The function converts the resume and each job requirement into embeddings, measures
    cosine similarity, and returns a ranked DataFrame with tier labels.

    Args:
        processed_resume: Either a structured resume dictionary or a pre-formatted string.
            If a dictionary is provided, it is converted into a text representation using
            the resume formatting helper.
        job_postings: Either a list of job posting dictionaries or a pandas DataFrame
            containing job rows.
        batch_size: Number of job embeddings to encode in each batch. Larger values can
            improve throughput, but may use more memory.

    Returns:
        A pandas DataFrame with one row per job posting and columns for title, company,
        link, similarity score, and tier.
    """
    # Normalize the input into a list of job dictionaries for consistent processing.
    if isinstance(job_postings, pd.DataFrame):
        jobs_list = job_postings.to_dict(orient="records") if not job_postings.empty else []
    else:
        jobs_list = job_postings or []

    output_columns = ["title", "company", "link", "similarity_score", "tier"]

    # Return an empty DataFrame early when there are no jobs to compare.
    if not jobs_list:
        return pd.DataFrame(columns=output_columns)

    # Convert a structured resume payload into a text summary for embedding.
    if isinstance(processed_resume, dict):
        raw_resume_text = format_resume_string(processed_resume)
    else:
        raw_resume_text = str(processed_resume)

    # Use a retrieval-oriented prompt prefix so the embedding is aligned with search use cases.
    query_text = f"{BGE_QUERY_INSTRUCTION}{raw_resume_text}"

    # Encode the resume once and the job requirements in batches.
    resume_embedding = model.encode(query_text, normalize_embeddings=True, convert_to_tensor=True)

    requirements_list = [str(job.get("requirements", "")) for job in jobs_list]
    job_embeddings = model.encode(
        requirements_list,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_tensor=True,
    )

    # Compute cosine similarity between the resume and each job requirement embedding.
    cosine_scores = util.cos_sim(resume_embedding, job_embeddings)[0].cpu().numpy()

    scored_results = []
    for idx, job in enumerate(jobs_list):
        # Round the score for easier readability while preserving ranking order.
        score = round(float(cosine_scores[idx]), 4)
        scored_results.append(
            {
                "title": job.get("title"),
                "company": job.get("company"),
                "link": job.get("link"),
                "similarity_score": score,
                "tier": get_tier(score),
            }
        )

    results_df = pd.DataFrame(scored_results, columns=output_columns)
    return results_df.sort_values(by="similarity_score", ascending=False).reset_index(drop=True)


def format_resume_string(processed_resume: dict) -> str:
    """Build a compact text summary from a structured resume payload.

    The returned string is designed for embedding generation and preserves the fields
    most relevant for matching against job requirements.

    Args:
        processed_resume: A dictionary containing resume data, usually with either a
            "validated_profile" or "fields" key.

    Returns:
        A single string that summarizes the current role, target role, experience,
        skills, certifications, and education.
    """
    profile = processed_resume.get("validated_profile") or processed_resume.get("fields", {})

    current_role = profile.get("current_role_category", "")
    target_role = profile.get("target_role", "")
    yoe = profile.get("years_of_experience", "")
    skills = ", ".join(profile.get("skills", []))
    certs = ", ".join(profile.get("certifications", []))
    education = profile.get("education_level", "")

    resume_text = (
        f"Current Role: {current_role}. Target Role: {target_role}. "
        f"Experience: {yoe} years. Skills: {skills}. "
        f"Certifications: {certs}. Education: {education}."
    )
    return resume_text