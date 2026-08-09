import pandas as pd
from sentence_transformers import SentenceTransformer, util
from config import EMBEDDING_MODEL

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

model = SentenceTransformer(EMBEDDING_MODEL)

STRONG_MATCH_THRESHOLD = 0.75
POSSIBLE_MATCH_THRESHOLD = 0.55

# Match label -> tier name mapping 
TIER_LABELS = {
    "Strong Match": "Ready to Apply",
    "Possible Match": "Near Match",
    "Weak Match": "Long-Term Upskilling",
}

def get_match_label(score: float) -> str:
    """Internal bucket name based on similarity score."""
    if score >= STRONG_MATCH_THRESHOLD:
        return "Strong Match"
    elif score >= POSSIBLE_MATCH_THRESHOLD:
        return "Possible Match"
    else:
        return "Weak Match"


def get_tier(score: float) -> str:
    """User-facing tier name"""
    return TIER_LABELS[get_match_label(score)]

def predict_job_matches(
    processed_resume: dict | str,
    job_postings: list[dict] | pd.DataFrame,
    batch_size: int = 32,
) -> pd.DataFrame:
    if isinstance(job_postings, pd.DataFrame):
        jobs_list = job_postings.to_dict(orient="records") if not job_postings.empty else []
    else:
        jobs_list = job_postings or []

    output_columns = ["title", "company", "link", "similarity_score", "tier"]

    if not jobs_list:
        return pd.DataFrame(columns=output_columns)

    if isinstance(processed_resume, dict):
        raw_resume_text = format_resume_string(processed_resume)
    else:
        raw_resume_text = str(processed_resume)

    query_text = f"{BGE_QUERY_INSTRUCTION}{raw_resume_text}"

    resume_embedding = model.encode(query_text, normalize_embeddings=True, convert_to_tensor=True)

    requirements_list = [str(job.get("requirements", "")) for job in jobs_list]
    job_embeddings = model.encode(
        requirements_list, batch_size=batch_size, normalize_embeddings=True, convert_to_tensor=True
    )

    cosine_scores = util.cos_sim(resume_embedding, job_embeddings)[0].cpu().numpy()

    scored_results = []
    for idx, job in enumerate(jobs_list):
        score = round(float(cosine_scores[idx]), 4)
        scored_results.append({
            "title": job.get("title"),
            "company": job.get("company"),
            "link": job.get("link"),
            "similarity_score": score,
            "tier": get_tier(score),
        })

    results_df = pd.DataFrame(scored_results, columns=output_columns)
    return results_df.sort_values(by="similarity_score", ascending=False).reset_index(drop=True)


def format_resume_string(processed_resume: dict) -> str:
    """Extracts relevant fields from validated_profile into a flat string for embedding."""
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