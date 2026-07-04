# corpus_build/build_corpus.py
from corpus_build.ingest import load_job_postings, postings_to_documents, build_chroma_collection

def main():
    postings = load_job_postings()
    print(f"Loaded {len(postings)} job postings.")

    by_role = {}
    for p in postings:
        by_role.setdefault(p.target_role, 0)
        by_role[p.target_role] += 1
    print(f"Breakdown: {by_role}")

    documents = postings_to_documents(postings)
    vectorstore = build_chroma_collection(documents)
    print(f"Chroma collection built and persisted with {len(documents)} documents.")

if __name__ == "__main__":
    main()