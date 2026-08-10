"""
Retrieval evals — given a query, is the right posting returned and ranked
correctly? Requires the same seeded postings collection used by the
trajectory/end-to-end evals. Run with: pytest evaluation/retrieval/ -v -s
"""
import pytest
from ds_integration.job_search import search_postings
from metrics import precision_at_k, recall_at_k, reciprocal_rank, collector


def _posting_id(posting: dict) -> tuple:
    return (posting.get("company"), posting.get("title"))


# Postings from the dataset, these should be returned for the queries below.
RETRIEVAL_CASES = [
    {"query": "Data scientist", 
     "relevant": {("JPMorganChase", "Data Scientist"), ("Maya", "Data Scientist"), ("Insight", "Data Scientist"), 
                  ("Smart Communications, Inc.", "Data Scientist"), ("Maya", "Lead Data Scientist")}}
    #{"query": "Non-data-science data roles", 
    # "relevant": {("iGaming Centre", "Data Engineer"), ("Ayannah", "Data Engineer"), 
    #                ("Pru Life UK", "Data Analytics Expert"), ("First Circle", "Data Analytics Lead")}},
    #{"query": "AI-related roles", 
    #"relevant": {("AI Rudder", "AI Data Operations Engineer (A37267)"), ("HedgeServ", "AI Engineer"), 
    #             ("Teoh Capital", "AI Agent Engineer"), ("Accenture in the Philippines", "AI / ML Engineer")}},

]

class TestSearchPostingRetrieval:

    @pytest.mark.parametrize("case", RETRIEVAL_CASES, ids=[c["query"] for c in RETRIEVAL_CASES])
    def test_precision_recall_mrr_at_k(self, case):
        results = search_postings(case["query"], n_results=100)
        retrieved = [_posting_id(r) for r in results]
        relevant = case["relevant"]

        k = max(len(case["relevant"]), 3)

        # compute precision, recall, and mmr
        p, r, rr = (
            precision_at_k(retrieved, relevant, k),
            recall_at_k(retrieved, relevant, k),
            reciprocal_rank(retrieved, relevant),
        )
        collector.record(f"retrieval_precision@{k}", p)
        collector.record(f"retrieval_recall@{k}", r)
        collector.record("retrieval_mrr", rr)
        print(f"\n[{case['query']}] p@{k}={p:.2f} r@{k}={r:.2f} mrr={rr:.2f}")
        assert rr > 0, f"No relevant postings retrieved for: {case['query']}"

    def test_mean_retrieval_metrics_meet_threshold(self):
        mean_mrr = collector.mean("retrieval_mrr")
        print(f"\nMean p@k={collector.mean('retrieval_precision@k'):.2f}, "
              f"r@k={collector.mean('retrieval_recall@k'):.2f}, mrr={mean_mrr:.2f}")
        assert mean_mrr >= 0.5, f"Mean MRR {mean_mrr:.2f} below 0.5 threshold"