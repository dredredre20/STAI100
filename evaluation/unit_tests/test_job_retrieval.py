"""
Retrieval evals — given a query, is the right posting returned and ranked
correctly? Requires the same seeded postings collection used by the
trajectory/end-to-end evals. Run with: pytest evaluation/retrieval/ -v -s
"""
import pytest
from ds_integration.job_search import search_postings
from metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank, collector


def _posting_id(posting: dict) -> tuple:
    return (posting.get("company"), posting.get("title"))


# Postings from the dataset, these should be returned for the queries below.
RETRIEVAL_CASES = [
    {"query": "Data scientist", 
     "relevant": {('LSEG', 'Data Scientist'), ('Home Credit Philippines', 'Data Scientist'), ("Maya", "Lead Data Scientist"),
                   ('Emapta Global', 'Applied Data Scientist'), ('Accenture in the Philippines', 'Data Science Specialist')}},
    {"query": "Data engineer",
     "relevant": {('Procter & Gamble', 'Data Engineer'), ('Insight', 'Data Engineer'), ('Thermo Fisher Scientific', 'Data Engineer'),
                    ('iGaming Centre', 'Data Engineer'), ('Luxor Technology', 'Data Engineer'), ('MicroSourcing', 'Data Engineer (pooling)'),
                    ('Viventis Search Asia', 'Junior Data Engineer'), ('QBE Insurance', 'Data and Reporting Engineer')}},
    {"query": "Data analyst",
     "relevant": {('MicroSourcing', 'Data Analyst (pooling)'), ('Concentrix', 'BI Analyst'), ('Pru Life UK', 'Data Analytics Expert'),
                                 ('First Circle', 'Data Analytics Lead'), ('Tech Mahindra', 'Data Analytics Manager (WFM)')}},
    {"query": "AI-related roles", 
    "relevant": {('Accenture in the Philippines', 'AI / ML Engineer'), ("AI Rudder", "AI Data Operations Engineer (A37267)"), 
                 ('Accenture in the Philippines', 'AI Architecture'), ('Teoh Capital', 'AI Agent Engineer'), 
                ('Bank of the Philippine Islands (BPI)', 'LEAD AI ENGINEER'), ('Qualfon', 'AI Solutions Lead'),
                ('My Amazon Guy', 'AI Engineer')}}
]

class TestSearchPostingRetrieval:

    @pytest.mark.parametrize("case", RETRIEVAL_CASES, ids=[c["query"] for c in RETRIEVAL_CASES])

    def test_precision_recall_mrr_at_k(self, case):
        """
        Tests precision, recall, and reciprocal rank at k for the given query and relevant postings. 
        Records metrics in the collector for later summary. We also look at the ndcg metric to 
        evaluate ranking quality rather than just presence/absence of relevant postings.
        """

        results = search_postings(case["query"], n_results=100)
        retrieved = [_posting_id(r) for r in results]
        relevant = case["relevant"]

        # Compute MRR
        rr = reciprocal_rank(retrieved, relevant)
        collector.record("retrieval_rr", rr)

        # Compute precision and recall across standardized top-k cutoffs
        summary_parts = []
        
        for k in [3, 5]:
            p = precision_at_k(retrieved, relevant, k)
            r = recall_at_k(retrieved, relevant, k)
            ndcg = ndcg_at_k(retrieved, relevant, k)
            
            collector.record(f"retrieval_precision@{k}", p)
            collector.record(f"retrieval_recall@{k}", r)
            collector.record(f"retrieval_ndcg@{k}", ndcg)

            summary_parts.append(f"@k={k}: p={p:.2f}, r={r:.2f}, ndcg={ndcg:.2f}")

        print(f"\n[{case['query']}] rr={rr:.2f} | " + " | ".join(summary_parts))

        assert rr > 0, f"No relevant postings retrieved for: {case['query']}"


    def test_mean_mmr_meets_threshold(self): # threshold can be stricter
        """
        This test checks the mean reciprocal rank across all retrieval test cases. 
        We expect that 75% of the test cases will be passing which is why the threshold is set to 0.75.
        """
        avg_rr = collector.mean("retrieval_rr")
        print(f"\nAverage MRR: {avg_rr:.2f}/1 ({collector.count('retrieval_rr')} cases)")
        assert avg_rr >= 0.75, f"Average MRR ({avg_rr:.2f}) is below 0.75 threshold"