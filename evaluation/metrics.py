from collections import defaultdict
import math
import statistics

# Evaluation metrics for information retrieval tasks - precision, recall, reciprocal rank taken from week 11 lab nb

def precision_at_k(retrieved_ids, relevant_ids, k):
    """
    param retrieved_ids: list of retrieved document IDs 
    param relevant_ids: set of relevant document IDs
    param k: the cutoff rank for evaluation
    return computed precision at k

    Function for computing the precision at k given a set of documents retrieved by the model and relevant documents for the query.
    """
    top_k = retrieved_ids[:k]
    return sum(1 for d in top_k if d in relevant_ids) / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    """
    param retrieved_ids: list of retrieved document IDs 
    param relevant_ids: set of relevant document IDs
    param k: the cutoff rank for evaluation
    return computed recall at k
    
    Function for computing the recall at k given a set of documents retrieved by the model and relevant documents for the query.
    """

    top_k = retrieved_ids[:k]
    return sum(1 for d in top_k if d in relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    """
    param retrieved_ids: list of retrieved document IDs 
    param relevant_ids: set of relevant document IDs
    return computed reciprocal rank

    Function for computing the reciprocal rank given a set of documents retrieved by the model and relevant documents for the query.
    """
    for i, d in enumerate(retrieved_ids, start=1):
        if d in relevant_ids:
            return 1 / i
    return 0.0

def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """
    param retrieved_ids: list of retrieved document IDs 
    param relevant_ids: set of relevant document IDs
    param k: the cutoff rank for evaluation
    return computed normalized discounted cumulative gain at k

    Function for computing the normalized discounted cumulative gain at k given a set of documents retrieved by the model and relevant documents for the query.
    """
    dcg = 0.0
    idcg = 0.0
    for i in range(k):
        if i < len(retrieved_ids) and retrieved_ids[i] in relevant_ids:
            dcg += 1 / (math.log2(i + 2))  # log2(i + 2) because i is 0-indexed
        if i < len(relevant_ids):
            idcg += 1 / (math.log2(i + 2))
    return dcg / idcg if idcg > 0 else 0.0


# Storage for the eval metrics which is used across all test files.
class EvaluationMetrics:
    """
    Class to store evaluation metrics across all test files. Some of the test metrics won't need 
    precision, recall, or reciprocal rank to test the model, so we will use other measures such as mean and count to evaluate the model's performance. 
    This class will be used to store the metrics and provide a summary of the metrics at the end of the test run.
    """


    def __init__(self):
        self._values = defaultdict(list)

    def record(self, name: str, value: float):
        self._values[name].append(value)

    def count(self, name: str) -> int:
        return len(self._values.get(name, []))
    
    def mean(self, name: str) -> float:
        vals = self._values.get(name, [])
        return statistics.mean(vals) if vals else 0.0

    def stats(self, name: str) -> dict:
        vals = self._values.get(name, [])
        if not vals:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "count": len(vals),
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }

    def summary(self) -> dict:
        return {name: self.stats(name) for name in self._values}


collector = EvaluationMetrics() # Singleton instance of EvaluationMetrics to be used across all test files