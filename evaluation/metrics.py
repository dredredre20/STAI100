from collections import defaultdict

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

    def mean(self, name: str) -> float:
        vals = self._values.get(name, [])
        return sum(vals) / len(vals) if vals else 0.0

    def count(self, name: str) -> int:
        return len(self._values.get(name, []))

    def summary(self) -> dict:
        return {name: self.mean(name) for name in self._values}


collector = EvaluationMetrics() # Singleton instance of EvaluationMetrics to be used across all test files