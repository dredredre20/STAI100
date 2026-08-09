from collections import defaultdict

# Evaluation metrics for information retrieval tasks - precision, recall, reciprocal rank taken from week 11 lab nb
def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    return sum(1 for d in top_k if d in relevant_ids) / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    return sum(1 for d in top_k if d in relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    for i, d in enumerate(retrieved_ids, start=1):
        if d in relevant_ids:
            return 1 / i
    return 0.0

# Storage for the eval metrics which is used across all test files.
class EvaluationMetrics:
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