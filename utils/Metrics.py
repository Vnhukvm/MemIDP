"""
Evaluation metrics for information diffusion prediction.
Computes Hits@K and MAP@K metrics.
"""
import numpy as np

class Metrics(object):
    """
    Calculator for evaluation metrics (Hits@K and MAP@K).
    """
    def __init__(self):
        """Initialize Metrics calculator with PAD token constant."""
        super().__init__()
        self.PAD = 0

    def apk(self, actual, predicted, k=10):
        """
        Compute Average Precision at K.
        
        Args:
            actual (list): List of actual target elements (only one in this task).
            predicted (list): List of predicted elements (ordered by confidence).
            k (int): Cutoff position. Defaults to 10.
        
        Returns:
            float: Average precision at K.
        """
        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        return score / min(len(actual), k)

    def compute_metric(self, y_prob, y_true, k_list=[10, 50, 100]):
        """
        Compute metrics (Hits@K and MAP@K) for multiple K values.
        
        Args:
            y_prob (list): Prediction scores for each candidate user.
                Shape: (#samples, #users)
            y_true (list): Ground truth user indices.
                Shape: (#samples,)
            k_list (list): List of K values to compute metrics at.
                Defaults to [10, 50, 100].
        
        Returns:
            tuple: (scores, scores_len) where:
                - scores: Dictionary containing mean metrics for all K.
                - scores_len: Number of non-padding samples evaluated.
        """
        scores_len = 0
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)

        scores = {'hits@'+str(k):[] for k in k_list}
        scores.update({'map@'+str(k):[] for k in k_list})
        for p_, y_ in zip(y_prob, y_true):
            if y_ != self.PAD:
                scores_len += 1.0
                p_sort = p_.argsort()
                for k in k_list:
                    topk = p_sort[-k:][::-1]
                    scores['hits@' + str(k)].extend([1. if y_ in topk else 0.])
                    scores['map@'+str(k)].extend([self.apk([y_], topk, k)])

        scores = {k: np.mean(v) for k, v in scores.items()}
        return scores, scores_len


