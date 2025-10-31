import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging


class ItemBasedCF:
    def __init__(self, interactions_matrix):
        # interactions_matrix: np.array [users × items]
        self.sim_matrix = cosine_similarity(interactions_matrix.T)
        logging.info(
            f"Item-based CF: similarity matrix shape = {self.sim_matrix.shape}"
        )

    def recommend(self, user_vector, top_k=10):
        scores = user_vector @ self.sim_matrix
        top_items = np.argsort(scores)[::-1][:top_k]
        return top_items


class PopularityBaseline:
    def __init__(self, item_counts):
        self.item_counts = item_counts / item_counts.sum()

    def recommend(self, top_k=10):
        return np.argsort(-self.item_counts)[:top_k]
