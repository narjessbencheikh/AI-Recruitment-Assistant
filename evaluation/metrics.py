# evaluation/metrics.py

from rag.retriever import CVRetriever
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGEvaluator:
    def __init__(self):
        self.retriever = CVRetriever()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # ============================================
    # RETRIEVAL METRICS
    # ============================================

    def precision_at_k(self, query: str, relevant_sources: list, k: int = 3) -> float:
        """
        Precision@K = combien de résultats retournés sont pertinents / K
        """
        results = self.retriever.retrieve(query, top_k=k)
        retrieved_sources = [r["source"] for r in results]
        relevant_retrieved = sum(1 for s in retrieved_sources if s in relevant_sources)
        return round(relevant_retrieved / k, 3)

    def recall_at_k(self, query: str, relevant_sources: list, k: int = 3) -> float:
        """
        Recall@K = combien de pertinents on a retrouvé / total pertinents
        """
        results = self.retriever.retrieve(query, top_k=k)
        retrieved_sources = [r["source"] for r in results]
        relevant_retrieved = sum(1 for s in retrieved_sources if s in relevant_sources)
        return round(relevant_retrieved / len(relevant_sources), 3)

    def mrr(self, query: str, relevant_sources: list, k: int = 3) -> float:
        """
        MRR = 1 / position du premier résultat pertinent
        """
        results = self.retriever.retrieve(query, top_k=k)
        for i, r in enumerate(results):
            if r["source"] in relevant_sources:
                return round(1 / (i + 1), 3)
        return 0.0

    # ============================================
    # GENERATION METRICS
    # ============================================

    def relevance_score(self, generated_cv: str, job_description: str) -> float:
        """
        Mesure la similarité sémantique entre le CV généré et le poste
        """
        embeddings = self.embedding_model.encode([generated_cv, job_description])
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return round(float(score), 3)

    def skill_match_score(self, generated_cv: str, required_skills: list) -> float:
        """
        % de compétences requises présentes dans le CV généré
        """
        cv_lower = generated_cv.lower()
        matched = sum(1 for skill in required_skills if skill.lower() in cv_lower)
        return round(matched / len(required_skills), 3)

    # ============================================
    # FULL EVALUATION REPORT
    # ============================================

    def evaluate(self, query: str, relevant_sources: list,
                 generated_cv: str, job_description: str,
                 required_skills: list, k: int = 3) -> dict:
        """
        Rapport complet des métriques
        """
        return {
            "retrieval": {
                "precision@k": self.precision_at_k(query, relevant_sources, k),
                "recall@k": self.recall_at_k(query, relevant_sources, k),
                "mrr": self.mrr(query, relevant_sources, k)
            },
            "generation": {
                "relevance_score": self.relevance_score(generated_cv, job_description),
                "skill_match_score": self.skill_match_score(generated_cv, required_skills)
            }
        }