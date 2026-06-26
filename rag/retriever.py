# rag/retriever.py

from sentence_transformers import SentenceTransformer
import chromadb

class CVRetriever:
    def __init__(self, vector_store_path: str = "data/vector_store"):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        self.client = chromadb.PersistentClient(path=vector_store_path)
        self.collection = self.client.get_or_create_collection(
            name="cvs",
            metadata={"hnsw:space": "cosine"}
        )
    
    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Recherche les chunks les plus similaires à la query.
        
        Args:
            query: texte de recherche (job description ou compétences)
            top_k: nombre de résultats à retourner
            
        Returns:
            liste de dicts avec le texte et les métadonnées
        """
        # 1. Transformer la query en embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # 2. Chercher les chunks similaires dans ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 3. Formater les résultats
        retrieved = []
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved.append({
                "text": doc,
                "source": meta["source"],
                "similarity_score": round(1 - distance, 3)  # cosine distance → similarity
            })
        
        return retrieved