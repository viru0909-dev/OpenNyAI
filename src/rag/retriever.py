"""
Legal Document Retriever
=========================
Retrieval component for finding relevant precedents and legal provisions.
"""

from typing import Dict, List, Optional

from loguru import logger


class LegalRetriever:
    """
    Retriever for legal documents with support for:
    - Semantic search
    - Hybrid search (semantic + keyword)
    - Metadata filtering (court, year, case type)
    """
    
    def __init__(
        self,
        vectorstore: "VectorStore",
        rerank: bool = True,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize the retriever.
        
        Args:
            vectorstore: VectorStore instance.
            rerank: Whether to use reranking.
            rerank_model: Cross-encoder model for reranking.
        """
        self.vectorstore = vectorstore
        self.rerank = rerank
        self.rerank_model_name = rerank_model
        self.reranker = None
        
        logger.info(f"Initialized LegalRetriever with rerank={rerank}")
    
    def _load_reranker(self):
        """Load the reranking model."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        
        self.reranker = CrossEncoder(self.rerank_model_name)
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        rerank_k: int = 5,
        filter_court: Optional[str] = None,
        filter_year: Optional[int] = None,
        filter_case_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents.
        
        Args:
            query: Search query.
            k: Number of candidates to retrieve.
            rerank_k: Number of final results after reranking.
            filter_court: Filter by court name.
            filter_year: Filter by year.
            filter_case_type: Filter by case type.
            
        Returns:
            List of retrieved documents.
        """
        # Build metadata filter
        metadata_filter = {}
        if filter_court:
            metadata_filter["court"] = filter_court
        if filter_year:
            metadata_filter["year"] = filter_year
        if filter_case_type:
            metadata_filter["case_type"] = filter_case_type
        
        filter_arg = metadata_filter if metadata_filter else None
        
        # Initial retrieval
        candidates = self.vectorstore.search(query, k=k, filter_metadata=filter_arg)
        
        if not candidates:
            return []
        
        # Rerank if enabled
        if self.rerank and len(candidates) > rerank_k:
            if self.reranker is None:
                self._load_reranker()
            
            # Prepare pairs for reranking
            pairs = [(query, c["text"]) for c in candidates]
            scores = self.reranker.predict(pairs)
            
            # Sort by reranker scores
            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = float(score)
            
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            candidates = candidates[:rerank_k]
        
        return candidates
    
    def retrieve_precedents(
        self,
        case_facts: str,
        k: int = 5,
        court_hierarchy: List[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant precedents for a case.
        
        Args:
            case_facts: Facts of the current case.
            k: Number of precedents to retrieve.
            court_hierarchy: Preferred court hierarchy (e.g., ["Supreme Court", "High Court"]).
            
        Returns:
            List of relevant precedents.
        """
        # Construct query focused on legal issues
        query = f"Legal precedent for: {case_facts}"
        
        all_precedents = []
        
        if court_hierarchy:
            # Retrieve from each court level
            for court in court_hierarchy:
                precedents = self.retrieve(
                    query=query,
                    k=k,
                    filter_court=court
                )
                all_precedents.extend(precedents)
        else:
            all_precedents = self.retrieve(query=query, k=k)
        
        # Deduplicate by ID
        seen = set()
        unique = []
        for p in all_precedents:
            if p["id"] not in seen:
                seen.add(p["id"])
                unique.append(p)
        
        return unique[:k]
    
    def retrieve_statutes(
        self,
        legal_issue: str,
        k: int = 5
    ) -> List[Dict]:
        """
        Retrieve relevant statutory provisions.
        
        Args:
            legal_issue: Legal issue description.
            k: Number of provisions to retrieve.
            
        Returns:
            List of relevant statutes.
        """
        query = f"Statutory provisions for: {legal_issue}"
        
        return self.retrieve(
            query=query,
            k=k,
            filter_metadata={"doc_type": "statute"}
        )


if __name__ == "__main__":
    print("LegalRetriever module ready")
