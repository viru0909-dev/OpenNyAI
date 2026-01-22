"""
Vector Store for Legal Documents
=================================
Vector database integration for storing and retrieving legal document embeddings.

Supports: Chroma (local), Milvus (distributed)
"""

from typing import Dict, List, Optional, Any
from pathlib import Path

from loguru import logger


class VectorStore:
    """
    Vector store interface for legal document embeddings.
    
    Supports multiple backends:
    - Chroma: Local, lightweight, perfect for development
    - Milvus: Distributed, scalable, for production
    """
    
    def __init__(
        self,
        backend: str = "chroma",
        collection_name: str = "legal_documents",
        persist_directory: str = "./data/vectorstore",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector store.
        
        Args:
            backend: Vector store backend ("chroma" or "milvus").
            collection_name: Name of the collection.
            persist_directory: Directory for persistent storage.
            embedding_model: Model for generating embeddings.
        """
        self.backend = backend
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.embedding_model_name = embedding_model
        
        self.client = None
        self.collection = None
        self.embedding_model = None
        
        logger.info(f"Initialized VectorStore with {backend} backend")
    
    def _load_embedding_model(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
    
    def connect(self):
        """Connect to the vector store backend."""
        if self.embedding_model is None:
            self._load_embedding_model()
        
        if self.backend == "chroma":
            self._connect_chroma()
        elif self.backend == "milvus":
            self._connect_milvus()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _connect_chroma(self):
        """Connect to Chroma vector store."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )
        
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(self.persist_directory)
        ))
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Indian Legal Documents"}
        )
        
        logger.info(f"Connected to Chroma collection: {self.collection_name}")
    
    def _connect_milvus(self):
        """Connect to Milvus vector store."""
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
        except ImportError:
            raise ImportError(
                "pymilvus is required. Install with: pip install pymilvus"
            )
        
        connections.connect("default", host="localhost", port="19530")
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # MiniLM dimension
        ]
        
        schema = CollectionSchema(fields, description="Legal document embeddings")
        
        try:
            self.collection = Collection(self.collection_name, schema)
        except:
            self.collection = Collection(self.collection_name)
        
        logger.info(f"Connected to Milvus collection: {self.collection_name}")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        if self.embedding_model is None:
            self._load_embedding_model()
        
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to the vector store.
        
        Args:
            texts: List of document texts.
            metadatas: Optional metadata for each document.
            ids: Optional IDs for each document.
        """
        if self.collection is None:
            self.connect()
        
        # Generate embeddings
        embeddings = self.embed(texts)
        
        # Generate IDs if not provided
        if ids is None:
            import hashlib
            ids = [hashlib.md5(t.encode()).hexdigest()[:16] for t in texts]
        
        # Add to collection
        if self.backend == "chroma":
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas or [{}] * len(texts),
                ids=ids
            )
        elif self.backend == "milvus":
            entities = [
                ids,
                texts,
                embeddings
            ]
            self.collection.insert(entities)
            self.collection.flush()
        
        logger.info(f"Added {len(texts)} documents to vector store")
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query.
            k: Number of results to return.
            filter_metadata: Optional metadata filter.
            
        Returns:
            List of results with text, score, and metadata.
        """
        if self.collection is None:
            self.connect()
        
        # Embed query
        query_embedding = self.embed([query])[0]
        
        if self.backend == "chroma":
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_metadata
            )
            
            return [
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": results["distances"][0][i] if results["distances"] else 0
                }
                for i in range(len(results["ids"][0]))
            ]
        
        elif self.backend == "milvus":
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            self.collection.load()
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=k,
                output_fields=["doc_id", "text"]
            )
            
            return [
                {
                    "id": hit.entity.get("doc_id"),
                    "text": hit.entity.get("text"),
                    "score": hit.distance
                }
                for hit in results[0]
            ]
    
    def delete(self, ids: List[str]):
        """Delete documents by ID."""
        if self.backend == "chroma":
            self.collection.delete(ids=ids)
        elif self.backend == "milvus":
            expr = f'doc_id in {ids}'
            self.collection.delete(expr)
        
        logger.info(f"Deleted {len(ids)} documents")
    
    def get_count(self) -> int:
        """Get the number of documents in the collection."""
        if self.collection is None:
            return 0
        
        if self.backend == "chroma":
            return self.collection.count()
        elif self.backend == "milvus":
            return self.collection.num_entities
    
    def persist(self):
        """Persist the vector store to disk."""
        if self.backend == "chroma" and self.client:
            self.client.persist()
            logger.info("Vector store persisted")


if __name__ == "__main__":
    print("VectorStore module ready")
    print("Supported backends: chroma, milvus")
