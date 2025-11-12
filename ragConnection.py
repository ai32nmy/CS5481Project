"""
Chroma RAG Module
=================

A comprehensive Retrieval-Augmented Generation (RAG) module using ChromaDB
for efficient document storage, retrieval, and semantic search.

Date: 2025-11-12
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import yaml
import logging
import os
from typing import List, Dict, Optional, Union, Any
import uuid
from pathlib import Path


class ChromaRAG:
    """
    A RAG module that provides document storage and retrieval using ChromaDB.

    This class handles configuration management, document indexing, and
    semantic search operations with ChromaDB as the vector database backend.

    Attributes:
        config (dict): Configuration dictionary loaded from YAML file
        client: ChromaDB client instance
        collection: ChromaDB collection for storing documents
        logger: Logger instance for tracking operations
    """

    def __init__(self, config_path: str = "rag_config.yaml"):
        """
        Initialize the ChromaRAG module.

        Args:
            config_path: Path to the YAML configuration file

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.client = self._initialize_client()
        self.collection = self._get_or_create_collection()

        self.logger.info("ChromaRAG module initialized successfully")

    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary containing configuration settings
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging configuration.

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger("ChromaRAG")
        log_level = getattr(logging, self.config["logging"]["level"])
        logger.setLevel(log_level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # File handler
        log_file = self.config["logging"]["log_file"]
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    def _initialize_client(self) -> chromadb.Client:
        """
        Initialize ChromaDB client based on configuration.

        Returns:
            ChromaDB client instance
        """
        client_type = self.config["chroma"]["client_type"]

        if client_type == "persistent":
            persist_dir = self.config["chroma"]["persist_directory"]
            Path(persist_dir).mkdir(parents=True, exist_ok=True)

            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            self.logger.info(f"Initialized persistent client at {persist_dir}")
        else:
            client = chromadb.Client()
            self.logger.info("Initialized in-memory client")

        return client

    def _get_embedding_function(self):
        """
        Get the embedding function based on configuration.

        Returns:
            Embedding function instance
        """
        emb_type = self.config["chroma"]["embedding_function"]

        if emb_type == "sentence_transformers":
            model_name = self.config["chroma"]["embedding_model"]
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        elif emb_type == "openai":
            # Requires OPENAI_API_KEY environment variable
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=self.config["chroma"]["embedding_model"],
            )
        else:
            # Default embedding function
            return None

    def _get_or_create_collection(self):
        """
        Get existing collection or create a new one.

        Returns:
            ChromaDB collection instance
        """
        collection_name = self.config["chroma"]["collection_name"]
        embedding_function = self._get_embedding_function()

        try:
            collection = self.client.get_collection(
                name=collection_name, embedding_function=embedding_function
            )
            self.logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            metadata = {"hnsw:space": self.config["chroma"]["distance_metric"]}

            # Add HNSW parameters if available
            if "performance" in self.config and "hnsw" in self.config["performance"]:
                hnsw_config = self.config["performance"]["hnsw"]
                metadata.update(
                    {
                        "hnsw:construction_ef": hnsw_config.get("ef_construction", 200),
                        "hnsw:search_ef": hnsw_config.get("ef_search", 50),
                        "hnsw:M": hnsw_config.get("M", 16),
                    }
                )

            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata=metadata,
            )
            self.logger.info(f"Created new collection: {collection_name}")

        return collection

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Add documents to the collection.

        Args:
            documents: List of document texts to add
            metadatas: Optional list of metadata dictionaries for each document
            ids: Optional list of document IDs (auto-generated if not provided)

        Returns:
            Dictionary containing operation status and document IDs
        """
        try:
            # Auto-generate IDs if not provided
            if ids is None and self.config["indexing"]["auto_generate_ids"]:
                ids = [str(uuid.uuid4()) for _ in documents]

            # Handle batch size
            batch_size = self.config["indexing"]["batch_size"]
            total_docs = len(documents)
            added_ids = []

            for i in range(0, total_docs, batch_size):
                batch_end = min(i + batch_size, total_docs)
                batch_docs = documents[i:batch_end]
                batch_ids = ids[i:batch_end] if ids else None
                batch_meta = metadatas[i:batch_end] if metadatas else None

                self.collection.add(
                    documents=batch_docs, metadatas=batch_meta, ids=batch_ids
                )

                added_ids.extend(batch_ids)
                self.logger.info(
                    f"Added batch {i // batch_size + 1}: {len(batch_docs)} documents"
                )

            return {
                "status": "success",
                "message": f"Added {total_docs} documents",
                "ids": added_ids,
            }

        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}")
            return {"status": "error", "message": str(e), "ids": []}

    def query(
        self,
        query_texts: Union[str, List[str]],
        n_results: Optional[int] = None,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Query the collection for similar documents.

        Args:
            query_texts: Single query string or list of query strings
            n_results: Number of results to return (uses config default if None)
            where: Metadata filter dictionary
            where_document: Document content filter dictionary

        Returns:
            Dictionary containing query results with documents, metadatas, and distances
        """
        try:
            # Convert single query to list
            if isinstance(query_texts, str):
                query_texts = [query_texts]

            # Use config defaults if not specified
            if n_results is None:
                n_results = self.config["query"]["n_results"]

            if where is None:
                where = self.config["query"]["where_filter"]

            if where_document is None:
                where_document = self.config["query"]["where_document_filter"]

            # Determine what to include in results
            include = []
            if self.config["query"]["include_metadata"]:
                include.append("metadatas")
            if self.config["query"]["include_documents"]:
                include.append("documents")
            if self.config["query"]["include_distances"]:
                include.append("distances")

            # Execute query
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include,
            )

            self.logger.info(
                f"Query executed: {len(query_texts)} queries, {n_results} results each"
            )

            return {
                "status": "success",
                "results": results,
                "num_queries": len(query_texts),
                "n_results": n_results,
            }

        except Exception as e:
            self.logger.error(f"Error querying collection: {str(e)}")
            return {"status": "error", "message": str(e), "results": None}

    def update_documents(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Update existing documents in the collection.

        Args:
            ids: List of document IDs to update
            documents: Optional new document texts
            metadatas: Optional new metadata dictionaries

        Returns:
            Dictionary containing operation status
        """
        try:
            self.collection.update(ids=ids, documents=documents, metadatas=metadatas)

            self.logger.info(f"Updated {len(ids)} documents")

            return {"status": "success", "message": f"Updated {len(ids)} documents"}

        except Exception as e:
            self.logger.error(f"Error updating documents: {str(e)}")
            return {"status": "error", "message": str(e)}

    def delete_documents(self, ids: List[str]) -> Dict[str, Any]:
        """
        Delete documents from the collection.

        Args:
            ids: List of document IDs to delete

        Returns:
            Dictionary containing operation status
        """
        try:
            self.collection.delete(ids=ids)
            self.logger.info(f"Deleted {len(ids)} documents")

            return {"status": "success", "message": f"Deleted {len(ids)} documents"}

        except Exception as e:
            self.logger.error(f"Error deleting documents: {str(e)}")
            return {"status": "error", "message": str(e)}

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.

        Returns:
            Dictionary containing collection statistics and metadata
        """
        try:
            count = self.collection.count()

            return {
                "status": "success",
                "name": self.collection.name,
                "count": count,
                "metadata": self.collection.metadata,
            }

        except Exception as e:
            self.logger.error(f"Error getting collection info: {str(e)}")
            return {"status": "error", "message": str(e)}

    def reset_collection(self) -> Dict[str, Any]:
        """
        Delete all documents from the collection.

        WARNING: This operation cannot be undone!

        Returns:
            Dictionary containing operation status
        """
        try:
            # Delete the collection
            self.client.delete_collection(name=self.collection.name)

            # Recreate it
            self.collection = self._get_or_create_collection()

            self.logger.warning("Collection has been reset (all documents deleted)")

            return {"status": "success", "message": "Collection reset successfully"}

        except Exception as e:
            self.logger.error(f"Error resetting collection: {str(e)}")
            return {"status": "error", "message": str(e)}


# Example usage
if __name__ == "__main__":
    # Initialize the RAG module
    rag = ChromaRAG(config_path="rag_config.yaml")

    # Add some sample documents
    sample_docs = [
        "ChromaDB is a vector database designed for AI applications.",
        "Retrieval-Augmented Generation combines retrieval with language models.",
        "Vector embeddings represent text as numerical vectors.",
        "Semantic search finds documents based on meaning, not just keywords.",
    ]

    sample_metadata = [
        {"category": "database", "topic": "chroma"},
        {"category": "ai", "topic": "rag"},
        {"category": "nlp", "topic": "embeddings"},
        {"category": "search", "topic": "semantic"},
    ]

    # Add documents
    print("\n=== Adding Documents ===")
    result = rag.add_documents(documents=sample_docs, metadatas=sample_metadata)
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")

    # Get collection info
    print("\n=== Collection Info ===")
    info = rag.get_collection_info()
    print(f"Collection Name: {info['name']}")
    print(f"Document Count: {info['count']}")

    # Query the collection
    print("\n=== Querying ===")
    query_result = rag.query(query_texts="What is a vector database?", n_results=2)

    if query_result["status"] == "success":
        results = query_result["results"]
        print(f"\nQuery: 'What is a vector database?'")
        print(f"Found {len(results['documents'][0])} results:\n")

        for i, (doc, distance) in enumerate(
            zip(results["documents"][0], results["distances"][0])
        ):
            print(f"Result {i + 1} (distance: {distance:.4f}):")
            print(f"  {doc}")
            print()
