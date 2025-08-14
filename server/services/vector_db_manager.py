"""Vector Database Manager Module

This module manages ChromaDB operations for storing and retrieving
OpenMRS concept embeddings for semantic search.
"""

import os
import logging
from typing import List, Dict, Optional, Any
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from ..config import VectorDBConfig

logger = logging.getLogger(__name__)


class VectorDBManager:
    """Manage vector database operations using ChromaDB"""
    
    def __init__(self, config: VectorDBConfig):
        """
        Initialize the vector database manager.
        
        Args:
            config: Vector database configuration
        """
        self.config = config
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize ChromaDB client."""
        try:
            # Check if we should use HTTP client (for external ChromaDB service)
            chromadb_host = os.getenv("CHROMADB_HOST")
            chromadb_port = os.getenv("CHROMADB_PORT", "8000")
            chroma_auth_token = os.getenv("CHROMA_AUTH_TOKEN")
            
            if chromadb_host:
                # Use HTTP client for external ChromaDB service
                logger.info(f"Connecting to ChromaDB at {chromadb_host}:{chromadb_port}")
                
                if chroma_auth_token:
                    self.client = chromadb.HttpClient(
                        host=chromadb_host,
                        port=chromadb_port,
                        settings=Settings(
                            chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                            chroma_client_auth_credentials=chroma_auth_token
                        )
                    )
                else:
                    self.client = chromadb.HttpClient(
                        host=chromadb_host,
                        port=chromadb_port
                    )
            else:
                # Fall back to persistent client for local development
                logger.info(f"Initializing local ChromaDB client at {self.config.persist_directory}")
                
                self.client = chromadb.PersistentClient(
                    path=self.config.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            
            logger.info("ChromaDB client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def initialize_collection(self, 
                            embedding_dimension: Optional[int] = None,
                            recreate: bool = False) -> chromadb.Collection:
        """
        Create or get collection for OpenMRS concepts.
        
        Args:
            embedding_dimension: Dimension of embeddings (for metadata)
            recreate: Whether to recreate the collection if it exists
            
        Returns:
            ChromaDB collection instance
        """
        try:
            if recreate:
                # Delete existing collection if requested
                try:
                    self.client.delete_collection(name=self.config.collection_name)
                    logger.info(f"Deleted existing collection: {self.config.collection_name}")
                except Exception:
                    pass  # Collection might not exist
            
            # Create or get collection
            metadata = {
                "description": "OpenMRS concept embeddings for semantic search",
                "provider": self.config.provider
            }
            
            if embedding_dimension:
                metadata["embedding_dimension"] = embedding_dimension
            
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata=metadata
            )
            
            logger.info(f"Collection '{self.config.collection_name}' ready. "
                       f"Current count: {self.collection.count()}")
            
            return self.collection
            
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def add_embeddings(self,
                      embeddings: np.ndarray,
                      metadata: List[Dict],
                      ids: Optional[List[str]] = None,
                      documents: Optional[List[str]] = None,
                      batch_size: int = 100) -> int:
        """
        Store embeddings with metadata in the vector database.
        
        Args:
            embeddings: NumPy array of embeddings (n_samples, n_features)
            metadata: List of metadata dictionaries for each embedding
            ids: Optional list of unique IDs. If None, will use concept_id from metadata
            documents: Optional list of document strings
            batch_size: Batch size for adding embeddings
            
        Returns:
            Number of embeddings added
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call initialize_collection first.")
        
        # Prepare IDs
        if ids is None:
            ids = [str(meta.get('concept_id', i)) for i, meta in enumerate(metadata)]
        
        # Ensure embeddings are in the right format
        if isinstance(embeddings, np.ndarray):
            embeddings_list = embeddings.tolist()
        else:
            embeddings_list = embeddings
        
        # Prepare documents if not provided
        if documents is None:
            documents = [self._create_document_from_metadata(meta) for meta in metadata]
        
        total_added = 0
        
        try:
            # Process in batches for large datasets
            for i in range(0, len(embeddings_list), batch_size):
                batch_end = min(i + batch_size, len(embeddings_list))
                
                batch_data = {
                    "embeddings": embeddings_list[i:batch_end],
                    "metadatas": metadata[i:batch_end],
                    "ids": ids[i:batch_end],
                    "documents": documents[i:batch_end]
                }
                
                self.collection.add(**batch_data)
                batch_size_actual = batch_end - i
                total_added += batch_size_actual
                
                if (i + batch_size) % 1000 == 0:
                    logger.info(f"Added {i + batch_size_actual} embeddings...")
            
            logger.info(f"Successfully added {total_added} embeddings to collection")
            return total_added
            
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            raise
    
    def search(self,
              query_embedding: np.ndarray,
              n_results: int = 10,
              where: Optional[Dict] = None,
              include: List[str] = None) -> Dict:
        """
        Semantic search for similar concepts.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional filter conditions
            include: Fields to include in results (default: all)
            
        Returns:
            Dictionary with search results
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call initialize_collection first.")
        
        # Default include fields
        if include is None:
            include = ['metadatas', 'distances', 'documents']
        
        # Ensure embedding is in the right format
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=include
            )
            
            # Flatten results since we only have one query
            flattened_results = {
                'ids': results['ids'][0] if results['ids'] else [],
                'distances': results['distances'][0] if results['distances'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'documents': results['documents'][0] if 'documents' in results else []
            }
            
            return flattened_results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise
    
    def search_by_text(self,
                      query_text: str,
                      embedding_function,
                      n_results: int = 10,
                      where: Optional[Dict] = None) -> Dict:
        """
        Search using text query (requires embedding function).
        
        Args:
            query_text: Text query
            embedding_function: Function to convert text to embedding
            n_results: Number of results
            where: Optional filter conditions
            
        Returns:
            Search results
        """
        # Generate embedding for query
        query_embedding = embedding_function(query_text)
        return self.search(query_embedding, n_results, where)
    
    def get_by_ids(self, ids: List[str]) -> Dict:
        """
        Retrieve specific concepts by their IDs.
        
        Args:
            ids: List of concept IDs
            
        Returns:
            Dictionary with concept data
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call initialize_collection first.")
        
        try:
            results = self.collection.get(
                ids=ids,
                include=['metadatas', 'embeddings', 'documents']
            )
            return results
            
        except Exception as e:
            logger.error(f"Error getting concepts by IDs: {e}")
            raise
    
    def update_embeddings(self,
                         ids: List[str],
                         embeddings: Optional[np.ndarray] = None,
                         metadata: Optional[List[Dict]] = None,
                         documents: Optional[List[str]] = None):
        """
        Update existing embeddings and/or metadata.
        
        Args:
            ids: IDs of embeddings to update
            embeddings: New embeddings (optional)
            metadata: New metadata (optional)
            documents: New documents (optional)
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call initialize_collection first.")
        
        update_data = {"ids": ids}
        
        if embeddings is not None:
            if isinstance(embeddings, np.ndarray):
                update_data["embeddings"] = embeddings.tolist()
            else:
                update_data["embeddings"] = embeddings
        
        if metadata is not None:
            update_data["metadatas"] = metadata
        
        if documents is not None:
            update_data["documents"] = documents
        
        try:
            self.collection.update(**update_data)
            logger.info(f"Updated {len(ids)} embeddings")
            
        except Exception as e:
            logger.error(f"Error updating embeddings: {e}")
            raise
    
    def delete_embeddings(self, ids: List[str]):
        """
        Delete embeddings by IDs.
        
        Args:
            ids: List of IDs to delete
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call initialize_collection first.")
        
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} embeddings")
            
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            raise
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with database statistics
        """
        stats = {
            "collection_name": self.config.collection_name,
            "persist_directory": self.config.persist_directory,
            "provider": self.config.provider
        }
        
        if self.collection:
            stats.update({
                "total_embeddings": self.collection.count(),
                "collection_metadata": self.collection.metadata
            })
            
            # Get sample of concept classes if available
            try:
                sample = self.collection.get(limit=100, include=['metadatas'])
                if sample['metadatas']:
                    classes = set()
                    for meta in sample['metadatas']:
                        if 'class_id' in meta:
                            classes.add(meta['class_id'])
                    stats['sample_classes'] = list(classes)
            except:
                pass
        
        return stats
    
    def _create_document_from_metadata(self, metadata: Dict) -> str:
        """
        Create a searchable document string from metadata.
        
        Args:
            metadata: Concept metadata dictionary
            
        Returns:
            Document string
        """
        parts = []
        
        if 'concept_name' in metadata:
            parts.append(f"Name: {metadata['concept_name']}")
        
        if 'description' in metadata and metadata['description']:
            parts.append(f"Description: {metadata['description']}")
        
        if 'synonyms' in metadata and metadata['synonyms']:
            parts.append(f"Synonyms: {metadata['synonyms']}")
        
        if 'mappings' in metadata and metadata['mappings']:
            parts.append(f"Mappings: {metadata['mappings']}")
        
        return " | ".join(parts) if parts else "No description available"
    
    def clear_collection(self):
        """Clear all data from the current collection."""
        if self.collection:
            try:
                # Get all IDs and delete them
                all_data = self.collection.get()
                if all_data['ids']:
                    self.collection.delete(ids=all_data['ids'])
                    logger.info(f"Cleared {len(all_data['ids'])} items from collection")
                else:
                    logger.info("Collection is already empty")
            except Exception as e:
                logger.error(f"Error clearing collection: {e}")
                raise
    
    def reset_database(self):
        """Reset the entire database (use with caution)."""
        if self.client:
            try:
                self.client.reset()
                logger.warning("Database has been reset")
                self.collection = None
            except Exception as e:
                logger.error(f"Error resetting database: {e}")
                raise


# Example usage and testing
if __name__ == "__main__":
    from ..config import vector_db_config
    import numpy as np
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create manager
    manager = VectorDBManager(vector_db_config)
    
    # Initialize collection
    collection = manager.initialize_collection(embedding_dimension=384)
    
    # Create sample data
    sample_embeddings = np.random.randn(5, 384).astype(np.float32)
    sample_metadata = [
        {"concept_id": 1, "concept_name": "Hypertension", "class_id": "Diagnosis"},
        {"concept_id": 2, "concept_name": "Diabetes", "class_id": "Diagnosis"},
        {"concept_id": 3, "concept_name": "Aspirin", "class_id": "Drug"},
        {"concept_id": 4, "concept_name": "Blood Pressure", "class_id": "Test"},
        {"concept_id": 5, "concept_name": "Glucose Test", "class_id": "Test"}
    ]
    
    # Add embeddings
    added = manager.add_embeddings(sample_embeddings, sample_metadata)
    print(f"Added {added} embeddings")
    
    # Search
    query_embedding = np.random.randn(384).astype(np.float32)
    results = manager.search(query_embedding, n_results=3)
    print(f"\nSearch results: {results}")
    
    # Get statistics
    stats = manager.get_statistics()
    print(f"\nDatabase statistics: {stats}")