"""Embedding Service Module

This module handles text-to-vector embeddings using sentence-transformers,
optimized for medical concept semantic search.
"""

import logging
from typing import List, Union, Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handle text-to-vector embeddings using sentence-transformers"""
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the embedding service with configuration.
        
        Args:
            config: Embedding configuration with model name and settings
        """
        self.config = config
        self.model: Optional[SentenceTransformer] = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.config.model_name}")
            self.model = SentenceTransformer(self.config.model_name)
            
            # Verify model dimension matches config
            test_embedding = self.model.encode("test")
            actual_dim = len(test_embedding)
            
            if actual_dim != self.config.dimension:
                logger.warning(
                    f"Model dimension ({actual_dim}) differs from config ({self.config.dimension}). "
                    f"Updating config to match model."
                )
                self.config.dimension = actual_dim
                
            logger.info(f"Model loaded successfully. Embedding dimension: {actual_dim}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def create_concept_text(self, row: pd.Series) -> str:
        """
        Create searchable text from concept data.
        Combines name, description, synonyms for rich semantic representation.
        
        Args:
            row: Pandas Series containing concept data
            
        Returns:
            Formatted text string for embedding
        """
        parts = []
        
        # Primary concept name
        if pd.notna(row.get('concept_name')):
            parts.append(f"Concept: {row['concept_name']}")
        
        # Description provides important context
        if pd.notna(row.get('description')) and row['description']:
            parts.append(f"Description: {row['description']}")
        
        # Synonyms help with alternative terminology
        if pd.notna(row.get('synonyms')) and row['synonyms']:
            synonyms = row['synonyms'].replace('|', ', ')
            parts.append(f"Also known as: {synonyms}")
        
        # Mappings to standard terminologies (SNOMED, ICD, etc.)
        if pd.notna(row.get('mappings')) and row['mappings']:
            mappings = row['mappings'].replace('|', ', ')
            parts.append(f"Mappings: {mappings}")
        
        # Class information for categorization
        if pd.notna(row.get('class_name')):
            parts.append(f"Category: {row['class_name']}")
            
        return " ".join(filter(None, parts))
    
    def batch_embed(self, 
                   texts: List[str], 
                   show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for batch of texts.
        
        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of embeddings with shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
            
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        try:
            # Process in batches for memory efficiency
            batch_size = self.config.batch_size
            embeddings = []
            
            # Create batches
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            
            # Process with progress bar if requested
            iterator = tqdm(batches, desc="Embedding batches") if show_progress else batches
            
            for batch in iterator:
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True  # Normalize for cosine similarity
                )
                embeddings.append(batch_embeddings)
            
            # Concatenate all embeddings
            all_embeddings = np.vstack(embeddings)
            
            logger.info(f"Generated {all_embeddings.shape[0]} embeddings of dimension {all_embeddings.shape[1]}")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query for search.
        
        Args:
            query: Query text string
            
        Returns:
            NumPy array of embedding with shape (embedding_dim,)
        """
        try:
            embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
    
    def embed_concepts_dataframe(self, 
                                concepts_df: pd.DataFrame,
                                text_column: Optional[str] = None,
                                show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a DataFrame of concepts.
        
        Args:
            concepts_df: DataFrame containing concept data
            text_column: Optional column name containing prepared text. 
                        If None, will generate text using create_concept_text
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of embeddings
        """
        if text_column and text_column in concepts_df.columns:
            texts = concepts_df[text_column].tolist()
        else:
            logger.info("Preparing concept texts for embedding...")
            texts = concepts_df.apply(self.create_concept_text, axis=1).tolist()
        
        return self.batch_embed(texts, show_progress=show_progress)
    
    def compute_similarity(self, 
                          query_embedding: np.ndarray,
                          document_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: Matrix of document embeddings
            
        Returns:
            Array of similarity scores
        """
        # Ensure query embedding is 2D for matrix multiplication
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute cosine similarity (assuming normalized embeddings)
        similarities = np.dot(document_embeddings, query_embedding.T).flatten()
        
        return similarities
    
    def find_similar_concepts(self,
                             query: str,
                             concept_embeddings: np.ndarray,
                             concepts_df: pd.DataFrame,
                             top_k: int = 10) -> pd.DataFrame:
        """
        Find most similar concepts to a query.
        
        Args:
            query: Query text
            concept_embeddings: Pre-computed concept embeddings
            concepts_df: DataFrame with concept metadata
            top_k: Number of top results to return
            
        Returns:
            DataFrame with top similar concepts and similarity scores
        """
        # Embed the query
        query_embedding = self.embed_query(query)
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, concept_embeddings)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Create results DataFrame
        results = concepts_df.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]
        
        return results
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.model:
            return {"error": "Model not initialized"}
        
        return {
            "model_name": self.config.model_name,
            "embedding_dimension": self.config.dimension,
            "batch_size": self.config.batch_size,
            "max_seq_length": self.model.max_seq_length,
            "model_card": str(self.model)
        }


# Example usage and testing
if __name__ == "__main__":
    from ..config import embedding_config
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create service
    service = EmbeddingService(embedding_config)
    
    # Test single embedding
    test_text = "Diabetes mellitus type 2 with hyperglycemia"
    embedding = service.embed_query(test_text)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding sample: {embedding[:5]}")
    
    # Test batch embedding
    test_texts = [
        "Hypertension",
        "Acute myocardial infarction",
        "Pneumonia",
        "COVID-19 infection"
    ]
    embeddings = service.batch_embed(test_texts)
    print(f"\nBatch embeddings shape: {embeddings.shape}")
    
    # Test similarity
    query = "High blood pressure"
    query_emb = service.embed_query(query)
    similarities = service.compute_similarity(query_emb, embeddings)
    
    print(f"\nSimilarity scores for '{query}':")
    for text, score in zip(test_texts, similarities):
        print(f"  {text}: {score:.3f}")
    
    # Model info
    print(f"\nModel info: {service.get_model_info()}")