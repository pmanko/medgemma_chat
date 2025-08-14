"""Hybrid RAG Module

This module implements hybrid retrieval combining semantic search 
with structured queries for enhanced clinical information retrieval.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .vector_db_manager import VectorDBManager
from .embedding_service import EmbeddingService
from ..config import llm_config, spark_config

logger = logging.getLogger(__name__)


@dataclass
class RAGSearchResult:
    """Container for RAG search results"""
    concept_ids: List[str]
    concept_names: List[str]
    similarity_scores: List[float]
    metadata: List[Dict]
    query: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "concept_ids": self.concept_ids,
            "concept_names": self.concept_names,
            "similarity_scores": self.similarity_scores,
            "metadata": self.metadata,
            "query": self.query,
            "num_results": len(self.concept_ids)
        }


class HybridRAG:
    """
    Implements hybrid retrieval combining semantic search with structured queries
    """
    
    def __init__(self,
                 vector_db: VectorDBManager,
                 embedding_service: EmbeddingService,
                 spark_conn: Optional[Any] = None):
        """
        Initialize the Hybrid RAG system.
        
        Args:
            vector_db: Vector database manager instance
            embedding_service: Embedding service instance
            spark_conn: Optional Spark connection for SQL queries
        """
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.spark_conn = spark_conn
        
        # Ensure vector DB collection is initialized
        self.vector_db.initialize_collection()
        
    async def semantic_search(self,
                             query: str,
                             n_results: int = 20,
                             scope: Optional[Dict] = None) -> RAGSearchResult:
        """
        Step 1: Semantic search over concept embeddings.
        
        Args:
            query: Natural language query
            n_results: Number of results to retrieve
            scope: Optional scope filters (facility_id, org_ids)
            
        Returns:
            RAGSearchResult with relevant concepts
        """
        try:
            logger.info(f"Performing semantic search for: {query}")
            
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query)
            
            # Build filter conditions if scope is provided
            where_clause = None
            if scope:
                if scope.get('facility_id'):
                    # In a real implementation, you'd filter by facility
                    # This would require facility info in concept metadata
                    pass
                if scope.get('concept_class'):
                    where_clause = {"class_id": scope['concept_class']}
            
            # Search vector database
            results = self.vector_db.search(
                query_embedding,
                n_results=n_results,
                where=where_clause
            )
            
            # Extract and format results
            concept_ids = []
            concept_names = []
            similarity_scores = []
            metadata = []
            
            for i, concept_id in enumerate(results.get('ids', [])):
                concept_ids.append(concept_id)
                
                # Extract metadata
                if results.get('metadatas'):
                    meta = results['metadatas'][i]
                    concept_names.append(meta.get('concept_name', 'Unknown'))
                    metadata.append(meta)
                else:
                    concept_names.append('Unknown')
                    metadata.append({})
                
                # Convert distance to similarity score (assuming cosine distance)
                if results.get('distances'):
                    # ChromaDB returns L2 distance for normalized vectors
                    # Convert to similarity score (1 - distance/2)
                    distance = results['distances'][i]
                    similarity = 1 - (distance / 2)
                    similarity_scores.append(float(similarity))
                else:
                    similarity_scores.append(0.0)
            
            logger.info(f"Found {len(concept_ids)} semantically similar concepts")
            
            return RAGSearchResult(
                concept_ids=concept_ids,
                concept_names=concept_names,
                similarity_scores=similarity_scores,
                metadata=metadata,
                query=query
            )
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise
    
    def _build_enhanced_sql(self,
                           original_query: str,
                           concept_ids: List[str],
                           table: str = "observation") -> str:
        """
        Build an enhanced SQL query with semantic concept constraints.
        
        Args:
            original_query: Original natural language query
            concept_ids: List of relevant concept IDs from semantic search
            table: Target table for query (default: observation)
            
        Returns:
            Enhanced SQL query string
        """
        # Convert concept IDs to SQL-safe format
        concept_list = ','.join([f"'{cid}'" for cid in concept_ids[:10]])  # Limit to top 10
        
        # Build base SQL with concept filtering
        sql = f"""
        SELECT 
            patient_id,
            concept_id,
            value_text,
            value_numeric,
            value_coded,
            obs_datetime,
            location_id
        FROM {table}
        WHERE concept_id IN ({concept_list})
        ORDER BY obs_datetime DESC
        LIMIT 100
        """
        
        return sql.strip()
    
    def _build_enhanced_fhir_query(self,
                                  original_query: str,
                                  concept_ids: List[str],
                                  concept_names: List[str]) -> str:
        """
        Build an enhanced FHIR query with semantic concept information.
        
        Args:
            original_query: Original natural language query
            concept_ids: List of relevant concept IDs
            concept_names: List of concept names
            
        Returns:
            Enhanced FHIR query parameters
        """
        # Build FHIR search parameters
        # This is a simplified example - real implementation would be more sophisticated
        
        # Use concept codes for FHIR query
        code_params = '|'.join(concept_ids[:5])  # Limit to top 5
        
        # Build FHIR query string
        fhir_query = f"/Observation?code={code_params}&_sort=-date&_count=50"
        
        return fhir_query
    
    async def augmented_retrieval(self,
                                 query: str,
                                 semantic_results: RAGSearchResult,
                                 retrieval_type: str = "sql") -> Dict:
        """
        Step 2: Use semantic results to enhance structured queries.
        
        Args:
            query: Original query
            semantic_results: Results from semantic search
            retrieval_type: Type of retrieval ("sql" or "fhir")
            
        Returns:
            Dictionary with augmented retrieval results
        """
        try:
            if not semantic_results.concept_ids:
                logger.warning("No concepts found in semantic search")
                return {
                    "semantic_context": semantic_results.to_dict(),
                    "structured_data": [],
                    "query_metadata": {
                        "original_query": query,
                        "concepts_found": 0,
                        "retrieval_type": retrieval_type
                    }
                }
            
            structured_data = None
            query_used = None
            
            if retrieval_type == "sql" and self.spark_conn:
                # Build and execute SQL query
                enhanced_sql = self._build_enhanced_sql(
                    query,
                    semantic_results.concept_ids
                )
                query_used = enhanced_sql
                
                # Execute SQL (placeholder - actual implementation would use Spark)
                # structured_data = await self._execute_spark_sql(enhanced_sql)
                structured_data = {"sql": enhanced_sql, "note": "SQL execution not implemented"}
                
            elif retrieval_type == "fhir":
                # Build FHIR query
                enhanced_fhir = self._build_enhanced_fhir_query(
                    query,
                    semantic_results.concept_ids,
                    semantic_results.concept_names
                )
                query_used = enhanced_fhir
                
                # Execute FHIR query (placeholder)
                structured_data = {"fhir_query": enhanced_fhir, "note": "FHIR execution not implemented"}
            
            else:
                logger.warning(f"Unknown retrieval type: {retrieval_type}")
                structured_data = []
            
            return {
                "semantic_context": semantic_results.to_dict(),
                "structured_data": structured_data,
                "query_metadata": {
                    "original_query": query,
                    "concepts_found": len(semantic_results.concept_ids),
                    "enhanced_query": query_used,
                    "retrieval_type": retrieval_type,
                    "top_concepts": list(zip(
                        semantic_results.concept_names[:5],
                        semantic_results.similarity_scores[:5]
                    ))
                }
            }
            
        except Exception as e:
            logger.error(f"Error in augmented retrieval: {e}")
            raise
    
    def build_synthesis_prompt(self,
                              query: str,
                              semantic_context: Dict,
                              structured_data: Any) -> List[Dict]:
        """
        Build a prompt for LLM synthesis of retrieved information.
        
        Args:
            query: Original user query
            semantic_context: Semantic search results
            structured_data: Structured query results
            
        Returns:
            List of message dictionaries for LLM
        """
        # Extract top concepts for context
        top_concepts = []
        if semantic_context.get('concept_names'):
            for name, score in zip(
                semantic_context['concept_names'][:5],
                semantic_context.get('similarity_scores', [])[:5]
            ):
                top_concepts.append(f"- {name} (relevance: {score:.2f})")
        
        concepts_text = "\n".join(top_concepts) if top_concepts else "No specific concepts identified"
        
        # Format structured data (simplified)
        if isinstance(structured_data, dict):
            data_text = f"Query generated: {structured_data.get('sql', structured_data.get('fhir_query', 'N/A'))}"
        else:
            data_text = str(structured_data)[:500]  # Limit length
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical assistant analyzing patient data. "
                    "Use the provided semantic context and structured data to answer the query. "
                    "Be specific and cite relevant concepts when applicable."
                )
            },
            {
                "role": "user",
                "content": f"""
Query: {query}

Relevant Medical Concepts Found:
{concepts_text}

Structured Data Query:
{data_text}

Please provide a comprehensive answer based on the semantic search results and structured data context.
If the data is insufficient, explain what additional information would be needed.
"""
            }
        ]
        
        return messages
    
    async def full_rag_pipeline(self,
                               query: str,
                               n_results: int = 20,
                               retrieval_type: str = "sql",
                               scope: Optional[Dict] = None) -> Dict:
        """
        Execute the complete RAG pipeline.
        
        Args:
            query: User query
            n_results: Number of semantic search results
            retrieval_type: Type of structured retrieval
            scope: Optional scope filters
            
        Returns:
            Complete RAG results
        """
        try:
            logger.info(f"Starting full RAG pipeline for: {query}")
            
            # Step 1: Semantic search
            semantic_results = await self.semantic_search(query, n_results, scope)
            
            # Step 2: Augmented retrieval
            augmented_data = await self.augmented_retrieval(
                query,
                semantic_results,
                retrieval_type
            )
            
            # Step 3: Build synthesis prompt
            synthesis_prompt = self.build_synthesis_prompt(
                query,
                augmented_data['semantic_context'],
                augmented_data['structured_data']
            )
            
            # Return complete results
            return {
                "query": query,
                "semantic_results": augmented_data['semantic_context'],
                "structured_results": augmented_data['structured_data'],
                "query_metadata": augmented_data['query_metadata'],
                "synthesis_prompt": synthesis_prompt,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                "query": query,
                "error": str(e),
                "status": "error"
            }
    
    def explain_concepts(self, concept_ids: List[str]) -> List[Dict]:
        """
        Get detailed explanations for concepts.
        
        Args:
            concept_ids: List of concept IDs to explain
            
        Returns:
            List of concept explanations
        """
        try:
            # Retrieve concept details from vector DB
            results = self.vector_db.get_by_ids(concept_ids)
            
            explanations = []
            for i, cid in enumerate(concept_ids):
                if results.get('metadatas') and i < len(results['metadatas']):
                    meta = results['metadatas'][i]
                    explanation = {
                        "concept_id": cid,
                        "name": meta.get('concept_name', 'Unknown'),
                        "description": meta.get('description', 'No description'),
                        "synonyms": meta.get('synonyms', '').split('|') if meta.get('synonyms') else [],
                        "class": meta.get('class_id', 'Unknown'),
                        "mappings": meta.get('mappings', '')
                    }
                    explanations.append(explanation)
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error explaining concepts: {e}")
            return []


# Example usage
if __name__ == "__main__":
    import asyncio
    from ..config import vector_db_config, embedding_config
    
    async def test_hybrid_rag():
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize components
        vector_db = VectorDBManager(vector_db_config)
        embedding_service = EmbeddingService(embedding_config)
        
        # Create hybrid RAG
        rag = HybridRAG(vector_db, embedding_service)
        
        # Test query
        test_query = "Patient showing signs of hyperglycemia and diabetes"
        
        # Run semantic search
        semantic_results = await rag.semantic_search(test_query, n_results=10)
        print(f"Semantic search found {len(semantic_results.concept_ids)} concepts")
        print(f"Top concepts: {semantic_results.concept_names[:3]}")
        
        # Run full pipeline
        full_results = await rag.full_rag_pipeline(test_query)
        print(f"\nFull RAG pipeline results:")
        print(f"Status: {full_results['status']}")
        print(f"Concepts found: {full_results['query_metadata']['concepts_found']}")
    
    # Run test
    asyncio.run(test_hybrid_rag())