"""OpenMRS Concept Extractor Module

This module provides functionality to extract medical concepts from the OpenMRS MySQL database,
including concept names, descriptions, synonyms, and mappings for embedding generation.
"""

import logging
from typing import Optional, Dict, List
import pandas as pd
import pymysql
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from ..config import MySQLConfig

logger = logging.getLogger(__name__)


class ConceptExtractor:
    """Extract concepts from OpenMRS MySQL database for embedding generation"""
    
    def __init__(self, mysql_config: MySQLConfig):
        """
        Initialize the concept extractor with MySQL configuration.
        
        Args:
            mysql_config: MySQL database configuration
        """
        self.config = mysql_config
        self.engine: Optional[Engine] = None
        
    def _create_engine(self) -> Engine:
        """
        Create SQLAlchemy engine for MySQL connection.
        
        Returns:
            SQLAlchemy engine instance
        """
        connection_string = (
            f"mysql+pymysql://{self.config.username}:{self.config.password}"
            f"@{self.config.host}:{self.config.port}/{self.config.database}"
        )
        
        try:
            engine = create_engine(
                connection_string,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                echo=False
            )
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(f"Successfully connected to MySQL database: {self.config.database}")
            return engine
        except Exception as e:
            logger.error(f"Failed to connect to MySQL database: {e}")
            raise
    
    def get_connection(self) -> Engine:
        """Get or create database connection."""
        if self.engine is None:
            self.engine = self._create_engine()
        return self.engine
    
    def extract_concepts(self, 
                        limit: Optional[int] = None,
                        include_retired: bool = False) -> pd.DataFrame:
        """
        Extract concepts with their names, descriptions, and mappings.
        
        Args:
            limit: Optional limit on number of concepts to extract
            include_retired: Whether to include retired concepts
            
        Returns:
            DataFrame with columns: concept_id, concept_name, description, 
            synonyms, mappings, class_id, datatype_id
        """
        engine = self.get_connection()
        
        retired_filter = "" if include_retired else "WHERE c.retired = 0"
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        query = f"""
        SELECT 
            c.concept_id,
            cn.name as concept_name,
            cn.locale,
            cd.description,
            c.class_id,
            c.datatype_id,
            c.date_created,
            c.date_changed,
            GROUP_CONCAT(DISTINCT cs.name SEPARATOR '|') as synonyms,
            GROUP_CONCAT(DISTINCT 
                CONCAT(COALESCE(crs.name, 'Unknown'), ':', crm.source_code) 
                SEPARATOR '|'
            ) as mappings
        FROM concept c
        LEFT JOIN concept_name cn ON c.concept_id = cn.concept_id 
            AND cn.locale = 'en' 
            AND cn.concept_name_type = 'FULLY_SPECIFIED'
        LEFT JOIN concept_description cd ON c.concept_id = cd.concept_id 
            AND cd.locale = 'en'
        LEFT JOIN concept_name cs ON c.concept_id = cs.concept_id 
            AND cs.locale = 'en' 
            AND cs.voided = 0
            AND cs.concept_name_type != 'FULLY_SPECIFIED'
        LEFT JOIN concept_reference_map crm ON c.concept_id = crm.concept_id
        LEFT JOIN concept_reference_source crs ON crm.concept_source_id = crs.concept_source_id
        {retired_filter}
        GROUP BY c.concept_id, cn.name, cn.locale, cd.description, 
                 c.class_id, c.datatype_id, c.date_created, c.date_changed
        {limit_clause}
        """
        
        try:
            logger.info(f"Extracting concepts from OpenMRS database...")
            df = pd.read_sql(query, engine)
            
            # Clean up null values
            df['description'] = df['description'].fillna('')
            df['synonyms'] = df['synonyms'].fillna('')
            df['mappings'] = df['mappings'].fillna('')
            
            logger.info(f"Successfully extracted {len(df)} concepts")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            raise
    
    def extract_concept_sets(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Extract concept set relationships for hierarchical understanding.
        
        Args:
            limit: Optional limit on number of relationships to extract
            
        Returns:
            DataFrame with concept set relationships
        """
        engine = self.get_connection()
        
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        query = f"""
        SELECT 
            cs.concept_set as parent_concept_id,
            parent.name as parent_concept_name,
            cs.concept_id as child_concept_id,
            child.name as child_concept_name,
            cs.sort_weight
        FROM concept_set cs
        JOIN concept_name parent ON cs.concept_set = parent.concept_id 
            AND parent.locale = 'en' 
            AND parent.concept_name_type = 'FULLY_SPECIFIED'
        JOIN concept_name child ON cs.concept_id = child.concept_id 
            AND child.locale = 'en' 
            AND child.concept_name_type = 'FULLY_SPECIFIED'
        ORDER BY cs.concept_set, cs.sort_weight
        {limit_clause}
        """
        
        try:
            logger.info("Extracting concept set relationships...")
            df = pd.read_sql(query, engine)
            logger.info(f"Successfully extracted {len(df)} concept set relationships")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting concept sets: {e}")
            raise
    
    def extract_concept_classes(self) -> pd.DataFrame:
        """
        Extract concept classes for better categorization.
        
        Returns:
            DataFrame with concept class information
        """
        engine = self.get_connection()
        
        query = """
        SELECT 
            concept_class_id,
            name,
            description
        FROM concept_class
        WHERE retired = 0
        """
        
        try:
            df = pd.read_sql(query, engine)
            logger.info(f"Extracted {len(df)} concept classes")
            return df
        except Exception as e:
            logger.error(f"Error extracting concept classes: {e}")
            raise
    
    def extract_concepts_by_class(self, 
                                  class_name: str, 
                                  limit: Optional[int] = None) -> pd.DataFrame:
        """
        Extract concepts filtered by a specific class.
        
        Args:
            class_name: Name of the concept class (e.g., 'Diagnosis', 'Drug', 'Test')
            limit: Optional limit on number of concepts
            
        Returns:
            DataFrame with filtered concepts
        """
        engine = self.get_connection()
        
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        query = f"""
        SELECT 
            c.concept_id,
            cn.name as concept_name,
            cd.description,
            cc.name as class_name,
            GROUP_CONCAT(DISTINCT cs.name SEPARATOR '|') as synonyms
        FROM concept c
        JOIN concept_class cc ON c.class_id = cc.concept_class_id
        LEFT JOIN concept_name cn ON c.concept_id = cn.concept_id 
            AND cn.locale = 'en' 
            AND cn.concept_name_type = 'FULLY_SPECIFIED'
        LEFT JOIN concept_description cd ON c.concept_id = cd.concept_id 
            AND cd.locale = 'en'
        LEFT JOIN concept_name cs ON c.concept_id = cs.concept_id 
            AND cs.locale = 'en' 
            AND cs.voided = 0
            AND cs.concept_name_type != 'FULLY_SPECIFIED'
        WHERE c.retired = 0 
            AND cc.name = %s
        GROUP BY c.concept_id, cn.name, cd.description, cc.name
        {limit_clause}
        """
        
        try:
            df = pd.read_sql(query, engine, params=[class_name])
            logger.info(f"Extracted {len(df)} concepts of class '{class_name}'")
            return df
        except Exception as e:
            logger.error(f"Error extracting concepts by class: {e}")
            raise
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the concept database.
        
        Returns:
            Dictionary with concept statistics
        """
        engine = self.get_connection()
        
        stats_queries = {
            "total_concepts": "SELECT COUNT(*) as count FROM concept WHERE retired = 0",
            "total_names": "SELECT COUNT(*) as count FROM concept_name WHERE voided = 0",
            "total_descriptions": "SELECT COUNT(*) as count FROM concept_description",
            "total_mappings": "SELECT COUNT(*) as count FROM concept_reference_map",
            "concept_classes": "SELECT COUNT(DISTINCT class_id) as count FROM concept WHERE retired = 0",
        }
        
        stats = {}
        try:
            for key, query in stats_queries.items():
                result = pd.read_sql(query, engine)
                stats[key] = int(result['count'][0])
            
            logger.info(f"Concept database statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            logger.info("Database connection closed")


# Example usage and testing
if __name__ == "__main__":
    from ..config import mysql_config
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create extractor
    extractor = ConceptExtractor(mysql_config)
    
    try:
        # Get statistics
        stats = extractor.get_statistics()
        print(f"Database statistics: {stats}")
        
        # Extract sample concepts
        concepts = extractor.extract_concepts(limit=10)
        print(f"\nSample concepts extracted: {len(concepts)}")
        print(concepts[['concept_id', 'concept_name']].head())
        
        # Extract concept classes
        classes = extractor.extract_concept_classes()
        print(f"\nConcept classes: {len(classes)}")
        print(classes.head())
        
    finally:
        extractor.close()