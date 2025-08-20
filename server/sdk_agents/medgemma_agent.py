"""
MedGemma Medical Agent using A2A SDK
Provides medical Q&A capabilities using the MedGemma model
"""

from a2a import Agent
from a2a.types import AgentCard, Skill, InputSchema, OutputSchema
from a2a.decorators import skill
import httpx
import logging
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)

class MedGemmaAgent(Agent):
    """Medical Q&A agent using MedGemma model with A2A SDK"""
    
    def __init__(self):
        super().__init__()
        self.llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.med_model = os.getenv("MED_MODEL", "medgemma-2")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.http_client = httpx.AsyncClient(timeout=90.0)
        logger.info(f"MedGemma agent initialized with model: {self.med_model}")
    
    async def get_agent_card(self) -> AgentCard:
        """Return agent capabilities following A2A protocol"""
        return AgentCard(
            name="MedGemma Medical Assistant",
            description="Provides evidence-based medical information and answers to general medical questions",
            version="1.0.0",
            skills=[
                Skill(
                    name="answer_medical_question",
                    description="Answer general medical questions with clinical accuracy and appropriate disclaimers",
                    input_schema=InputSchema(
                        type="object",
                        properties={
                            "query": {
                                "type": "string", 
                                "description": "Medical question to answer"
                            },
                            "include_references": {
                                "type": "boolean",
                                "description": "Whether to include medical references",
                                "default": False
                            }
                        },
                        required=["query"]
                    ),
                    output_schema=OutputSchema(
                        type="object",
                        properties={
                            "answer": {"type": "string", "description": "Medical answer"},
                            "confidence": {"type": "number", "description": "Confidence score 0-1"},
                            "disclaimer": {"type": "string", "description": "Medical disclaimer"},
                            "references": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Medical references if requested"
                            }
                        },
                        required=["answer", "disclaimer"]
                    )
                )
            ]
        )
    
    @skill("answer_medical_question")
    async def answer_medical_question(
        self, 
        query: str, 
        include_references: bool = False
    ) -> Dict[str, Any]:
        """
        Process medical question using MedGemma model
        
        Args:
            query: Medical question to answer
            include_references: Whether to include medical references
            
        Returns:
            Dictionary with answer, confidence, disclaimer, and optional references
        """
        logger.info(f"Processing medical query: {query[:100]}...")
        
        try:
            # Prepare system prompt
            system_prompt = """You are MedGemma, a medical AI assistant trained on clinical literature. 
            Provide accurate, evidence-based medical information. 
            Be clear about limitations and always recommend consulting healthcare professionals for personal medical advice.
            """
            
            if include_references:
                system_prompt += "\nInclude relevant medical references or guidelines when applicable."
            
            # Call LLM via OpenAI-compatible API
            headers = {"Content-Type": "application/json"}
            if self.llm_api_key:
                headers["Authorization"] = f"Bearer {self.llm_api_key}"
            
            response = await self.http_client.post(
                f"{self.llm_base_url}/v1/chat/completions",
                json={
                    "model": self.med_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": 1000
                },
                headers=headers
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract answer from LLM response
            answer = data["choices"][0]["message"]["content"].strip()
            
            # Calculate confidence based on response characteristics
            confidence = self._calculate_confidence(answer, query)
            
            # Standard medical disclaimer
            disclaimer = (
                "This information is for educational purposes only and should not replace "
                "professional medical advice, diagnosis, or treatment. Always consult with "
                "a qualified healthcare provider for medical concerns."
            )
            
            result = {
                "answer": answer,
                "confidence": confidence,
                "disclaimer": disclaimer
            }
            
            # Add references if requested and found in answer
            if include_references:
                references = self._extract_references(answer)
                if references:
                    result["references"] = references
            
            logger.info(f"Successfully processed medical query with confidence: {confidence}")
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM API error: {e}")
            raise RuntimeError(f"Failed to get response from medical model: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing medical query: {e}")
            raise RuntimeError(f"Error processing medical query: {e}")
    
    def _calculate_confidence(self, answer: str, query: str) -> float:
        """
        Calculate confidence score based on answer characteristics
        
        Simple heuristic: longer, more detailed answers with medical terms
        indicate higher confidence
        """
        score = 0.5  # Base confidence
        
        # Increase for length and detail
        if len(answer) > 200:
            score += 0.1
        if len(answer) > 500:
            score += 0.1
            
        # Check for medical terminology indicators
        medical_indicators = [
            "diagnosis", "treatment", "symptom", "medication",
            "clinical", "patient", "condition", "syndrome"
        ]
        
        found_terms = sum(1 for term in medical_indicators if term.lower() in answer.lower())
        score += min(found_terms * 0.05, 0.2)
        
        # Check for uncertainty markers (decreases confidence)
        uncertainty_markers = [
            "might", "possibly", "unclear", "varies", "depends",
            "not certain", "may be", "could be"
        ]
        
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in answer.lower())
        score -= min(uncertainty_count * 0.05, 0.2)
        
        # Clamp between 0.1 and 0.95
        return max(0.1, min(0.95, score))
    
    def _extract_references(self, answer: str) -> list:
        """
        Extract medical references from answer text
        
        Look for patterns like guidelines, studies, or journal citations
        """
        references = []
        
        # Simple pattern matching for common reference formats
        import re
        
        # Look for guideline mentions
        guideline_pattern = r"(?:ACC/AHA|WHO|CDC|FDA|NIH|NICE)\s+[Gg]uidelines?"
        guidelines = re.findall(guideline_pattern, answer)
        references.extend(guidelines)
        
        # Look for journal citations (simplified)
        journal_pattern = r"(?:NEJM|JAMA|Lancet|BMJ|Nature Medicine)[\s,]+"
        journals = re.findall(journal_pattern, answer)
        references.extend(journals)
        
        # Look for year citations that might be studies
        study_pattern = r"(?:study|trial|research).*?\(\d{4}\)"
        studies = re.findall(study_pattern, answer, re.IGNORECASE)
        references.extend(studies[:3])  # Limit to 3 study references
        
        return list(set(references))  # Remove duplicates
    
    async def cleanup(self):
        """Cleanup resources on shutdown"""
        await self.http_client.aclose()
        logger.info("MedGemma agent cleanup completed")


# For standalone testing
if __name__ == "__main__":
    import asyncio
    from a2a.server import AgentServer
    import uvicorn
    
    async def main():
        agent = MedGemmaAgent()
        server = AgentServer(agent)
        
        # Test the agent directly
        card = await agent.get_agent_card()
        print(f"Agent: {card.name}")
        print(f"Skills: {[s.name for s in card.skills]}")
        
        # Test medical question
        result = await agent.answer_medical_question(
            "What are the common symptoms of hypertension?",
            include_references=True
        )
        print(f"\nAnswer: {result['answer'][:200]}...")
        print(f"Confidence: {result['confidence']}")
        print(f"References: {result.get('references', [])}")
        
        # Run server
        config = uvicorn.Config(
            server.app,
            host="0.0.0.0",
            port=9101,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    asyncio.run(main())
