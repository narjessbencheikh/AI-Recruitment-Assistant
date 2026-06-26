# agents/skill_extractor.py

import re

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from rag.prompt_templates import SKILL_EXTRACTION_PROMPT

load_dotenv()

class SkillExtractorAgent:
    def __init__(self):
        self.llm = OllamaLLM(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("MODEL_NAME")
        )
        
        self.prompt = PromptTemplate(
            input_variables=["job_description"],
            template=SKILL_EXTRACTION_PROMPT
        )
    
    def extract(self, job_description: str) -> str:
        """
        Extracts skills from a job description.
        
        Args:
            job_description: Raw text of the job offer
            
        Returns:
            JSON string with categorized skills
        """
        chain = self.prompt | self.llm
        response = chain.invoke({"job_description": job_description})

        # Extract only the JSON block from the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        
        if json_match:
            return json_match.group()
        
        
        return response