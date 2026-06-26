# agents/job_analyzer.py

import re

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from rag.prompt_templates import JOB_ANALYSIS_PROMPT

load_dotenv()

# Structure de la réponse de l'agent
class JobAnalysis(BaseModel):
    job_title: str
    sector: str
    experience_level: str
    company_context: str
    key_responsibilities: list[str]

class JobAnalyzerAgent:
    def __init__(self):
        self.llm = OllamaLLM(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("MODEL_NAME")
        )
        
        self.prompt = PromptTemplate(
            input_variables=["job_description"],
            template=JOB_ANALYSIS_PROMPT
            
        )
    
    def analyze(self, job_description: str) -> dict:
        """
        Analyzes a job description and returns structured information.
        
        Args:
            job_description: Raw text of the job offer
            
        Returns:
            dict with job analysis
        """
        chain = self.prompt | self.llm
        response = chain.invoke({"job_description": job_description})
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        
        if json_match:
            return json_match.group()

        
        
        return response