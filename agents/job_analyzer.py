# agents/job_analyzer.py

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv
import os

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
            template="""
            You are an expert HR analyst. Analyze this job description and extract:
            
            1. Job title
            2. Sector/Industry
            3. Experience level required (junior/mid/senior)
            4. Company context
            5. Key responsibilities (list of 3-5 points)
            
            Job Description:
            {job_description}
            
            Respond in JSON format only.
            """
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
        return response