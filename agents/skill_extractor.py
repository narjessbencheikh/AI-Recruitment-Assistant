# agents/skill_extractor.py

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

class SkillExtractorAgent:
    def __init__(self):
        self.llm = OllamaLLM(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("MODEL_NAME")
        )
        
        self.prompt = PromptTemplate(
            input_variables=["job_description"],
            template="""
            You are an expert technical recruiter. Extract all skills from this job description.
            
            Categorize them as:
            1. Hard skills - Must have (technical, mandatory)
            2. Hard skills - Nice to have (technical, optional)
            3. Soft skills (communication, leadership, etc.)
            
            Job Description:
            {job_description}
            
            Respond in JSON format only:
            {{
                "must_have": ["skill1", "skill2"],
                "nice_to_have": ["skill1", "skill2"],
                "soft_skills": ["skill1", "skill2"]
            }}
            """
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
        return response