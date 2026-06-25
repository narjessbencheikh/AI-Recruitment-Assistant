# agents/profile_searcher.py

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

class ProfileSearcherAgent:
    def __init__(self):
        self.llm = OllamaLLM(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("MODEL_NAME")
        )
        
        self.prompt = PromptTemplate(
            input_variables=["job_title", "must_have_skills", "experience_level"],
            template="""
            You are an expert recruiter doing benchmarking.
            Based on this job profile, describe what a successful candidate looks like.
            
            Job Title: {job_title}
            Required Skills: {must_have_skills}
            Experience Level: {experience_level}
            
            Provide a benchmark profile in JSON format only:
            {{
                "ideal_background": "...",
                "typical_experience_years": "...",
                "key_past_roles": ["...", "..."],
                "differentiating_factors": ["...", "..."],
                "red_flags": ["...", "..."]
            }}
            """
        )
    
    def search(self, job_title: str, must_have_skills: list, experience_level: str) -> str:
        """
        Génère un profil benchmark pour un poste donné.
        
        Args:
            job_title: Intitulé du poste
            must_have_skills: Liste des compétences obligatoires
            experience_level: Niveau d'expérience requis
            
        Returns:
            JSON string avec le profil benchmark
        """
        chain = self.prompt | self.llm
        response = chain.invoke({
            "job_title": job_title,
            "must_have_skills": ", ".join(must_have_skills),
            "experience_level": experience_level
        })
        return response