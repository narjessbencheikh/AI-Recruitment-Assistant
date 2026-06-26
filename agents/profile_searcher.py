# agents/profile_searcher.py

import re

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from rag.prompt_templates import BENCHMARK_PROFILE_PROMPT

load_dotenv()

class ProfileSearcherAgent:
    def __init__(self):
        self.llm = OllamaLLM(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("MODEL_NAME")
        )
        
        self.prompt = PromptTemplate(
            input_variables=["job_title", "must_have_skills", "experience_level"],
            template=BENCHMARK_PROFILE_PROMPT
        )
    
    def search(self, job_title: str, must_have_skills: list, experience_level: str) -> dict:
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
        json_match = re.search(r'\{.*\}', response, re.DOTALL)

        if json_match:
             return json_match.group()
        

        return response