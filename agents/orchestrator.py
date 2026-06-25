# agents/orchestrator.py

from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents.job_analyzer import JobAnalyzerAgent
from agents.skill_extractor import SkillExtractorAgent
from agents.profile_searcher import ProfileSearcherAgent
import json

# État partagé entre tous les agents
class RecruitmentState(TypedDict):
    job_description: str
    job_analysis: dict
    skills: dict
    benchmark_profile: str
    error: str

# Noeuds du graph
def analyze_job(state: RecruitmentState) -> RecruitmentState:
    print("🔍 Step 1/3 — Analyzing job description...")
    try:
        agent = JobAnalyzerAgent()
        result = agent.analyze(state["job_description"])
        if isinstance(result, str):
            result = json.loads(result)
        state["job_analysis"] = result
        state["error"] = ""
    except Exception as e:
        state["error"] = f"JobAnalyzer failed: {str(e)}"
    return state

def extract_skills(state: RecruitmentState) -> RecruitmentState:
    print("🛠️  Step 2/3 — Extracting skills...")
    try:
        agent = SkillExtractorAgent()
        result = agent.extract(state["job_description"])
        if isinstance(result, str):
            result = json.loads(result)
        state["skills"] = result
        state["error"] = ""
    except Exception as e:
        state["error"] = f"SkillExtractor failed: {str(e)}"
    return state

def search_profile(state: RecruitmentState) -> RecruitmentState:
    print("👤 Step 3/3 — Building benchmark profile...")
    try:
        agent = ProfileSearcherAgent()
        result = agent.search(
            job_title=state["job_analysis"].get("job_title", "Unknown"),
            must_have_skills=state["skills"].get("must_have", []),
            experience_level=state["job_analysis"].get("experience_level", "mid")
        )

        if isinstance(result, str):
            result = json.loads(result)
        state["benchmark_profile"] = result

        state["error"] = ""
    except Exception as e:
        state["error"] = f"ProfileSearcher failed: {str(e)}"
    return state

# Décision : continuer ou arrêter si erreur
def should_continue(state: RecruitmentState) -> str:
    if state.get("error"):
        print(f"❌ Error detected: {state['error']}")
        return "end"
    return "continue"

# Construction du graph
def build_graph() -> StateGraph:
    graph = StateGraph(RecruitmentState)
    
    # Ajout des noeuds
    graph.add_node("analyze_job", analyze_job)
    graph.add_node("extract_skills", extract_skills)
    graph.add_node("search_profile", search_profile)
    
    # Point d'entrée
    graph.set_entry_point("analyze_job")
    
    # Edges conditionnels — si erreur on stoppe
    graph.add_conditional_edges(
        "analyze_job",
        should_continue,
        {"continue": "extract_skills", "end": END}
    )
    graph.add_conditional_edges(
        "extract_skills",
        should_continue,
        {"continue": "search_profile", "end": END}
    )
    graph.add_edge("search_profile", END)
    
    return graph.compile()

class OrchestratorAgent:
    def __init__(self):
        self.graph = build_graph()
    
    def run(self, job_description: str) -> dict:
        """
        Orchestre les 3 agents via LangGraph.
        
        Args:
            job_description: Texte brut de l'offre d'emploi
            
        Returns:
            dict avec l'analyse complète
        """
        initial_state = RecruitmentState(
            job_description=job_description,
            job_analysis={},
            skills={},
            benchmark_profile="",
            error=""
        )
        
        result = self.graph.invoke(initial_state)
        return result