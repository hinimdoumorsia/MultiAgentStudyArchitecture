"""
ExampleCustomAgent — Template pour ajouter un nouvel agent

Copier ce fichier, renommer la classe, et l'enregistrer dans main.py.
L'architecture s'adapte automatiquement (routeur + graphe).

Pour l'activer :
    # Dans backend/main.py, après les autres registry.register() :
    from backend.agents.example_custom_agent import CitationAgent
    registry.register(CitationAgent())
    orchestrator.rebuild_graph()
"""

from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.base import BaseAgent
from backend.state import AcademicState


class CitationAgent(BaseAgent):
    """
    Agent de citation académique — exemple d'extension.

    Génère des citations APA/MLA/IEEE automatiquement.
    Démontre la scalabilité : ajout sans toucher aux autres agents.
    """

    name = "citation"
    description = "Génère des citations académiques (APA, MLA, IEEE) à partir des sources trouvées."

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.llm = ChatGroq(model=model, max_tokens=512)

    def process(self, state: AcademicState) -> Dict[str, Any]:
        # Reads RAG output via shared state (A2A communication)
        rag_docs = state.get("retrieved_docs", "")
        query = state["user_query"]

        if not rag_docs or "aucun document" in rag_docs.lower():
            return {"tool_results": (state.get("tool_results", "") or "") + "\n\n📖 **Citations :** Aucune source à citer."}

        prompt = (
            f"À partir de ces informations sur '{query}', "
            f"génère 2-3 citations académiques au format APA :\n{rag_docs[:600]}"
        )
        messages = [
            SystemMessage(content="Tu es un expert en citations académiques. Génère des citations APA précises et brèves."),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        existing = state.get("tool_results", "") or ""
        return {"tool_results": existing + f"\n\n📖 **Citations APA :**\n{response.content}"}

    def get_metadata(self, state: AcademicState) -> Dict[str, Any]:
        return {"citation_style": "APA"}
