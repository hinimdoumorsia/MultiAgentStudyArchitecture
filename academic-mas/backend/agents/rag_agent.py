"""
RAGAgent — Couche 2, Agent 2

Rôle : recherche vectorielle dans une base de documents académiques.
Utilise ChromaDB (local, persistant) + Groq LLM.
Protocole A2A : reçoit le plan de PlanningAgent via l'état partagé.
"""

from typing import Dict, Any, List
import os
import logging
import time

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.base import BaseAgent
from backend.state import AcademicState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Tu es un agent de recherche documentaire académique (RAG).
Tu reçois une question et un contexte optionnel de documents.
Synthétise les informations pertinentes trouvées pour répondre à la question.
Si aucun document n'est disponible, indique-le clairement et propose une réponse basée sur tes connaissances.
Sois précis, cite tes sources quand disponibles, et reste factuel."""


class RAGAgent(BaseAgent):
    name = "rag"
    description = (
        "Effectue une recherche vectorielle dans la base documentaire académique "
        "et retourne les passages les plus pertinents avec leur score de similarité."
    )

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        # 🔥 Utilisation de Groq (disponible et fonctionnel)
        self.llm = ChatGroq(
            model=model,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=512
        )
        self.model = model
        self._chroma = None
        self._collection = None
        self._init_vector_store()

    def _init_vector_store(self):
        """Initialize ChromaDB. Graceful fallback if unavailable."""
        try:
            import chromadb
            db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
            os.makedirs(db_path, exist_ok=True)
            self._chroma = chromadb.PersistentClient(path=db_path)
            self._collection = self._chroma.get_or_create_collection(
                name="academic_docs",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"[RAGAgent] ChromaDB initialized at {db_path}")
        except Exception as e:
            logger.warning(f"[RAGAgent] ChromaDB unavailable ({e}), using fallback mode")

    def _search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search vector store. Returns empty list if unavailable."""
        if self._collection is None:
            return []
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, self._collection.count() or 1),
            )
            docs = []
            for i, doc in enumerate(results.get("documents", [[]])[0]):
                meta = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
                dist = results.get("distances", [[]])[0][i] if results.get("distances") else 0
                docs.append({
                    "content": doc,
                    "source": meta.get("source", "Document inconnu"),
                    "similarity": round(1 - dist, 3),
                })
            return docs
        except Exception as e:
            logger.warning(f"[RAGAgent] Search error: {e}")
            return []

    def add_document(self, content: str, source: str, doc_id: str = None):
        """Add a document to the vector store (called externally)."""
        if self._collection is None:
            raise RuntimeError("ChromaDB not available")
        import uuid
        self._collection.add(
            documents=[content],
            metadatas=[{"source": source}],
            ids=[doc_id or str(uuid.uuid4())],
        )

    def _get_fallback_response(self, query: str) -> str:
        """Réponse simple sans appel LLM (quand ChromaDB est indisponible)"""
        return f"Je n'ai pas pu accéder à la base documentaire. Pour répondre à '{query}', veuillez vérifier que ChromaDB est correctement installé et que des documents sont indexés."

    def process(self, state: AcademicState) -> Dict[str, Any]:
        query = state["user_query"]
        plan = state.get("plan", "")

        # Gérer le cas où plan est None
        plan_text = plan if plan is not None else ""

        # Enrich query with plan context (A2A communication via state)
        enriched_query = f"{query}\nContexte du plan : {plan_text[:300]}" if plan_text else query

        docs = self._search_documents(enriched_query)

        if docs:
            context = "\n\n".join(
                f"[Source: {d['source']} | Similarité: {d['similarity']}]\n{d['content']}"
                for d in docs
            )
            prompt = f"Question : {query}\n\nDocuments trouvés :\n{context}"
        else:
            prompt = (
                f"Question : {query}\n\n"
                "Aucun document spécifique trouvé dans la base. "
                "Réponds en utilisant tes connaissances académiques générales."
            )

        # Appel à Groq
        response = None
        for attempt in range(3):
            try:
                response = self.llm.invoke(prompt)
                break
            except Exception as e:
                logger.warning(f"[RAGAgent] Tentative {attempt+1}/3 échouée: {e}")
                if attempt < 2:
                    time.sleep(5)
                else:
                    return {
                        "retrieved_docs": self._get_fallback_response(query),
                        "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    }

        sources_info = (
            f"\n\n📚 **Sources consultées :** {len(docs)} document(s)"
            if docs else "\n\n📚 **Sources :** Connaissances générales (aucun document indexé)"
        )

        # Calcul des tokens approximatif (Groq ne retourne pas toujours l'usage)
        response_content = response.content if response else "Réponse non disponible"
        tokens_data = {
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": len(response_content) // 4,
            "total_tokens": (len(prompt) + len(response_content)) // 4
        }

        return {
            "retrieved_docs": response_content + sources_info,
            "tokens": tokens_data
        }

    def get_metadata(self, state: AcademicState) -> Dict[str, Any]:
        count = 0
        if self._collection:
            try:
                count = self._collection.count()
            except Exception:
                pass
        return {"indexed_documents": count}