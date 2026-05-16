"""
RAGAgent — Couche 2, Agent 2

Rôle : recherche vectorielle dans une base de documents académiques.
Utilise ChromaDB (local, persistant) + Mistral Direct.
Protocole A2A : reçoit le plan de PlanningAgent via l'état partagé.
"""

from typing import Dict, Any, List
import os
import logging
import time

from mistralai import Mistral
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

    def __init__(self, model: str = "mistral-large-latest"):
        # 🔥 Client Mistral DIRECT (pas OpenRouter)
        self.llm = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
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

    def process(self, state: AcademicState) -> Dict[str, Any]:
        query = state["user_query"]
        plan = state.get("plan", "")

        # 🔥 Gérer le cas où plan est None
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

        # 🔥 Appel direct à Mistral
        response = None
        for attempt in range(3):
            try:
                response = self.llm.chat.complete(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=512
                )
                break
            except Exception as e:
                logger.warning(f"[RAGAgent] Tentative {attempt+1}/3 échouée: {e}")
                if attempt < 2:
                    time.sleep(5)
                else:
                    return {
                        "retrieved_docs": "Je n'ai pas pu traiter cette demande. Veuillez réessayer.\n\n📚 **Sources :** Aucune source disponible",
                        "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    }

        sources_info = (
            f"\n\n📚 **Sources consultées :** {len(docs)} document(s)"
            if docs else "\n\n📚 **Sources :** Connaissances générales (aucun document indexé)"
        )

        # 🔥 AJOUT DES TOKENS
        tokens_data = {}
        if response and hasattr(response, 'usage'):
            tokens_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        else:
            tokens_data = {
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(response.choices[0].message.content) // 4 if response else 0,
                "total_tokens": (len(prompt) + (len(response.choices[0].message.content) if response else 0)) // 4
            }

        return {
            "retrieved_docs": (response.choices[0].message.content if response else "Réponse non disponible") + sources_info,
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