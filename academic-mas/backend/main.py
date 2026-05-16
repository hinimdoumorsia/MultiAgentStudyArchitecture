"""
FastAPI server — REST API for the Academic MAS

Endpoints :
  POST /api/query             → Architecture HIÉRARCHIQUE (orchestrateur central)
  POST /api/query/distributed → Architecture DISTRIBUÉE (peer-to-peer)
  GET  /api/agents            → list registered agents
  GET  /api/stats             → memory + performance stats
  GET  /api/history           → session conversation history
  POST /api/documents         → add a document to RAG
  GET  /api/health            → health check
"""

import logging
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.orchestrator import AcademicOrchestrator
from backend.memory.memory_manager import memory_manager

# ================================================================
# 🔥 NOUVEAU : Import pour l'architecture distribuée
# ================================================================
from backend.distributed.peer_to_peer_runner import PeerToPeerRunner

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Academic MAS API",
    description="Multi-Agent System for Academic Assistance - Comparaison Hiérarchique vs Distribuée",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# Initialisation des DEUX architectures
# ================================================================

# Architecture HIÉRARCHIQUE (existante)
orchestrator = AcademicOrchestrator()

# 🔥 Architecture DISTRIBUÉE (nouvelle)
distributed_runner = PeerToPeerRunner()

logger.info("✅ Architectures chargées : Hiérarchique + Distribuée")


# ── Pydantic models ────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class DocumentRequest(BaseModel):
    content: str
    source: str
    doc_id: Optional[str] = None


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "architectures": {
            "hierarchique": "/api/query",
            "distribuee": "/api/query/distributed"
        },
        "agents": orchestrator.get_registered_agents()
    }


# ================================================================
# ENDPOINT HIÉRARCHIQUE (existant)
# ================================================================
@app.post("/api/query")
def query(request: QueryRequest):
    """Architecture hiérarchique avec orchestrateur central"""
    if not request.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"[API Hierarchique] Query: {request.query[:80]} | session: {session_id}")
    result = orchestrator.run(request.query, session_id)
    return result


# ================================================================
# 🔥 NOUVEAU : ENDPOINT DISTRIBUÉ (peer-to-peer)
# ================================================================
@app.post("/api/query/distributed")
def query_distributed(request: QueryRequest):
    """
    Architecture distribuée (peer-to-peer) - sans orchestrateur central.
    Les agents communiquent via EventBus et s'auto-organisent.
    """
    if not request.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"[API Distribuee] Query: {request.query[:80]} | session: {session_id}")
    result = distributed_runner.run(request.query, session_id)
    return result


# ================================================================
# Autres endpoints (inchangés)
# ================================================================

@app.get("/api/agents")
def list_agents():
    return {
        "agents": orchestrator.get_registered_agents(),
        "count": len(orchestrator.get_registered_agents()),
    }


@app.get("/api/stats")
def stats():
    return memory_manager.get_stats()


@app.get("/api/history/{session_id}")
def history(session_id: str, limit: int = 10):
    return {
        "session_id": session_id,
        "history": memory_manager.persistent.get_session_history(session_id, limit),
    }


@app.post("/api/documents")
def add_document(request: DocumentRequest):
    from backend.agents.registry import registry
    rag = registry.get("rag")
    if rag is None:
        raise HTTPException(404, "RAG agent not registered")
    try:
        rag.add_document(request.content, request.source, request.doc_id)
        return {"status": "added", "source": request.source}
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)