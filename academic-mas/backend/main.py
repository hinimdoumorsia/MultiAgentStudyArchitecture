"""
FastAPI server — REST API for the Academic MAS

Endpoints :
  POST /api/query             → Architecture HIÉRARCHIQUE (orchestrateur central)
  POST /api/query/distributed → Architecture DISTRIBUÉE (peer-to-peer)
  POST /api/query/dynamic     → Architecture DYNAMIQUE (choisie par le routeur) 🔥 NOUVEAU
  GET  /api/agents            → list registered agents
  GET  /api/stats             → memory + performance stats
  GET  /api/history           → session conversation history
  POST /api/documents         → add a document to RAG
  GET  /api/health            → health check
  GET  /api/router/status     → État du routeur dynamique 🔥 NOUVEAU
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

# ================================================================
# 🔥🔥🔥 NOUVEAU : Import pour le routeur dynamique
# ================================================================
import joblib
import re
import numpy as np
from pathlib import Path

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

# ================================================================
# 🔥🔥🔥 NOUVEAU : Routeur dynamique (Logistic Regression)
# ================================================================

class DynamicRouter:
    """Routeur dynamique qui choisit entre hiérarchique et distribuée"""
    
    def __init__(self, model_path: str = "final_router_pipeline.pkl"):
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            logger.warning(f"⚠️ Modèle non trouvé: {model_path}")
            logger.info("   Routeur dynamique désactivé")
            self.pipeline = None
        else:
            self.pipeline = joblib.load(self.model_path)
            self.confidence_threshold = 0.55
            logger.info(f"✅ Routeur dynamique chargé: {self.pipeline['model_name']}")
    
    def _clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_features(self, question: str) -> dict:
        features = {}
        features['len_chars'] = len(question)
        features['len_words'] = len(question.split())
        features['avg_word_len'] = features['len_chars'] / max(features['len_words'], 1)
        
        math_keywords = ['sin', 'cos', 'tan', 'equation', 'derive', 'integrale', 
                        'suite', 'fraction', 'calculer', 'trigonométrie']
        features['math_keyword_count'] = sum(1 for kw in math_keywords if kw in question.lower())
        
        action_verbs = ['calculer', 'trouver', 'resoudre', 'determiner', 
                       'simplifier', 'convertir']
        features['action_verb_count'] = sum(1 for verb in action_verbs if verb in question.lower())
        
        features['has_formula'] = int(any(c in question for c in ['²', '^', '√', '=']))
        features['special_chars_count'] = sum(1 for c in question if c in '²^√=+-*/')
        features['has_numbers'] = int(any(c.isdigit() for c in question))
        
        return features
    
    def is_available(self) -> bool:
        return self.pipeline is not None
    
    def predict(self, question: str):
        if not self.is_available():
            raise RuntimeError("Routeur non disponible")
        
        question_clean = self._clean_text(question)
        tfidf_vec = self.pipeline['tfidf_vectorizer'].transform([question_clean]).toarray()
        
        feats = self._extract_features(question)
        extra_vec = np.array([[feats[col] for col in self.pipeline['feature_columns']]])
        extra_scaled = self.pipeline['feature_scaler'].transform(extra_vec)
        
        X_new = np.hstack([tfidf_vec, extra_scaled])
        
        pred_enc = self.pipeline['model'].predict(X_new)[0]
        pred_proba = self.pipeline['model'].predict_proba(X_new)[0]
        
        architecture = self.pipeline['label_encoder'].inverse_transform([pred_enc])[0]
        confidence = max(pred_proba)
        
        probabilities = {
            'hierarchique': pred_proba[1],
            'distribue': pred_proba[0]
        }
        
        return architecture, confidence, probabilities
    
    def route(self, question: str, session_id: str = None):
        """Route la question vers la bonne architecture"""
        architecture, confidence, probabilities = self.predict(question)
        
        logger.info(f"🔀 ROUTING: {question[:60]}... -> {architecture.upper()} (conf={confidence:.3f})")
        
        if architecture == "hierarchique":
            result = orchestrator.run(question, session_id)
        else:
            result = distributed_runner.run(question, session_id)
        
        result['routing'] = {
            'architecture': architecture,
            'confidence': confidence,
            'probabilities': probabilities,
            'router_model': self.pipeline['model_name']
        }
        
        return result


# 🔥🔥🔥 Initialisation du routeur dynamique
dynamic_router = DynamicRouter("models/final_router_pipeline.pkl")


# ── Pydantic models ────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class DocumentRequest(BaseModel):
    content: str
    source: str
    doc_id: Optional[str] = None


# 🔥🔥🔥 NOUVEAU : Model pour la requête unifiée
class UnifiedQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "architectures": {
            "hierarchique": "/api/query",
            "distribuee": "/api/query/distributed",
            "DYNAMIQUE": "/api/query/dynamic"
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
# 🔥🔥🔥 NOUVEAU : ENDPOINT DYNAMIQUE (avec routeur)
# ================================================================
@app.post("/api/query/dynamic")
def query_dynamic(request: UnifiedQueryRequest):
    """
    Architecture DYNAMIQUE - le routeur choisit automatiquement
    entre hiérarchique et distribuée selon la question.
    """
    if not request.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    
    if not dynamic_router.is_available():
        raise HTTPException(503, "Routeur dynamique non disponible")
    
    session_id = request.session_id or str(uuid.uuid4())
    result = dynamic_router.route(request.query, session_id)
    
    return result


# ================================================================
# 🔥🔥🔥 NOUVEAU : ENDPOINT POUR L'ÉTAT DU ROUTEUR
# ================================================================
@app.get("/api/router/status")
def router_status():
    """Vérifier l'état du routeur dynamique"""
    if not dynamic_router.is_available():
        return {
            "status": "unavailable",
            "message": "Modèle final_router_pipeline.pkl non trouvé"
        }
    
    return {
        "status": "available",
        "model_name": dynamic_router.pipeline['model_name'],
        "confidence_threshold": dynamic_router.confidence_threshold
    }


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