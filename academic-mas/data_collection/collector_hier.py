#!/usr/bin/env python3
"""
Dataset Collector - Architecture HIERARCHIQUE
Collecte les métriques de l'architecture hiérarchique uniquement.
Version avec évaluateur de qualité NVIDIA Llama-3.1-70B
OPTIMISÉ : Timeout augmenté à 180 secondes
"""

import requests
import time
import re
import os
import json
import csv
from datetime import datetime
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# CHARGER .env
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION NVIDIA NIM
# ─────────────────────────────────────────────
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    print("❌ ERREUR: Clé API NVIDIA non trouvée dans .env")
    print("   Assurez-vous d'avoir un fichier .env avec: NVIDIA_API_KEY=ta_clé")
    exit(1)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"

# ─────────────────────────────────────────────
# CONFIGURATION HIERARCHIQUE
# ─────────────────────────────────────────────
API_URL = "http://localhost:8000/api/query"
CSV_PATH = "dataset_hierarchique.csv"
SESSION_BASE = "session_hier_"

# 🔥 TIMEOUT AUGMENTÉ (120 → 180 secondes)
API_TIMEOUT = 180  # secondes
TIMEOUT_MS = 180000  # millisecondes

# Poids pour le score global
W_TOKENS = 0.15
W_TIME = 0.15
W_HALLU = 0.15
W_TOOLS = 0.15
W_QUALITY = 0.40

HEADERS = [
    "question_id", "timestamp", "question_raw", "question_length",
    "nb_mots", "domaine_detecte", "type_question",
    "temps_ms", "tokens_total", "tokens_prompt",
    "tokens_completion", "nb_agents", "nb_outils",
    "reponse", "score_hallucination", "score_outils",
    "score_qualite", "score_global",
    "notes_annotateur"
]

def call_nvidia_llm(prompt: str, max_tokens: int = 100, timeout: int = 45) -> str:
    """Appelle l'API NVIDIA NIM pour l'évaluation de qualité"""
    
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": NVIDIA_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "top_p": 0.95
    }
    
    try:
        response = requests.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout  # 🔥 Timeout personnalisable
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        print(f"⚠️ Timeout NVIDIA LLM après {timeout}s")
        return "5.0"
    except Exception as e:
        print(f"⚠️ Erreur NVIDIA LLM: {e}")
        return "5.0"

def evaluate_quality_with_nvidia(question: str, response: str) -> float:
    """Évalue la qualité de la réponse avec NVIDIA Llama-3.1-70B"""
    
    if not response or "ERREUR" in response:
        return 0.0
    
    # Si la réponse est trop courte ou vide
    if len(response.strip()) < 10:
        return 0.0
    
    prompt = f"""Tu es un evaluateur de qualite pour un assistant academique. Note la reponse suivante de 0 a 10 selon ces criteres :

1. EXACTITUDE FACTUELLE (0-3) : La reponse contient-elle des erreurs ?
2. PERTINENCE (0-3) : Repond-elle precisement a la question ?
3. COMPLETUDE (0-2) : Manque-t-il des informations importantes ?
4. CLARTE (0-2) : Est-elle bien structuree et comprehensible ?

QUESTION: {question}
REPONSE A EVALUER:
{response[:2000]}

Reponds UNIQUEMENT par un nombre entre 0 et 10 (exemple: 7.5). Ne mets aucun autre texte."""

    try:
        score_text = call_nvidia_llm(prompt, max_tokens=50, timeout=30)
        match = re.search(r"(\d+(?:\.\d+)?)", score_text)
        if match:
            score = float(match.group(1))
            return min(10.0, max(0.0, score))
        return 5.0
    except Exception as e:
        print(f"⚠️ Erreur évaluation: {e}")
        return 5.0

def detect_domain(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["code", "python", "fonction", "bug", "programme", "javascript", "html", "css", "algorithme"]):
        return "code"
    if any(k in q for k in ["calcul", "math", "équation", "nombre", "somme", "suite", "arithmétique"]):
        return "math"
    if any(k in q for k in ["compare", "différence", "versus", "vs", "entre"]):
        return "comparatif"
    if any(k in q for k in ["explique", "comment", "pourquoi", "définition", "concept"]):
        return "analytique"
    if any(k in q for k in ["liste", "donne", "énumère", "quels sont"]):
        return "enumeration"
    return "general"

def detect_type(question: str) -> str:
    q = question.lower()
    if "?" not in question:
        return "instruction"
    if any(k in q for k in ["compare", "différence", "vs"]):
        return "comparative"
    if any(k in q for k in ["comment", "pourquoi", "explique"]):
        return "analytique"
    if any(k in q for k in ["qui", "quoi", "quand", "où"]):
        return "factuelle"
    return "ouverte"

def score_hallucination(response_text: str) -> float:
    if not response_text or "ERREUR" in response_text:
        return 0.0
    penalties = [
        r"\bje ne suis pas sûr\b", r"\bpeut-être\b", r"\bil est possible\b",
        r"\bje pense que\b", r"\bprobablement\b", r"\bje crois\b",
        r"\bà mon avis\b", r"\bpeut-être que\b", r"\bje suppose\b",
    ]
    score = 1.0
    for p in penalties:
        if re.search(p, response_text, re.IGNORECASE):
            score -= 0.12
    return max(0.0, round(score, 2))

def score_tools(nb_tools: int, domain: str) -> float:
    if domain in ["code", "math"] and nb_tools > 0:
        return min(1.0, 0.5 + nb_tools * 0.2)
    if domain == "general" and nb_tools == 0:
        return 0.8
    if nb_tools > 5:
        return 0.6
    return 0.7

def global_score(temps_ms, tokens, hallu_score, tools_score, quality_score,
                 ref_temps=5000, ref_tokens=2000) -> float:
    # 🔥 Normalisation améliorée
    s_temps = max(0, 1 - min(1, temps_ms / ref_temps))
    s_tokens = max(0, 1 - min(1, tokens / ref_tokens))
    quality_norm = quality_score / 10.0
    
    score = (W_TIME * s_temps +
             W_TOKENS * s_tokens +
             W_HALLU * hallu_score +
             W_TOOLS * tools_score +
             W_QUALITY * quality_norm)
    return round(score, 4)

def call_api(question: str, session_id: str) -> dict:
    payload = {"query": question, "session_id": session_id}
    start = time.time()

    try:
        resp = requests.post(API_URL, json=payload,
                             headers={"Content-Type": "application/json"},
                             timeout=API_TIMEOUT)  # 🔥 180 secondes
        elapsed_ms = round((time.time() - start) * 1000, 1)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return {"error": "timeout", "temps_ms": TIMEOUT_MS}  # 🔥 180000 ms
    except Exception as e:
        return {"error": str(e), "temps_ms": round((time.time() - start) * 1000, 1)}

    response_text = (data.get("final_answer") or 
                     data.get("response") or 
                     data.get("answer") or 
                     data.get("result") or 
                     data.get("message") or 
                     "[Pas de réponse]")

    agent_results = data.get("agent_results", [])
    tokens_total = 0
    tokens_prompt = 0
    tokens_compl = 0
    nb_outils = 0
    
    for agent in agent_results:
        agent_tokens = agent.get("tokens", {})
        tokens_total += agent_tokens.get("total_tokens", 0)
        tokens_prompt += agent_tokens.get("prompt_tokens", 0)
        tokens_compl += agent_tokens.get("completion_tokens", 0)
        if agent.get("agent_name") == "tools":
            output = agent.get("output", "")
            if output and "Aucun outil" not in output:
                nb_outils += 1

    return {
        "error": None,
        "temps_ms": elapsed_ms,
        "tokens_total": tokens_total,
        "tokens_prompt": tokens_prompt,
        "tokens_completion": tokens_compl,
        "nb_agents": len(agent_results) or 1,
        "nb_outils": nb_outils,
        "reponse": response_text,
    }

def append_row_csv(path: str, row_data: dict):
    try:
        file_exists = os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(HEADERS)
                print(f"[CSV] 📝 En-têtes créées dans {path}")
        
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row = [row_data.get(col, "") for col in HEADERS]
            writer.writerow(row)
        print(f"[CSV] ✅ Ligne {row_data['question_id']} ajoutée")
    except Exception as e:
        print(f"[CSV] ❌ Erreur: {e}")

class HierarchicalCollector:
    def __init__(self):
        self.question_counter = self._load_counter()

    def _load_counter(self) -> int:
        meta = "collector_hier_meta.json"
        if os.path.exists(meta):
            with open(meta) as f:
                return json.load(f).get("counter", 1)
        return 1

    def _save_counter(self):
        with open("collector_hier_meta.json", "w") as f:
            json.dump({"counter": self.question_counter}, f)

    def collect(self, question: str, notes: str = "") -> dict:
        qid = self.question_counter
        session_id = f"{SESSION_BASE}{qid}_{int(time.time())}"
        domain = detect_domain(question)
        q_type = detect_type(question)

        print(f"\n{'='*60}")
        print(f"[Q-{qid:04d}] {question[:80]}...")
        print(f"         Domaine: {domain} | Type: {q_type}")
        print(f"         Timeout: {API_TIMEOUT} secondes")
        print(f"{'='*60}")

        print("[API] Envoi de la requête...")
        raw = call_api(question, session_id)
        
        if raw.get("error"):
            print(f"❌ Erreur: {raw['error']}")
            raw["tokens_total"] = raw["tokens_prompt"] = raw["tokens_completion"] = 0
            raw["reponse"] = f"ERREUR: {raw['error']}"
            raw["nb_agents"] = 0
            raw["nb_outils"] = 0

        hallu = score_hallucination(raw.get("reponse", ""))
        tools = score_tools(raw.get("nb_outils", 0), domain)
        
        # 🔥 N'évalue la qualité que si la réponse n'est pas vide
        if raw.get("reponse") and "ERREUR" not in raw["reponse"] and len(raw["reponse"]) > 20:
            print("[EVAL] Évaluation qualité avec NVIDIA Llama-3.1-70B...")
            quality = evaluate_quality_with_nvidia(question, raw.get("reponse", ""))
            print(f"[EVAL] Score qualité: {quality}/10")
        else:
            print("[EVAL] ⚠️ Pas d'évaluation qualité (réponse invalide)")
            quality = 0.0
        
        score = global_score(raw.get("temps_ms", TIMEOUT_MS),
                             raw.get("tokens_total", 0),
                             hallu, tools, quality)

        status = "✅" if not raw.get("error") else "⚠️"
        print(f"[HIER] {status} {raw['temps_ms']}ms | {raw.get('tokens_total',0)} tokens | qualité={quality} | score={score}")

        row = {
            "question_id": qid,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question_raw": question,
            "question_length": len(question),
            "nb_mots": len(question.split()),
            "domaine_detecte": domain,
            "type_question": q_type,
            "temps_ms": raw.get("temps_ms", TIMEOUT_MS),
            "tokens_total": raw.get("tokens_total", 0),
            "tokens_prompt": raw.get("tokens_prompt", 0),
            "tokens_completion": raw.get("tokens_completion", 0),
            "nb_agents": raw.get("nb_agents", 0),
            "nb_outils": raw.get("nb_outils", 0),
            "reponse": raw.get("reponse", ""),
            "score_hallucination": hallu,
            "score_outils": tools,
            "score_qualite": quality,
            "score_global": score,
            "notes_annotateur": notes,
        }

        append_row_csv(CSV_PATH, row)
        self.question_counter += 1
        self._save_counter()
        return row

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   Dataset Collector — ARCHITECTURE HIERARCHIQUE                  ║")
    print("║   Évaluateur: NVIDIA Llama-3.1-70B-Instruct                      ║")
    print("║   Timeout API: 180 secondes                                      ║")
    print("║   Fichier: dataset_hierarchique.csv                              ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    print(f"  API URL : {API_URL}")
    print(f"  NVIDIA Model : {NVIDIA_MODEL}")
    print(f"  API Timeout : {API_TIMEOUT} secondes")
    print(f"  CSV output : {CSV_PATH}\n")

    collector = HierarchicalCollector()

    while True:
        try:
            question = input("❓ Question : ").strip()
            if not question:
                continue
            if question.lower() in ["quit", "exit", "q"]:
                print("\n✅ Session terminée.")
                break
            notes = input("📝 Notes (optionnel) : ").strip()
            collector.collect(question, notes)
        except KeyboardInterrupt:
            print("\n\n✅ Interrompu.")
            break

if __name__ == "__main__":
    main()