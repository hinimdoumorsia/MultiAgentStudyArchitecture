#!/usr/bin/env python3
"""
Dataset Collector - Multi-Agent Architecture Router
Collecte les métriques des architectures hiérarchique et distribuée
et construit un dataset CSV de recherche (plus fiable qu'Excel).
Version avec évaluateur de qualité Mistral-Large
"""

import requests
import time
import re
import os
import json
import csv
from datetime import datetime
from mistralai import Mistral 
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# CHARGER .env
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION MISTRAL
# ─────────────────────────────────────────────
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("❌ ERREUR: Clé API Mistral non trouvée dans .env")
    print("   Assurez-vous d'avoir un fichier .env avec: MISTRAL_API_KEY=ta_clé")
    exit(1)

# Configuration du client Mistral
client = Mistral(api_key=MISTRAL_API_KEY)
MISTRAL_MODEL = "mistral-large-latest"

# ─────────────────────────────────────────────
# CONFIG — adapte ces URLs à tes deux serveurs
# ─────────────────────────────────────────────
HIER_URL     = "http://localhost:8000/api/query"
DIST_URL     = "http://localhost:8000/api/query/distributed"
CSV_PATH     = "dataset_routing.csv"  # ← Changé pour CSV
SESSION_BASE = "session_collect_"

# Poids pour le score global (somme = 1.0)
W_TOKENS  = 0.15
W_TIME    = 0.15
W_HALLU   = 0.15
W_TOOLS   = 0.15
W_QUALITY = 0.40

HEADERS_EXCEL = [
    "question_id", "timestamp", "question_raw", "question_length",
    "nb_mots", "domaine_detecte", "type_question",
    "hier_temps_ms", "hier_tokens_total", "hier_tokens_prompt",
    "hier_tokens_completion", "hier_nb_agents", "hier_nb_outils",
    "hier_reponse",
    "hier_score_hallucination", "hier_score_outils",
    "hier_score_qualite",
    "hier_score_global",
    "dist_temps_ms", "dist_tokens_total", "dist_tokens_prompt",
    "dist_tokens_completion", "dist_nb_agents", "dist_nb_outils",
    "dist_reponse",
    "dist_score_hallucination", "dist_score_outils",
    "dist_score_qualite",
    "dist_score_global",
    "delta_temps_ms", "delta_tokens", "delta_qualite",
    "gagnant_temps", "gagnant_tokens", "gagnant_qualite",
    "architecture_optimale",
    "confiance_label",
    "notes_annotateur",
]

def evaluate_quality_with_mistral(question: str, response: str, architecture: str) -> float:
    if not response or "ERREUR" in response:
        return 0.0
    
    prompt = f"""Tu es un evaluateur de qualite pour un assistant academique. Note la reponse suivante de 0 a 10 selon ces criteres :

1. EXACTITUDE FACTUELLE (0-3) : La reponse contient-elle des erreurs ?
2. PERTINENCE (0-3) : Repond-elle precisement a la question ?
3. COMPLETUDE (0-2) : Manque-t-il des informations importantes ?
4. CLARTE (0-2) : Est-elle bien structuree et comprehensible ?

QUESTION: {question}
ARCHITECTURE UTILISEE: {architecture}
REPONSE A EVALUER:
{response[:2000]}

Reponds UNIQUEMENT par un nombre entre 0 et 10 (exemple: 7.5). Ne mets aucun autre texte."""

    try:
        mistral_response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50
        )
        score_text = mistral_response.choices[0].message.content.strip()
        
        match = re.search(r"(\d+(?:\.\d+)?)", score_text)
        if match:
            score = float(match.group(1))
            return min(10.0, max(0.0, score))
        return 5.0
        
    except Exception as e:
        print(f"⚠️ Erreur Mistral: {e}")
        return 5.0

def detect_domain(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["code", "python", "fonction", "bug", "programme"]):
        return "code"
    if any(k in q for k in ["calcul", "math", "équation", "nombre", "somme"]):
        return "math"
    if any(k in q for k in ["compare", "différence", "versus", "vs", "entre"]):
        return "comparatif"
    if any(k in q for k in ["explique", "comment", "pourquoi", "définition"]):
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
    if not response_text:
        return 0.0
    penalties = [
        r"\bje ne suis pas sûr\b", r"\bpeut-être\b", r"\bil est possible\b",
        r"\bje pense que\b", r"\bprobablement\b", r"\bje crois\b",
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
    s_temps = max(0, 1 - temps_ms / ref_temps)
    s_tokens = max(0, 1 - tokens / ref_tokens)
    quality_norm = quality_score / 10.0
    
    score = (W_TIME * s_temps +
             W_TOKENS * s_tokens +
             W_HALLU * hallu_score +
             W_TOOLS * tools_score +
             W_QUALITY * quality_norm)
    return round(score, 4)

def decide_winner(hier_data: dict, dist_data: dict) -> tuple:
    h = hier_data["score_global"]
    d = dist_data["score_global"]
    delta = abs(h - d)

    if delta < 0.05:
        return "equivalent", 2
    if h > d:
        confiance = 5 if delta > 0.2 else (4 if delta > 0.1 else 3)
        return "hierarchique", confiance
    else:
        confiance = 5 if delta > 0.2 else (4 if delta > 0.1 else 3)
        return "distribuee", confiance

def call_architecture(url: str, question: str, session_id: str) -> dict:
    payload = {"query": question, "session_id": session_id}
    start = time.time()

    try:
        resp = requests.post(url, json=payload,
                             headers={"Content-Type": "application/json",
                                      "accept": "application/json"},
                             timeout=120)
        elapsed_ms = round((time.time() - start) * 1000, 1)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return {"error": "timeout", "temps_ms": 60000}
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

    nb_agents = len(agent_results)

    return {
        "error": None,
        "temps_ms": elapsed_ms,
        "tokens_total": tokens_total,
        "tokens_prompt": tokens_prompt,
        "tokens_completion": tokens_compl,
        "nb_agents": nb_agents if nb_agents > 0 else 1,
        "nb_outils": nb_outils,
        "reponse": response_text,
        "raw": data,
    }

def append_row_csv(path: str, row_data: dict, question_id: int):
    """Écriture directe en CSV (plus fiable qu'Excel)"""
    try:
        file_exists = os.path.exists(path)
        
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Écrire l'en-tête si le fichier n'existe pas
            if not file_exists:
                writer.writerow(HEADERS_EXCEL)
                print(f"[CSV] En-têtes écrites dans {path}")
            
            # Écrire les valeurs
            row = [row_data.get(col, "") for col in HEADERS_EXCEL]
            writer.writerow(row)
        
        # Compter les lignes
        with open(path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f) - 1  # -1 pour l'en-tête
        
        print(f"[CSV] ✅ Ligne {question_id} ajoutée (total: {line_count} lignes)")
        
    except Exception as e:
        print(f"[CSV] ❌ ERREUR écriture ligne {question_id}: {e}")

class DatasetCollector:
    def __init__(self):
        self.question_counter = self._load_counter()

    def _load_counter(self) -> int:
        meta = "collector_meta.json"
        if os.path.exists(meta):
            with open(meta) as f:
                return json.load(f).get("counter", 1)
        return 1

    def _save_counter(self):
        with open("collector_meta.json", "w") as f:
            json.dump({"counter": self.question_counter}, f)

    def collect(self, question: str, notes: str = "") -> dict:
        qid = self.question_counter
        session_id = f"{SESSION_BASE}{qid}_{int(time.time())}"
        domain = detect_domain(question)
        q_type = detect_type(question)

        print(f"\n{'='*60}")
        print(f"[Q-{qid:04d}] {question[:80]}...")
        print(f"         Domaine: {domain} | Type: {q_type}")
        print(f"{'='*60}")

        print("[HIER] Envoi de la requête...")
        hier_raw = call_architecture(HIER_URL, question, session_id + "_hier")
        if hier_raw.get("error"):
            print(f"[HIER] ❌ Erreur : {hier_raw['error']}")
            hier_raw["tokens_total"] = hier_raw["tokens_prompt"] = \
            hier_raw["tokens_completion"] = hier_raw["nb_agents"] = \
            hier_raw["nb_outils"] = 0
            hier_raw["reponse"] = f"ERREUR: {hier_raw['error']}"

        h_hallu = score_hallucination(hier_raw.get("reponse", ""))
        h_tools = score_tools(hier_raw.get("nb_outils", 0), domain)
        
        print("[EVAL] Évaluation qualité Hiérarchique avec Mistral...")
        h_quality = evaluate_quality_with_mistral(question, hier_raw.get("reponse", ""), "hierarchique")
        print(f"[EVAL] Score qualité Hiérarchique: {h_quality}/10")
        
        h_global = global_score(hier_raw.get("temps_ms", 99999),
                                hier_raw.get("tokens_total", 99999),
                                h_hallu, h_tools, h_quality)

        print(f"[HIER] ✅ {hier_raw['temps_ms']}ms | {hier_raw.get('tokens_total',0)} tokens | qualité={h_quality} | score={h_global}")

        print("[DIST] Attente de 15 secondes pour éviter rate limit...")
        time.sleep(15)

        print("[DIST] Envoi de la requête...")
        dist_raw = call_architecture(DIST_URL, question, session_id + "_dist")
        if dist_raw.get("error"):
            print(f"[DIST] ❌ Erreur : {dist_raw['error']}")
            dist_raw["tokens_total"] = dist_raw["tokens_prompt"] = \
            dist_raw["tokens_completion"] = dist_raw["nb_agents"] = \
            dist_raw["nb_outils"] = 0
            dist_raw["reponse"] = f"ERREUR: {dist_raw['error']}"

        d_hallu = score_hallucination(dist_raw.get("reponse", ""))
        d_tools = score_tools(dist_raw.get("nb_outils", 0), domain)
        
        print("[EVAL] Évaluation qualité Distribuée avec Mistral...")
        d_quality = evaluate_quality_with_mistral(question, dist_raw.get("reponse", ""), "distribuee")
        print(f"[EVAL] Score qualité Distribuée: {d_quality}/10")
        
        d_global = global_score(dist_raw.get("temps_ms", 99999),
                                dist_raw.get("tokens_total", 99999),
                                d_hallu, d_tools, d_quality)

        print(f"[DIST] ✅ {dist_raw['temps_ms']}ms | {dist_raw.get('tokens_total',0)} tokens | qualité={d_quality} | score={d_global}")

        archi_opt, confiance = decide_winner(
            {"score_global": h_global},
            {"score_global": d_global}
        )
        print(f"\n[VERDICT] → {archi_opt.upper()} (confiance: {confiance}/5)")

        row = {
            "question_id": qid,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question_raw": question,
            "question_length": len(question),
            "nb_mots": len(question.split()),
            "domaine_detecte": domain,
            "type_question": q_type,
            "hier_temps_ms": hier_raw.get("temps_ms", 0),
            "hier_tokens_total": hier_raw.get("tokens_total", 0),
            "hier_tokens_prompt": hier_raw.get("tokens_prompt", 0),
            "hier_tokens_completion": hier_raw.get("tokens_completion", 0),
            "hier_nb_agents": hier_raw.get("nb_agents", 0),
            "hier_nb_outils": hier_raw.get("nb_outils", 0),
            "hier_reponse": hier_raw.get("reponse", ""),
            "hier_score_hallucination": h_hallu,
            "hier_score_outils": h_tools,
            "hier_score_qualite": h_quality,
            "hier_score_global": h_global,
            "dist_temps_ms": dist_raw.get("temps_ms", 0),
            "dist_tokens_total": dist_raw.get("tokens_total", 0),
            "dist_tokens_prompt": dist_raw.get("tokens_prompt", 0),
            "dist_tokens_completion": dist_raw.get("tokens_completion", 0),
            "dist_nb_agents": dist_raw.get("nb_agents", 0),
            "dist_nb_outils": dist_raw.get("nb_outils", 0),
            "dist_reponse": dist_raw.get("reponse", ""),
            "dist_score_hallucination": d_hallu,
            "dist_score_outils": d_tools,
            "dist_score_qualite": d_quality,
            "dist_score_global": d_global,
            "delta_temps_ms": round(hier_raw.get("temps_ms", 0) - dist_raw.get("temps_ms", 0), 1),
            "delta_tokens": hier_raw.get("tokens_total", 0) - dist_raw.get("tokens_total", 0),
            "delta_qualite": round(h_quality - d_quality, 1),
            "gagnant_temps": "hier" if hier_raw.get("temps_ms", 99999) < dist_raw.get("temps_ms", 99999) else "dist",
            "gagnant_tokens": "hier" if hier_raw.get("tokens_total", 99999) < dist_raw.get("tokens_total", 99999) else "dist",
            "gagnant_qualite": "hier" if h_quality > d_quality else ("dist" if d_quality > h_quality else "equivalent"),
            "architecture_optimale": archi_opt,
            "confiance_label": confiance,
            "notes_annotateur": notes,
        }

        append_row_csv(CSV_PATH, row, qid)
        self.question_counter += 1
        self._save_counter()

        return row

def main():
    print("╔══════════════════════════════════════════════╗")
    print("║   Dataset Collector — Multi-Agent Router     ║")
    print("║   Évaluateur de qualité: Mistral-Large       ║")
    print("║   Fichier: dataset_routing.csv              ║")
    print("║   Tape 'quit' pour arrêter                   ║")
    print("╚══════════════════════════════════════════════╝\n")
    
    print(f"  Hiérarchique : {HIER_URL}")
    print(f"  Distribuée   : {DIST_URL}")
    print(f"  CSV output   : {CSV_PATH}\n")

    collector = DatasetCollector()

    while True:
        try:
            question = input("❓ Question : ").strip()
            if not question:
                continue
            if question.lower() in ["quit", "exit", "q"]:
                print("\n✅ Session terminée. Dataset sauvegardé.")
                break

            notes = input("📝 Notes (optionnel, Entrée pour passer) : ").strip()
            collector.collect(question, notes)

        except KeyboardInterrupt:
            print("\n\n✅ Interrompu. Dataset sauvegardé.")
            break

if __name__ == "__main__":
    main()