#!/usr/bin/env python3
"""
Quick-start script — vérifie l'environnement et lance un test rapide du pipeline.
Usage : python start.py
"""

import os
import sys

def check_env():
    print("🔍 Vérification de l'environnement...")
    errors = []

    # Python version
    if sys.version_info < (3, 10):
        errors.append(f"Python 3.10+ requis (actuel: {sys.version})")

    # API key
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        errors.append("GROQ_API_KEY manquante dans .env")

    if errors:
        print("\n❌ Erreurs :")
        for e in errors:
            print(f"  • {e}")
        sys.exit(1)

    print("✅ Environnement OK")


def run_test():
    print("\n🚀 Test du pipeline multi-agents...")
    print("─" * 50)

    from backend.orchestrator import AcademicOrchestrator
    orch = AcademicOrchestrator()

    print(f"📋 Agents enregistrés : {list(orch.get_registered_agents().keys())}")

    query = "Qu'est-ce que le théorème de Pythagore et comment l'utiliser ?"
    print(f"\n❓ Question de test : {query}")
    print("⏳ Exécution du pipeline...\n")

    result = orch.run(query, session_id="test-session")

    print("─" * 50)
    print(f"✅ Pipeline terminé en {result['total_latency_ms']:.0f}ms")
    print(f"🤖 Agents utilisés : {[r['agent_name'] for r in result['agent_results']]}")
    print(f"📊 Routage : {result['router_decision']['reasoning']}")

    verification = result.get("verification_report", {})
    if verification:
        print(f"🎯 Confiance : {verification.get('confidence_score', 0):.0%}")

    print("\n📝 Réponse :")
    print("─" * 50)
    answer = result.get("final_answer", "")
    print(answer[:800] + ("..." if len(answer) > 800 else ""))
    print("\n✨ Tout fonctionne ! Lancez maintenant :")
    print("  Backend : uvicorn backend.main:app --reload --port 8000")
    print("  Frontend : cd frontend && npm install && npm run dev")


if __name__ == "__main__":
    check_env()
    run_test()
