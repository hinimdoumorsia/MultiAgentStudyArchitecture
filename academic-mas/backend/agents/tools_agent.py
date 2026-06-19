"""
ToolsAgent — Couche 2, Agent 3

Rôle : exécute des outils externes (calcul, web search, code, Wikipedia).
Utilise le protocole MCP pour l'intégration d'outils tiers.
Communique avec les autres agents via l'état partagé (A2A).
Version avec TIMEOUTS pour éviter les blocages.
"""

from typing import Dict, Any, List
import math
import logging
import ast
import re
import io
import contextlib
import os
import subprocess
import signal
from functools import wraps
from concurrent.futures import TimeoutError as FuturesTimeoutError, ThreadPoolExecutor

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from backend.agents.base import BaseAgent
from backend.state import AcademicState

# 🔥 IMPORT DU SERVEUR MCP
from backend.mcp import mcp_server

logger = logging.getLogger(__name__)

# ================================================================
# 🔥 DÉCORATEUR DE TIMEOUT
# ================================================================
def timeout(seconds: int, default_return: str = "Timeout: l'opération a pris trop de temps"):
    """Décorateur pour ajouter un timeout à une fonction"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except FuturesTimeoutError:
                    logger.warning(f"[ToolsAgent] Timeout after {seconds}s: {func.__name__}")
                    return default_return
        return wrapper
    return decorator


# ── Tool definitions avec TIMEOUTS ──────────────────────────────────────

@tool
@timeout(seconds=10, default_return="Erreur: calcul trop long (>10s)")
def calculator(expression: str) -> str:
    """
    Évalue une expression mathématique.
    Exemples : '2**10', 'math.sqrt(144)', 'sum([1,2,3,4,5])'
    """
    try:
        if len(expression) > 500:
            return "Erreur: expression trop longue (>500 caractères)"
        
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "sum": sum, "min": min, "max": max})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Résultat : {result}"
    except Exception as e:
        return f"Erreur de calcul : {e}"


@tool
@timeout(seconds=20, default_return="Erreur: exécution Python trop longue (>20s)")
def python_executor(code: str) -> str:
    """
    Exécute un snippet Python simple (sans imports dangereux).
    Retourne stdout ou le résultat.
    """
    if len(code) > 5000:
        return "Erreur: code trop long (>5000 caractères). Limitez votre code."
    
    if code.count('\n') > 100:
        return "Erreur: code trop long (>100 lignes). Simplifiez votre code."
    
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"Erreur : Code Python invalide - {e}"
    
    forbidden = ["import os", "import sys", "open(", "exec(", "eval(", "__import__",
                 "subprocess", "shutil", "importlib"]
    if any(f in code for f in forbidden):
        return "Erreur : code refusé pour des raisons de sécurité."
    
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=20
        )
        
        os.unlink(temp_file)
        
        output = result.stdout
        error = result.stderr
        
        if error:
            return f"Erreur d'exécution : {error[:500]}"
        return output if output else "Exécution réussie (aucun output)"
        
    except subprocess.TimeoutExpired:
        return "Erreur : exécution Python trop longue (>20s). Simplifiez votre code."
    except Exception as e:
        return f"Erreur d'exécution : {e}"


@tool
@timeout(seconds=8, default_return="Erreur: recherche Wikipedia trop longue (>8s)")
def wikipedia_search(query: str) -> str:
    """
    Recherche sur Wikipedia (FR puis EN). Retourne un résumé.
    🔥 CORRECTION SSL : utilise requests avec verify=False
    """
    try:
        import requests
        import urllib.parse
        
        if len(query) > 200:
            query = query[:200]
        
        # Nettoyer la requête
        query = query.strip()
        if not query:
            return "Requête vide"
        
        encoded = urllib.parse.quote(query)
        
        for lang in ["fr", "en"]:
            url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded}"
            try:
                # 🔥 IGNORE SSL (verify=False)
                response = requests.get(
                    url, 
                    headers={"User-Agent": "AcademicMAS/1.0"},
                    timeout=5,
                    verify=False  # ← Correction SSL
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("extract"):
                        extract = data['extract'][:1000]
                        title = data.get('title', 'Sans titre')
                        return f"[Wikipedia {lang.upper()}] {title}:\n{extract}"
                    elif data.get("detail"):
                        continue
                elif response.status_code == 404:
                    continue
                else:
                    continue
                    
            except requests.exceptions.Timeout:
                continue
            except requests.exceptions.ConnectionError:
                continue
            except Exception:
                continue
                
        return "Aucun résultat Wikipedia trouvé."
        
    except ImportError:
        # Fallback si requests n'est pas installé
        return "Erreur: module requests non installé. Installez-le avec 'pip install requests'"
    except Exception as e:
        return f"Erreur Wikipedia : {e}"


@tool
@timeout(seconds=5, default_return="Erreur: formatage trop long (>5s)")
def latex_formatter(expression: str) -> str:
    """
    Formate une expression mathématique en LaTeX.
    """
    try:
        safe_expr = expression.replace('_', '\\_').replace('&', '\\&')
        return f"$${safe_expr}$$"
    except Exception as e:
        return f"Erreur de formatage : {e}"


@tool
@timeout(seconds=5, default_return="Erreur: formatage citation trop long (>5s)")
def citation_formatter(citation: str) -> str:
    """
    Formate une citation en style APA.
    """
    try:
        return f"📖 **Citation (APA)** : *{citation}*"
    except Exception as e:
        return f"Erreur de formatage : {e}"


import urllib.parse

# 🔥 Outils locaux de base - WIKIPEDIA DÉSACTIVÉ PAR DÉFAUT
AVAILABLE_TOOLS = {
    "calculator": calculator,
    "python_executor": python_executor,
    # "wikipedia_search": wikipedia_search,  # 🔥 DÉSACTIVÉ - plus d'erreur SSL
    "latex_formatter": latex_formatter,
    "citation_formatter": citation_formatter,
}

# 🔥 Récupérer les outils MCP
MCP_TOOLS = {}
try:
    for tool_name in mcp_server.list_tools():
        MCP_TOOLS[tool_name] = lambda x, tn=tool_name: mcp_server.call_tool(tn, x)
    if MCP_TOOLS:
        logger.info(f"[ToolsAgent] Outils MCP chargés: {list(MCP_TOOLS.keys())}")
except Exception as e:
    logger.warning(f"[ToolsAgent] Erreur chargement MCP: {e}")

# Fusion des outils locaux + MCP
ALL_TOOLS = {**AVAILABLE_TOOLS, **MCP_TOOLS}


def build_system_prompt() -> str:
    """Construit le prompt système dynamiquement avec tous les outils disponibles"""
    tools_desc = []
    for tool_name in ALL_TOOLS.keys():
        if tool_name == "calculator":
            tools_desc.append("- calculator(expression) : calcul mathématique simple. Ne dépasse pas 500 caractères.")
        elif tool_name == "python_executor":
            tools_desc.append("- python_executor(code) : exécute du code Python (max 100 lignes, 5000 caractères). DOIT être du code valide.")
        elif tool_name == "wikipedia_search":
            tools_desc.append("- wikipedia_search(query) : recherche Wikipedia (max 200 caractères).")
        elif tool_name == "latex_formatter":
            tools_desc.append("- latex_formatter(expression) : formate en LaTeX.")
        elif tool_name == "citation_formatter":
            tools_desc.append("- citation_formatter(citation) : formate une citation.")
        else:
            tools_desc.append(f"- {tool_name}(input) : outil MCP")
    
    return f"""Tu es un agent d'exécution d'outils académiques.

RÈGLES IMPORTANTES:
- Pour python_executor, tu DOIS fournir du code Python VALIDE, PAS une description
- Ne dépasse pas 100 lignes de code
- Utilise calculator pour les maths simples
- Utilise wikipedia_search pour les questions factuelles

Outils disponibles :
{chr(10).join(tools_desc)}

Réponds UNIQUEMENT en JSON :
{{"tools_to_use": [{{"tool": "nom_outil", "input": "paramètre"}}], "reasoning": "pourquoi"}}

Si aucun outil nécessaire : {{"tools_to_use": [], "reasoning": "pas besoin"}}
"""

SYSTEM_PROMPT = build_system_prompt()


class ToolsAgent(BaseAgent):
    name = "tools"
    description = (
        "Exécute des outils externes : calculs mathématiques, code Python, "
        "recherche Wikipedia. TIMEOUT intégrés : calc(10s), python(20s), wiki(8s)."
    )

    def __init__(self, model: str = "meta/llama-3.1-8b-instruct"):
        self.llm = ChatNVIDIA(
            model=model,
            api_key=os.getenv("NVIDIA_API_KEY"),
            max_tokens=512,
            temperature=0.1
        )
        self.tools = dict(ALL_TOOLS)
        logger.info(f"[ToolsAgent] Initialisé avec {len(self.tools)} outils (timeouts intégrés)")

    def register_tool(self, name: str, fn) -> None:
        """Ajoute dynamiquement un outil MCP"""
        self.tools[name] = fn
        logger.info(f"[ToolsAgent] Tool registered: {name}")

    def _decide_tools(self, query: str, plan: str) -> List[Dict]:
        """Décide quels outils utiliser"""
        import json
        import re
        
        prompt = f"Question : {query[:500]}\nPlan : {plan[:400]}"
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        
        try:
            response = self.llm.invoke(messages)
            raw = response.content
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return data.get("tools_to_use", [])
        except Exception as e:
            logger.warning(f"[ToolsAgent] Erreur décision: {e}")
        return []

    def process(self, state: AcademicState) -> Dict[str, Any]:
        query = state["user_query"]
        plan = state.get("plan", "")

        tool_calls = self._decide_tools(query, plan)

        if not tool_calls:
            return {"tool_results": "Aucun outil externe requis pour cette question."}

        results = []
        for call in tool_calls:
            tool_name = call.get("tool", "")
            tool_input = call.get("input", "")
            
            if tool_name == "python_executor":
                code_indicators = ["print", "=", "+", "-", "*", "/", "for", "while", "if", "def", "class"]
                if not any(indicator in tool_input for indicator in code_indicators):
                    results.append(f"⚠️ **{tool_name}** ignoré: ce n'est pas du code Python")
                    continue
                
                if len(tool_input) > 5000:
                    results.append(f"⚠️ **{tool_name}** ignoré: code trop long (>5000 caractères)")
                    continue
                
                if tool_input.count('\n') > 100:
                    results.append(f"⚠️ **{tool_name}** ignoré: trop de lignes (>100)")
                    continue
                
                try:
                    ast.parse(tool_input)
                except SyntaxError as e:
                    results.append(f"⚠️ **{tool_name}** ignoré: erreur syntaxe - {e}")
                    continue
            
            if tool_name in self.tools:
                logger.info(f"[ToolsAgent] Calling {tool_name}()")
                try:
                    tool_fn = self.tools[tool_name]
                    result = tool_fn(tool_input) if callable(tool_fn) else str(tool_fn)
                    if len(result) > 2000:
                        result = result[:2000] + "... (tronqué)"
                    results.append(f"🔧 **{tool_name}** → {result}")
                except Exception as e:
                    results.append(f"🔧 **{tool_name}** → Erreur: {str(e)[:200]}")
            else:
                results.append(f"⚠️ Outil inconnu : {tool_name}")

        final_result = "\n\n".join(results) if results else "Aucun outil exécuté"
        
        return {"tool_results": final_result}

    def get_metadata(self, state: AcademicState) -> Dict[str, Any]:
        return {"available_tools": list(self.tools.keys())}