"""
ToolsAgent — Couche 2, Agent 3

Rôle : exécute des outils externes (calcul, web search, code, Wikipedia).
Utilise le protocole MCP pour l'intégration d'outils tiers.
Communique avec les autres agents via l'état partagé (A2A).
"""

from typing import Dict, Any, List
import math
import logging
import ast
import re
import io
import contextlib
import os  # ← AJOUTÉ pour os.getenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA  # ← CHANGÉ (remplace ChatGroq)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from backend.agents.base import BaseAgent
from backend.state import AcademicState

# 🔥 IMPORT DU SERVEUR MCP (AJOUTÉ)
from backend.mcp import mcp_server

logger = logging.getLogger(__name__)

# ── Tool definitions (MCP-compatible signatures) ──────────────────────

@tool
def calculator(expression: str) -> str:
    """
    Évalue une expression mathématique.
    Exemples : '2**10', 'math.sqrt(144)', 'sum([1,2,3,4,5])'
    """
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "sum": sum, "min": min, "max": max})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Résultat : {result}"
    except Exception as e:
        return f"Erreur de calcul : {e}"


@tool
def python_executor(code: str) -> str:
    """
    Exécute un snippet Python simple (sans imports dangereux).
    Retourne stdout ou le résultat.
    """
    # ✅ Correction 1 : Valider la syntaxe Python avec ast.parse
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"Erreur : Code Python invalide - {e}"
    
    # Vérification des imports dangereux
    forbidden = ["import os", "import sys", "open(", "exec(", "eval(", "__import__"]
    if any(f in code for f in forbidden):
        return "Erreur : code refusé pour des raisons de sécurité."
    
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": {"print": print, "range": range, "len": len,
                                          "list": list, "dict": dict, "str": str,
                                          "int": int, "float": float, "sum": sum,
                                          "min": min, "max": max, "sorted": sorted}})
        output = buf.getvalue()
        return output if output else "Exécution réussie (aucun output)"
    except Exception as e:
        return f"Erreur d'exécution : {e}"


@tool
def wikipedia_search(query: str) -> str:
    """
    Recherche sur Wikipedia (FR puis EN). Retourne un résumé.
    """
    try:
        import urllib.request
        import json
        encoded = urllib.parse.quote(query)
        for lang in ["fr", "en"]:
            url = (
                f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded}"
            )
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "AcademicMAS/1.0"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                    if data.get("extract"):
                        return f"[Wikipedia {lang.upper()}] {data['title']}:\n{data['extract']}"
            except Exception:
                continue
        return "Aucun résultat Wikipedia trouvé."
    except Exception as e:
        return f"Erreur Wikipedia : {e}"


import urllib.parse  # needed by wikipedia_search

# Outils locaux de base
AVAILABLE_TOOLS = {
    "calculator": calculator,
    "python_executor": python_executor,
    "wikipedia_search": wikipedia_search,
}

# 🔥 Récupérer les outils MCP (AJOUTÉ)
MCP_TOOLS = {}
try:
    for tool_name in mcp_server.list_tools():
        # Créer un wrapper qui appelle le serveur MCP
        MCP_TOOLS[tool_name] = lambda x, tn=tool_name: mcp_server.call_tool(tn, x)
    logger.info(f"[ToolsAgent] Outils MCP chargés: {list(MCP_TOOLS.keys())}")
except Exception as e:
    logger.warning(f"[ToolsAgent] Erreur chargement MCP: {e}")

# Fusion des outils locaux + MCP
ALL_TOOLS = {**AVAILABLE_TOOLS, **MCP_TOOLS}

# ✅ Correction 2 : Renforcer le prompt SYSTEM avec les outils MCP
def build_system_prompt() -> str:
    """Construit le prompt système dynamiquement avec tous les outils disponibles"""
    tools_desc = []
    for tool_name in ALL_TOOLS.keys():
        if tool_name == "calculator":
            tools_desc.append("- calculator(expression) : pour tout calcul mathématique. Exemple: '2+2', 'sqrt(16)'")
        elif tool_name == "python_executor":
            tools_desc.append("- python_executor(code) : pour exécuter du code Python valide")
        elif tool_name == "wikipedia_search":
            tools_desc.append("- wikipedia_search(query) : pour chercher des informations factuelles")
        elif tool_name == "latex_formatter":
            tools_desc.append("- latex_formatter(expression) : pour formater une expression mathématique en LaTeX")
        elif tool_name == "citation_formatter":
            tools_desc.append("- citation_formatter(citation) : pour formater une citation en style APA")
        else:
            tools_desc.append(f"- {tool_name}(input) : outil MCP supplémentaire")
    
    return f"""Tu es un agent d'exécution d'outils académiques.
Tu reçois une question et tu dois décider quels outils utiliser et dans quel ordre.

Outils disponibles :
{chr(10).join(tools_desc)}

RÈGLES IMPORTANTES:
- Pour python_executor, tu DOIS fournir du code Python valide, PAS une description en français
- Ne mets jamais de texte explicatif dans le paramètre de python_executor
- Si tu ne peux pas écrire du code valide, n'utilise PAS python_executor
- Le paramètre doit être une chaîne simple, pas une phrase complète

Réponds UNIQUEMENT en JSON :
{{
  "tools_to_use": [{{"tool": "nom_outil", "input": "paramètre"}}],
  "reasoning": "pourquoi ces outils"
}}

Si aucun outil n'est nécessaire, retourne {{"tools_to_use": [], "reasoning": "pas besoin d'outils"}}
"""

SYSTEM_PROMPT = build_system_prompt()


class ToolsAgent(BaseAgent):
    name = "tools"
    description = (
        "Exécute des outils externes : calculs mathématiques, code Python, "
        "recherche Wikipedia. Extensible via le protocole MCP."
    )

    # 🔥 CHANGEMENT ICI : modèle NVIDIA
    def __init__(self, model: str = "meta/llama-3.1-70b-instruct"):
        self.llm = ChatNVIDIA(
            model=model,
            api_key=os.getenv("NVIDIA_API_KEY"),
            max_tokens=1024,
            temperature=0.1
        )
        # 🔥 Utiliser TOUS les outils (locaux + MCP)
        self.tools = dict(ALL_TOOLS)
        logger.info(f"[ToolsAgent] Initialisé avec {len(self.tools)} outils: {list(self.tools.keys())}")

    def register_tool(self, name: str, fn) -> None:
        """Dynamically add an MCP tool at runtime."""
        self.tools[name] = fn
        logger.info(f"[ToolsAgent] Tool registered: {name}")

    def _decide_tools(self, query: str, plan: str) -> List[Dict]:
        import json
        prompt = f"Question : {query}\nPlan : {plan[:400]}"
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        raw = response.content
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("tools_to_use", [])
        return []

    # ✅ Correction 3 : Bloquer exécution si invalide
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
            
            # ✅ Validation supplémentaire pour python_executor
            if tool_name == "python_executor":
                # Vérifier que l'input ressemble à du code Python
                code_indicators = ["print", "=", "+", "-", "*", "/", "for", "while", "if", "def", "import", "from"]
                if not any(indicator in tool_input for indicator in code_indicators):
                    results.append(f"⚠️ **{tool_name}** ignoré: l'input n'est pas du code Python valide ('{tool_input[:50]}...')")
                    continue
                # Vérifier la syntaxe Python
                try:
                    ast.parse(tool_input)
                except SyntaxError as e:
                    results.append(f"⚠️ **{tool_name}** ignoré: erreur de syntaxe Python - {e}")
                    continue
            
            if tool_name in self.tools:
                logger.info(f"[ToolsAgent] Calling {tool_name}({tool_input!r})")
                try:
                    # 🔥 Utiliser le handler approprié (MCP ou local)
                    tool_fn = self.tools[tool_name]
                    result = tool_fn(tool_input) if callable(tool_fn) else str(tool_fn)
                    results.append(f"🔧 **{tool_name}**(`{tool_input}`) → {result}")
                except Exception as e:
                    results.append(f"🔧 **{tool_name}**(`{tool_input}`) → Erreur: {str(e)[:100]}")
            else:
                results.append(f"⚠️ Outil inconnu : {tool_name}")

        return {"tool_results": "\n\n".join(results) if results else "Aucun outil exécuté"}

    def get_metadata(self, state: AcademicState) -> Dict[str, Any]:
        return {"available_tools": list(self.tools.keys())}