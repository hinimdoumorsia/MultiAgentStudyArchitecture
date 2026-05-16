# backend/mcp/server.py
"""
VRAI serveur MCP - Compatible avec l'architecture Academic MAS
- S'intègre avec le ToolsAgent existant
- Respecte le protocole MCP standard (JSON-RPC sur stdio)
- Peut tourner en mode direct (intégré) ou séparé (sous-processus)
"""

import sys
import json
import logging
import math
import urllib.request
import urllib.parse
import io
import contextlib
import ast
from typing import Dict, Any, Callable, Optional, List

logger = logging.getLogger(__name__)


class MCPServer:
    """
    VRAI serveur MCP compatible avec l'architecture Academic MAS.
    
    Deux modes de fonctionnement :
    1. Mode direct (intégré) : utilisé par ToolsAgent via import
    2. Mode stdio : serveur indépendant (pour Claude Desktop, etc.)
    """
    
    def __init__(self):
        self._tools: Dict[str, Dict] = {}
        self._running = False
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """Enregistre les outils compatibles avec ceux du ToolsAgent"""
        
        # Outil 1: Calculatrice (identique à celle du ToolsAgent)
        self.register_tool(
            name="calculator",
            description="Effectue des calculs mathématiques. Exemple: '2+2', 'sqrt(16)', '2**10'",
            handler=self._calculator,
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Expression mathématique"}
                },
                "required": ["expression"]
            }
        )
        
        # Outil 2: Exécution Python (identique à celle du ToolsAgent)
        self.register_tool(
            name="python_executor",
            description="Exécute du code Python sécurisé. Exemple: 'print(2+2)'",
            handler=self._python_executor,
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code Python à exécuter"}
                },
                "required": ["code"]
            }
        )
        
        # Outil 3: Recherche Wikipedia (identique à celle du ToolsAgent)
        self.register_tool(
            name="wikipedia_search",
            description="Recherche sur Wikipedia (FR/EN). Exemple: 'intelligence artificielle'",
            handler=self._wikipedia_search,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Terme à rechercher"}
                },
                "required": ["query"]
            }
        )
        
        # Outil 4: Latex formatter
        self.register_tool(
            name="latex_formatter",
            description="Formate une expression mathématique en LaTeX",
            handler=lambda args: f"$${args.get('expression', '')}$$",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Expression mathématique"}
                },
                "required": ["expression"]
            }
        )
        
        # Outil 5: Citation formatter
        self.register_tool(
            name="citation_formatter",
            description="Formate une citation en style APA",
            handler=lambda args: f"[{args.get('citation', '')}] (APA style)",
            input_schema={
                "type": "object",
                "properties": {
                    "citation": {"type": "string", "description": "Citation à formater"}
                },
                "required": ["citation"]
            }
        )
    
    def register_tool(self, name: str, description: str, handler: Callable, input_schema: Dict = None):
        """Enregistre un outil dans le serveur MCP"""
        self._tools[name] = {
            "name": name,
            "description": description,
            "handler": handler,
            "inputSchema": input_schema or {"type": "object"}
        }
        logger.info(f"[MCP] Tool registered: {name}")
    
    def _calculator(self, args: Dict) -> str:
        """Calcule une expression mathématique (identique au ToolsAgent)"""
        expression = args.get("expression", "")
        try:
            allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            allowed.update({"abs": abs, "round": round, "sum": sum, "min": min, "max": max})
            result = eval(expression, {"__builtins__": {}}, allowed)
            return f"Résultat : {result}"
        except Exception as e:
            return f"Erreur de calcul : {e}"
    
    def _python_executor(self, args: Dict) -> str:
        """Exécute du code Python (identique au ToolsAgent)"""
        code = args.get("code", "")
        
        # Validation syntaxique
        try:
            ast.parse(code)
        except SyntaxError as e:
            return f"Erreur : Code Python invalide - {e}"
        
        # Vérification sécurité
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
    
    def _wikipedia_search(self, args: Dict) -> str:
        """Recherche Wikipedia (identique au ToolsAgent)"""
        query = args.get("query", "")
        try:
            encoded = urllib.parse.quote(query)
            for lang in ["fr", "en"]:
                url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded}"
                req = urllib.request.Request(url, headers={"User-Agent": "AcademicMAS-MCP/1.0"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                    if data.get("extract"):
                        return f"[Wikipedia {lang.upper()}] {data['title']}:\n{data['extract'][:500]}"
            return "Aucun résultat Wikipedia trouvé."
        except Exception as e:
            return f"Erreur Wikipedia : {e}"
    
    # ─── Interface pour le ToolsAgent (compatible avec l'existant) ───
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """Retourne un handler d'outil (compatible avec l'interface existante)"""
        tool = self._tools.get(name)
        if tool:
            return tool["handler"]
        return None
    
    def list_tools(self) -> Dict[str, Dict]:
        """Retourne la liste des outils (compatible avec l'interface existante)"""
        return {name: {
            "name": info["name"],
            "description": info["description"],
            "inputSchema": info["inputSchema"]
        } for name, info in self._tools.items()}
    
    def call_tool(self, name: str, input_data: Any) -> Any:
        """
        Appelle un outil (compatible avec l'interface existante)
        Supporte à la fois string et dict en entrée
        """
        if name not in self._tools:
            raise ValueError(f"Unknown MCP tool: {name}")
        
        tool = self._tools[name]
        logger.info(f"[MCP] Calling {name}")
        
        # Compatibilité : si input_data est string, l'encapsuler dans dict
        if isinstance(input_data, str):
            # Déterminer le bon nom de paramètre
            if name == "calculator":
                args = {"expression": input_data}
            elif name == "python_executor":
                args = {"code": input_data}
            elif name == "wikipedia_search":
                args = {"query": input_data}
            elif name == "latex_formatter":
                args = {"expression": input_data}
            elif name == "citation_formatter":
                args = {"citation": input_data}
            else:
                args = {"input": input_data}
        else:
            args = input_data
        
        try:
            result = tool["handler"](args)
            return result
        except Exception as e:
            return f"Erreur MCP: {e}"
    
    def tool_manifest(self) -> Dict:
        """Retourne le manifeste MCP complet"""
        return {
            "tools": list(self._tools.values()),
            "version": "1.0",
            "protocol": "MCP/1.0",
        }
    
    # ─── Interface serveur stdio (pour mode indépendant) ───
    
    def handle_jsonrpc_request(self, request: Dict) -> Dict:
        """Traite une requête JSON-RPC conforme au protocole MCP"""
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params", {})
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "academic-mcp-server", "version": "1.0.0"}
                },
                "id": request_id
            }
        
        elif method == "tools/list":
            tools_list = [
                {
                    "name": name,
                    "description": info["description"],
                    "inputSchema": info["inputSchema"]
                }
                for name, info in self._tools.items()
            ]
            return {
                "jsonrpc": "2.0",
                "result": {"tools": tools_list},
                "id": request_id
            }
        
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            result = self.call_tool(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [{"type": "text", "text": result}]
                },
                "id": request_id
            }
        
        elif method == "ping":
            return {
                "jsonrpc": "2.0",
                "result": {"status": "ok"},
                "id": request_id
            }
        
        else:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Méthode inconnue: {method}"},
                "id": request_id
            }
    
    def run_stdio(self):
        """Lance le serveur en mode stdio (pour Claude Desktop, etc.)"""
        self._running = True
        logger.info("[MCP] Serveur démarré sur stdio - en attente de requêtes...")
        
        while self._running:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                request = json.loads(line)
                response = self.handle_jsonrpc_request(request)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                
            except json.JSONDecodeError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                    "id": None
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()
            except Exception as e:
                logger.error(f"Erreur serveur: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": str(e)},
                    "id": None
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()
        
        logger.info("[MCP] Serveur arrêté")


# ── Instance globale (compatible avec l'existant) ─────────────────────
mcp_server = MCPServer()


# ── Point d'entrée pour ligne de commande ─────────────────────────────
if __name__ == "__main__":
    """Lancement: python -m backend.mcp.server"""
    mcp_server.run_stdio()