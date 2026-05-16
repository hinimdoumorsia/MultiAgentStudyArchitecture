# backend/mcp/__init__.py
"""
MCPServer — Serveur MCP (Model Context Protocol)

Ce module expose un serveur MCP complet conforme au protocole standard.
Il permet d'intégrer des outils externes dans le ToolsAgent.

Usage :
    from backend.mcp import mcp_server
    mcp_server.register_tool("my_tool", my_function)
    tool_fn = mcp_server.get_tool("my_tool")
"""

import logging
from .server import MCPServer, mcp_server

__all__ = ["MCPServer", "mcp_server"]

logger = logging.getLogger(__name__)