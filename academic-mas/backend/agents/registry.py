"""
AgentRegistry — central registry for all agents in the system.

This is what makes the architecture SCALABLE:
- Add an agent: registry.register(MyAgent())
- Remove an agent: registry.unregister("my_agent")
- List active agents: registry.list_agents()

The orchestrator and router always query the registry at runtime,
so no other file needs to change when you add/remove agents.
"""

from typing import Dict, List, Optional
import logging

from backend.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents: Dict[str, BaseAgent] = {}
        return cls._instance

    def register(self, agent: BaseAgent) -> None:
        """Register an agent. Overwrites if same name."""
        self._agents[agent.name] = agent
        logger.info(f"[Registry] Registered agent: {agent.name}")

    def unregister(self, name: str) -> None:
        """Remove an agent by name."""
        if name in self._agents:
            del self._agents[name]
            logger.info(f"[Registry] Unregistered agent: {name}")

    def get(self, name: str) -> Optional[BaseAgent]:
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())

    def all_agents(self) -> Dict[str, BaseAgent]:
        return dict(self._agents)

    def agent_descriptions(self) -> Dict[str, str]:
        return {name: agent.description for name, agent in self._agents.items()}


# Global singleton
registry = AgentRegistry()
