"""CrewAI integration for AgentBay memory.

Usage::

    from agentbay.integrations.crewai import AgentBayCrewAIMemory

    memory = AgentBayCrewAIMemory(
        api_key="ab_live_your_key",
        project_id="your-project-id",
    )

    # Pass to your CrewAI agent
    from crewai import Agent
    agent = Agent(
        role="Researcher",
        goal="Find information",
        memory=memory,
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentbay.client import AgentBay


class AgentBayCrewAIMemory:
    """Drop-in memory backend for CrewAI agents.

    Stores and retrieves memories through AgentBay's Knowledge Brain,
    giving your CrewAI agents persistent, cross-session memory with
    semantic search and confidence decay.

    Args:
        api_key: Your AgentBay API key.
        project_id: The Knowledge Brain project ID to use.
        base_url: API base URL (optional).
    """

    def __init__(
        self,
        api_key: str,
        project_id: str,
        base_url: str = "https://www.aiagentsbay.com",
    ) -> None:
        self.client = AgentBay(
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
        )
        self.project_id = project_id

    def save(
        self,
        value: str,
        metadata: Dict[str, Any] | None = None,
        agent: str | None = None,
    ) -> Dict[str, Any]:
        """Save a memory entry (CrewAI memory interface).

        Args:
            value: The content to remember.
            metadata: Optional metadata dict. Recognized keys:
                ``title``, ``type``, ``tier``, ``tags``.
            agent: Optional agent name (stored as a tag).

        Returns:
            Dict with the created entry.
        """
        metadata = metadata or {}
        tags = metadata.get("tags", [])
        if agent:
            tags = list(tags) + [f"agent:{agent}"]

        return self.client.store(
            content=value,
            title=metadata.get("title"),
            type=metadata.get("type", "PATTERN"),
            tier=metadata.get("tier", "semantic"),
            tags=tags or None,
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search memories by semantic similarity (CrewAI memory interface).

        Args:
            query: Natural-language search query.
            limit: Maximum number of results.
            score_threshold: Minimum confidence score (0.0-1.0).

        Returns:
            List of matching entries.
        """
        results = self.client.recall(query=query, limit=limit)
        if score_threshold > 0:
            results = [
                r for r in results
                if r.get("confidence", r.get("score", 0)) >= score_threshold
            ]
        return results

    def reset(self) -> None:
        """Reset is not supported -- memories are persistent by design.

        This is a no-op to satisfy the CrewAI memory interface.
        Use ``client.forget()`` to archive individual entries.
        """
        pass

    def __repr__(self) -> str:
        return f"AgentBayCrewAIMemory(project_id={self.project_id!r})"
