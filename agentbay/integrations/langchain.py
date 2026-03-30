"""LangChain integration for AgentBay memory.

Usage::

    from agentbay.integrations.langchain import AgentBayMemoryTool

    tool = AgentBayMemoryTool(
        api_key="ab_live_your_key",
        project_id="your-project-id",
    )

    # Use in a LangChain agent
    from langchain.agents import initialize_agent, AgentType
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI()
    agent = initialize_agent(
        tools=[tool],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
    )
    agent.run("Remember that the deploy key is stored in 1Password")
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Type

from agentbay.client import AgentBay

try:
    from langchain_core.callbacks import CallbackManagerForToolRun
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field

    class _MemoryInput(BaseModel):
        """Input schema for AgentBay memory tool."""

        action: str = Field(
            description='Action to perform: "store" or "recall".'
        )
        content: str = Field(
            description=(
                'For "store": the knowledge to save. '
                'For "recall": the search query.'
            )
        )
        title: str = Field(
            default="",
            description='Optional title for the memory (only used with "store").',
        )

    class AgentBayMemoryTool(BaseTool):
        """LangChain tool for storing and recalling agent memories via AgentBay.

        This tool lets LLM agents persist knowledge across sessions using
        AgentBay's Knowledge Brain -- semantic search, confidence decay,
        and cross-agent sharing included.

        Args:
            api_key: Your AgentBay API key.
            project_id: The Knowledge Brain project ID.
            base_url: API base URL (optional).
        """

        name: str = "agentbay_memory"
        description: str = (
            "Store or recall persistent memories. "
            'Use action="store" with content to save knowledge. '
            'Use action="recall" with content as the search query to retrieve memories.'
        )
        args_schema: Type[BaseModel] = _MemoryInput

        # Internal -- not part of the tool schema
        _client: AgentBay

        def __init__(
            self,
            api_key: str,
            project_id: str,
            base_url: str = "https://www.aiagentsbay.com",
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            object.__setattr__(
                self,
                "_client",
                AgentBay(
                    api_key=api_key,
                    base_url=base_url,
                    project_id=project_id,
                ),
            )

        def _run(
            self,
            action: str,
            content: str,
            title: str = "",
            run_manager: Optional[CallbackManagerForToolRun] = None,
        ) -> str:
            """Execute the memory tool."""
            if action == "store":
                result = self._client.store(
                    content=content,
                    title=title or None,
                )
                entry_id = result.get("id", result.get("entryId", "unknown"))
                return f"Stored memory (id: {entry_id})"

            elif action == "recall":
                results = self._client.recall(query=content)
                if not results:
                    return "No matching memories found."
                # Format results for the LLM
                formatted = []
                for i, entry in enumerate(results, 1):
                    title_str = entry.get("title", "Untitled")
                    body = entry.get("content", entry.get("text", ""))
                    score = entry.get("confidence", entry.get("score", "?"))
                    formatted.append(f"{i}. [{title_str}] (confidence: {score})\n   {body}")
                return "\n\n".join(formatted)

            else:
                return f'Unknown action "{action}". Use "store" or "recall".'

except ImportError:
    # LangChain not installed -- provide a helpful error at import time.
    class AgentBayMemoryTool:  # type: ignore[no-redef]
        """Placeholder when langchain-core is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "langchain-core is required for the LangChain integration. "
                "Install it with: pip install agentbay[langchain]"
            )
