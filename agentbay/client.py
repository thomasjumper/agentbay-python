"""AgentBay client - 3 lines to give your agent a brain."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests
from typing_extensions import TypeAlias

MemoryEntry: TypeAlias = Dict[str, Any]


class AgentBayError(Exception):
    """Base exception for AgentBay SDK errors."""

    def __init__(self, message: str, status_code: int | None = None, response: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class AuthenticationError(AgentBayError):
    """Raised when the API key is invalid or missing."""


class NotFoundError(AgentBayError):
    """Raised when a resource is not found."""


class RateLimitError(AgentBayError):
    """Raised when the API rate limit is exceeded."""


class AgentBay:
    """Persistent memory for AI agents.

    Usage::

        from agentbay import AgentBay

        brain = AgentBay("ab_live_your_key")
        brain.store("Next.js 16 + Prisma + PostgreSQL", title="Project stack")
        results = brain.recall("What stack does this project use?")

    Args:
        api_key: Your AgentBay API key (starts with ``ab_live_`` or ``ab_test_``).
        base_url: API base URL. Defaults to ``https://www.aiagentsbay.com``.
        project_id: Default project ID to use for all operations.
            Can be overridden per-call.
        timeout: Request timeout in seconds. Defaults to 30.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://www.aiagentsbay.com",
        project_id: str | None = None,
        timeout: int = 30,
    ) -> None:
        if not api_key:
            raise AuthenticationError("api_key is required. Get one at https://www.aiagentsbay.com/dashboard")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.project_id = project_id
        self.timeout = timeout

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "agentbay-python/0.1.0",
            }
        )

    # ------------------------------------------------------------------
    # Core memory operations
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        title: str | None = None,
        project_id: str | None = None,
        type: str = "PATTERN",
        tier: str = "semantic",
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        """Store a memory in your Knowledge Brain.

        Args:
            content: The knowledge content to store.
            title: Optional short title for the entry.
            project_id: Project to store in (overrides default).
            type: Entry type -- PATTERN, FACT, PREFERENCE, PROCEDURE, CONTEXT.
            tier: Storage tier -- semantic, episodic, procedural.
            tags: Optional list of tags for categorization.

        Returns:
            Dict with the created entry, including its ``id``.

        Raises:
            AgentBayError: If the request fails.
        """
        pid = self._resolve_project(project_id)
        body: Dict[str, Any] = {
            "content": content,
            "type": type,
            "tier": tier,
        }
        if title is not None:
            body["title"] = title
        if tags is not None:
            body["tags"] = tags

        return self._post(f"/api/v1/projects/{pid}/memory/store", body)

    def recall(
        self,
        query: str,
        project_id: str | None = None,
        limit: int = 5,
        tier: str | None = None,
        tags: list[str] | None = None,
    ) -> List[MemoryEntry]:
        """Search memories by semantic similarity.

        Args:
            query: Natural-language search query.
            project_id: Project to search in (overrides default).
            limit: Maximum number of results (1-50). Defaults to 5.
            tier: Filter by storage tier.
            tags: Filter by tags.

        Returns:
            List of matching entries with confidence scores.
        """
        pid = self._resolve_project(project_id)
        body: Dict[str, Any] = {
            "query": query,
            "limit": limit,
        }
        if tier is not None:
            body["tier"] = tier
        if tags is not None:
            body["tags"] = tags

        resp = self._post(f"/api/v1/projects/{pid}/memory/recall", body)
        # The API may return results under a "results" key or as a list directly.
        if isinstance(resp, list):
            return resp
        return resp.get("results", resp.get("entries", []))

    def forget(
        self,
        knowledge_id: str,
        project_id: str | None = None,
    ) -> None:
        """Archive (soft-delete) a memory entry.

        Args:
            knowledge_id: The ID of the memory to archive.
            project_id: Project containing the entry (overrides default).
        """
        pid = self._resolve_project(project_id)
        self._post(f"/api/v1/projects/{pid}/memory/forget", {"knowledgeId": knowledge_id})

    def verify(
        self,
        knowledge_id: str,
        project_id: str | None = None,
    ) -> None:
        """Confirm a memory is still accurate, resetting its confidence decay.

        Args:
            knowledge_id: The ID of the memory to verify.
            project_id: Project containing the entry (overrides default).
        """
        pid = self._resolve_project(project_id)
        self._post(f"/api/v1/projects/{pid}/memory/verify", {"knowledgeId": knowledge_id})

    def health(
        self,
        project_id: str | None = None,
    ) -> Dict[str, Any]:
        """Get memory health statistics for a project.

        Returns entry counts, average confidence, stale entries, etc.

        Args:
            project_id: Project to check (overrides default).

        Returns:
            Dict with health metrics.
        """
        pid = self._resolve_project(project_id)
        return self._get(f"/api/v1/projects/{pid}/memory/health")

    # ------------------------------------------------------------------
    # Brain management
    # ------------------------------------------------------------------

    def setup_brain(
        self,
        name: str,
        description: str | None = None,
    ) -> Dict[str, Any]:
        """Create a new private Knowledge Brain for your agent.

        This provisions a project with vector search, confidence decay,
        and all memory features enabled.

        Args:
            name: Human-readable name for the brain.
            description: Optional description.

        Returns:
            Dict with brain/project details including ``projectId``.
        """
        body: Dict[str, Any] = {"name": name}
        if description is not None:
            body["description"] = description

        resp = self._post("/api/v1/brain/setup", body)

        # Auto-set as default project if none was configured.
        project_id = resp.get("projectId") or resp.get("project", {}).get("id")
        if project_id and self.project_id is None:
            self.project_id = project_id

        return resp

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_project(self, project_id: str | None) -> str:
        pid = project_id or self.project_id
        if not pid:
            raise AgentBayError(
                "No project_id provided. Either pass project_id to this method, "
                "set it in the constructor, or call setup_brain() first."
            )
        return pid

    def _post(self, path: str, body: Dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.post(url, json=body, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise AgentBayError(f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise AgentBayError(f"Request timed out after {self.timeout}s") from exc
        return self._handle_response(resp)

    def _get(self, path: str, params: Dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise AgentBayError(f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise AgentBayError(f"Request timed out after {self.timeout}s") from exc
        return self._handle_response(resp)

    def _handle_response(self, resp: requests.Response) -> Any:
        if resp.status_code == 401:
            raise AuthenticationError(
                "Invalid API key. Check your key at https://www.aiagentsbay.com/dashboard",
                status_code=401,
            )
        if resp.status_code == 404:
            raise NotFoundError(
                f"Resource not found: {resp.url}",
                status_code=404,
            )
        if resp.status_code == 429:
            raise RateLimitError(
                "Rate limit exceeded. Please slow down or upgrade your plan.",
                status_code=429,
            )
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except ValueError:
                detail = {"text": resp.text}
            raise AgentBayError(
                f"API error {resp.status_code}: {detail}",
                status_code=resp.status_code,
                response=detail,
            )
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()

    def __repr__(self) -> str:
        masked = f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "***"
        return f"AgentBay(api_key='{masked}', project_id={self.project_id!r})"
