"""Local memory store using SQLite -- works offline with no API key."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _auto_title(text: str, max_len: int = 100) -> str:
    """Extract a title from the first sentence or first N chars."""
    match = re.match(r"^(.+?[.!?])\s", text)
    if match and len(match.group(1)) <= max_len:
        return match.group(1)
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


def _auto_type(text: str) -> str:
    """Auto-detect memory type from content."""
    lower = text.lower()
    if re.search(r"\b(?:bug|error|fix|crash|fail|broke|issue|problem|exception)\b", lower):
        return "PITFALL"
    if re.search(r"\b(?:decided|chose|picked|went with|settled on|decision)\b", lower):
        return "DECISION"
    if re.search(r"\b(?:step\s*\d|first.*then|how to|procedure|process|workflow)\b", lower):
        return "PROCEDURE"
    return "PATTERN"


class LocalMemory:
    """SQLite-backed local memory store.

    Usage::

        from agentbay.local import LocalMemory
        mem = LocalMemory()  # stores in ~/.agentbay/local.db
        mem.store("Always use connection pooling", title="DB pattern")
        results = mem.recall("database connection")
    """

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_dir = Path.home() / ".agentbay"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "local.db")
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL DEFAULT 'PATTERN',
                    tier TEXT NOT NULL DEFAULT 'semantic',
                    tags TEXT NOT NULL DEFAULT '[]',
                    user_id TEXT,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    helpful_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)
            """)

    def store(
        self,
        content: str,
        title: str | None = None,
        type: str = "PATTERN",
        tier: str = "semantic",
        tags: list[str] | None = None,
        user_id: str | None = None,
        confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """Store a memory locally.

        Args:
            content: The knowledge content to store.
            title: Optional short title (auto-generated from first sentence if not provided).
            type: Entry type -- PATTERN, FACT, PREFERENCE, PROCEDURE, CONTEXT, PITFALL, DECISION.
            tier: Storage tier -- semantic, episodic, procedural.
            tags: Optional list of tags.
            user_id: Optional user ID for scoping.
            confidence: Initial confidence score (0.0-1.0). Defaults to 0.5.

        Returns:
            Dict with ``id`` and ``deduplicated`` keys.
        """
        if title is None:
            title = _auto_title(content)

        tags_json = json.dumps(tags or [])
        now = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Dedup check: same title + type + user_id
            row = conn.execute(
                "SELECT id FROM memories WHERE title = ? AND type = ? AND (user_id = ? OR (user_id IS NULL AND ? IS NULL))",
                (title, type, user_id, user_id),
            ).fetchone()

            if row:
                # Update existing
                conn.execute(
                    "UPDATE memories SET content = ?, tags = ?, confidence = ?, updated_at = ? WHERE id = ?",
                    (content, tags_json, confidence, now, row[0]),
                )
                return {"id": row[0], "deduplicated": True}

            entry_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO memories (id, title, content, type, tier, tags, user_id, confidence, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (entry_id, title, content, type, tier, tags_json, user_id, confidence, now, now),
            )
            return {"id": entry_id, "deduplicated": False}

    def recall(
        self,
        query: str,
        limit: int = 5,
        user_id: str | None = None,
        type: str | None = None,
        tags: list[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Search memories using keyword matching.

        Splits the query into words and scores each entry by how many
        words appear in the title (2x boost) and content. Returns the
        top N results sorted by score.

        Args:
            query: Natural-language search query.
            limit: Maximum number of results. Defaults to 5.
            user_id: Filter to a specific user's memories.
            type: Filter by entry type.
            tags: Filter by tags (entry must contain all listed tags).

        Returns:
            List of dicts with ``id``, ``title``, ``content``, ``type``,
            ``tier``, ``tags``, ``confidence``, and ``score``.
        """
        # Build WHERE clauses
        conditions: list[str] = []
        params: list[Any] = []

        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)
        if type is not None:
            conditions.append("type = ?")
            params.append(type)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM memories {where}",
                params,
            ).fetchall()

        # Keyword scoring
        words = [w.lower() for w in re.split(r"\W+", query) if len(w) >= 2]
        if not words:
            return []

        # Tag filter (post-query since tags are JSON)
        filter_tags = set(tags or [])

        scored: list[tuple[float, dict]] = []
        for row in rows:
            row_dict = dict(row)
            entry_tags = json.loads(row_dict["tags"])

            # Tag filter
            if filter_tags and not filter_tags.issubset(set(entry_tags)):
                continue

            title_lower = row_dict["title"].lower()
            content_lower = row_dict["content"].lower()
            score = 0.0
            for w in words:
                if w in title_lower:
                    score += 2.0
                if w in content_lower:
                    score += 1.0

            if score > 0:
                # Bump access count
                scored.append((score, {
                    "id": row_dict["id"],
                    "title": row_dict["title"],
                    "content": row_dict["content"],
                    "type": row_dict["type"],
                    "tier": row_dict["tier"],
                    "tags": entry_tags,
                    "confidence": row_dict["confidence"],
                    "score": score,
                }))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item[1] for item in scored[:limit]]

        # Bump access counts for returned results
        if results:
            with sqlite3.connect(self.db_path) as conn:
                for r in results:
                    conn.execute(
                        "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                        (r["id"],),
                    )

        return results

    def add(
        self,
        data: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict | None = None,
    ) -> Dict[str, Any]:
        """Mem0-compatible store. Auto-detects title and type.

        Args:
            data: The knowledge content to store.
            user_id: Optional user ID for scoping.
            agent_id: Ignored (kept for API compatibility).
            metadata: Ignored (kept for API compatibility).

        Returns:
            Dict with ``id`` and ``deduplicated`` keys.
        """
        return self.store(
            content=data,
            title=_auto_title(data),
            type=_auto_type(data),
            tier="semantic",
            user_id=user_id,
        )

    def search(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Mem0-compatible alias for recall.

        Args:
            query: Natural-language search query.
            user_id: Optional user ID for scoping.
            limit: Maximum results. Defaults to 5.

        Returns:
            List of matching entries with scores.
        """
        return self.recall(query, limit=limit, user_id=user_id)

    def forget(self, memory_id: str) -> None:
        """Delete a memory by ID.

        Args:
            memory_id: The ID of the memory to delete.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    def health(self) -> Dict[str, Any]:
        """Return memory stats.

        Returns:
            Dict with ``total_entries``, ``by_tier``, ``by_type``, and
            ``total_tokens`` (approximate).
        """
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

            by_tier: Dict[str, int] = {}
            for row in conn.execute("SELECT tier, COUNT(*) FROM memories GROUP BY tier"):
                by_tier[row[0]] = row[1]

            by_type: Dict[str, int] = {}
            for row in conn.execute("SELECT type, COUNT(*) FROM memories GROUP BY type"):
                by_type[row[0]] = row[1]

            # Rough token estimate: ~4 chars per token
            total_chars = conn.execute(
                "SELECT COALESCE(SUM(LENGTH(content) + LENGTH(title)), 0) FROM memories"
            ).fetchone()[0]
            total_tokens = total_chars // 4

        return {
            "total_entries": total,
            "by_tier": by_tier,
            "by_type": by_type,
            "total_tokens": total_tokens,
        }

    def export(self) -> List[Dict[str, Any]]:
        """Export all memories as a list of dicts.

        Returns:
            List of all memory entries.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM memories ORDER BY created_at").fetchall()

        return [
            {
                "id": r["id"],
                "title": r["title"],
                "content": r["content"],
                "type": r["type"],
                "tier": r["tier"],
                "tags": json.loads(r["tags"]),
                "user_id": r["user_id"],
                "confidence": r["confidence"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "access_count": r["access_count"],
                "helpful_count": r["helpful_count"],
            }
            for r in rows
        ]

    def upgrade(self, api_key: str, project_id: str | None = None) -> Any:
        """Migrate local memories to cloud AgentBay.

        Reads all local memories and stores each one via the cloud API.
        Returns a fully initialized cloud AgentBay client.

        Args:
            api_key: Your AgentBay API key.
            project_id: Cloud project ID to migrate into.

        Returns:
            A cloud :class:`AgentBay` instance with all local memories migrated.
        """
        from .client import AgentBay

        cloud = AgentBay(api_key=api_key, project_id=project_id)
        entries = self.export()

        for entry in entries:
            tags = entry.get("tags", [])
            if entry.get("user_id"):
                tags = list(tags) + [f"user:{entry['user_id']}"]
            cloud.store(
                content=entry["content"],
                title=entry["title"],
                type=entry["type"],
                tier=entry["tier"],
                tags=tags if tags else None,
            )

        return cloud

    def __repr__(self) -> str:
        stats = self.health()
        return f"LocalMemory(db='{self.db_path}', entries={stats['total_entries']})"
