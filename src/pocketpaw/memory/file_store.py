# File-based memory store implementation.
# Created: 2026-02-02 - Memory System
# Updated: 2026-02-09 - Fixed UUID collision, daily file loading, search, persistent delete
# Updated: 2026-02-10 - Session index for fast listing, delete/rename support
#
# Stores memories as markdown files for human readability:
# - ~/.pocketpaw/memory/MEMORY.md     (long-term)
# - ~/.pocketpaw/memory/2026-02-02.md (daily)
# - ~/.pocketpaw/memory/sessions/     (session JSON files)
# - ~/.pocketpaw/memory/sessions/_index.json (session metadata index)

import asyncio
import hashlib
import json
import logging
import math
import re
import sqlite3
import uuid
from datetime import UTC, date, datetime
from pathlib import Path

from pocketpaw.memory.protocol import MemoryEntry, MemoryType

logger = logging.getLogger(__name__)


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware (UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


# Stop words excluded from word-overlap search scoring
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "like",
        "through",
        "after",
        "over",
        "between",
        "out",
        "against",
        "during",
        "without",
        "before",
        "under",
        "around",
        "among",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "because",
        "if",
        "when",
        "where",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "it",
        "its",
        "they",
        "them",
        "their",
    }
)


def _make_deterministic_id(path: Path, header: str, body: str) -> str:
    """Generate a deterministic UUID5 from path, header, AND body content."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{path}:{header}:{body}"))


def _tokenize(text: str) -> set[str]:
    """Lowercase, split on non-alpha, strip stop words."""
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return words - _STOP_WORDS


class FileMemoryStore:
    """
    File-based memory store.

    Human-readable markdown for long-term and daily memories.
    JSON for session memories (machine-readable).
    """

    def __init__(
        self,
        base_path: Path | None = None,
        *,
        vector_enabled: bool = True,
        vector_store: str = "sqlite-vec",
        embedding_model: str = "nomic-embed-text",
        embedding_provider: str = "ollama",
        embedding_base_url: str = "http://localhost:11434",
    ):
        self.base_path = base_path or (Path.home() / ".pocketpaw" / "memory")
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Sub-directories
        self.sessions_path = self.base_path / "sessions"
        self.sessions_path.mkdir(exist_ok=True)

        # File paths
        self.long_term_file = self.base_path / "MEMORY.md"

        # In-memory index for fast lookup
        self._index: dict[str, MemoryEntry] = {}
        self._session_write_locks: dict[str, asyncio.Lock] = {}
        self._session_index_lock = asyncio.Lock()  # Protects _index.json read-modify-write
        self._alias_lock = asyncio.Lock()  # Protects _aliases.json read-modify-write

        # Vector-backed semantic memory (phase 1)
        self._vector_enabled = vector_enabled
        self._vector_store = vector_store.strip().lower()
        self._embedding_model = embedding_model
        self._embedding_provider = embedding_provider.strip().lower()
        self._embedding_base_url = embedding_base_url
        self._vector_lock = asyncio.Lock()
        self._vector_db_path = self.base_path / "vector_index.sqlite3"
        self._vector_backend = "none"
        self._chroma_collection = None

        # Inverted index for O(k) search narrowing (word -> set of entry IDs)
        self._inverted: dict[str, set[str]] = {}
        self._inv_dirty = True

        self._initialize_vector_backend()

        self._load_index()

        # Build session index on first run (migration)
        if not self._index_path.exists():
            self.rebuild_session_index()

    def _initialize_vector_backend(self) -> None:
        """Initialize vector backend for semantic retrieval.

        Keeps markdown storage as source of truth and augments it with vector search.
        """
        if not self._vector_enabled:
            self._vector_backend = "disabled"
            return

        if self._vector_store == "chromadb":
            try:
                import importlib

                chromadb = importlib.import_module("chromadb")

                chroma_path = self.base_path / "chroma_db"
                chroma_path.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(chroma_path))
                self._chroma_collection = client.get_or_create_collection(
                    name="pocketpaw_file_memory"
                )
                self._vector_backend = "chromadb"
                return
            except ImportError:
                logger.warning(
                    "vector_store=chromadb requested but chromadb is not installed; "
                    "falling back to sqlite-vec style local index"
                )

        if self._vector_store == "qdrant":
            logger.warning(
                "vector_store=qdrant requested for file backend; "
                "using sqlite-vec style local index for now"
            )

        self._initialize_sqlite_vector_store()
        self._vector_backend = "sqlite-vec"

    def _initialize_sqlite_vector_store(self) -> None:
        """Initialize local sqlite vector table (embeddings stored as JSON)."""
        with sqlite3.connect(self._vector_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    doc_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    user_scope TEXT NOT NULL,
                    session_key TEXT,
                    role TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_vectors_user_scope "
                "ON memory_vectors(user_scope)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_vectors_type "
                "ON memory_vectors(memory_type)"
            )

    @staticmethod
    def _hash_embedding(text: str, dim: int = 384) -> list[float]:
        """Deterministic local embedding fallback (no network/deps)."""
        tokens = _tokenize(text)
        if not tokens:
            return [0.0] * dim

        vector = [0.0] * dim
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + (digest[5] / 255.0)
            vector[index] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1e-12:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        """Cosine similarity for equal-length vectors."""
        if len(left) != len(right) or not left:
            return 0.0
        return sum(a * b for a, b in zip(left, right, strict=False))

    async def _embed_text(self, text: str) -> list[float]:
        """Embed text using configured provider, falling back to local hash embedding."""
        if self._embedding_provider == "ollama":
            try:
                import ollama

                client = ollama.AsyncClient(host=self._embedding_base_url)
                result = await client.embeddings(model=self._embedding_model, prompt=text)
                embedding = result.get("embedding") if isinstance(result, dict) else None
                if isinstance(embedding, list) and embedding:
                    return [float(value) for value in embedding]
            except Exception:
                logger.debug("Ollama embedding failed, using local hash embedding", exc_info=True)

        return self._hash_embedding(text)

    def _entry_user_scope(self, entry: MemoryEntry) -> str:
        """Resolve memory ownership scope for semantic retrieval."""
        metadata = entry.metadata or {}
        if entry.type == MemoryType.LONG_TERM:
            return str(metadata.get("user_id", "default"))
        if entry.type == MemoryType.SESSION:
            return str(metadata.get("sender_id", metadata.get("user_id", "default")))
        return "default"

    async def _upsert_vector_record(self, entry: MemoryEntry) -> None:
        """Insert or update vector record for a memory entry."""
        if not self._vector_enabled:
            return

        user_scope = self._entry_user_scope(entry)
        metadata_json = json.dumps(entry.metadata or {}, ensure_ascii=False)
        created_at_iso = _ensure_utc(entry.created_at).isoformat()

        if self._vector_backend == "chromadb" and self._chroma_collection is not None:
            chroma_meta = {
                "memory_type": entry.type.value,
                "user_scope": user_scope,
                "session_key": entry.session_key or "",
                "role": entry.role or "",
                "created_at": created_at_iso,
                "metadata_json": metadata_json,
            }

            await asyncio.to_thread(
                self._chroma_collection.upsert,
                ids=[entry.id],
                documents=[entry.content],
                metadatas=[chroma_meta],
            )
            return

        embedding = await self._embed_text(entry.content)

        async with self._vector_lock:
            await asyncio.to_thread(
                self._upsert_sqlite_vector_record,
                entry,
                user_scope,
                created_at_iso,
                metadata_json,
                embedding,
            )

    def _upsert_sqlite_vector_record(
        self,
        entry: MemoryEntry,
        user_scope: str,
        created_at_iso: str,
        metadata_json: str,
        embedding: list[float],
    ) -> None:
        """Upsert vector record into local sqlite vector table."""
        with sqlite3.connect(self._vector_db_path) as conn:
            conn.execute(
                """
                INSERT INTO memory_vectors (
                    doc_id, content, memory_type, user_scope, session_key,
                    role, created_at, metadata_json, embedding_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    content = excluded.content,
                    memory_type = excluded.memory_type,
                    user_scope = excluded.user_scope,
                    session_key = excluded.session_key,
                    role = excluded.role,
                    created_at = excluded.created_at,
                    metadata_json = excluded.metadata_json,
                    embedding_json = excluded.embedding_json
                """,
                (
                    entry.id,
                    entry.content,
                    entry.type.value,
                    user_scope,
                    entry.session_key,
                    entry.role,
                    created_at_iso,
                    metadata_json,
                    json.dumps(embedding),
                ),
            )

    async def _delete_vector_record(self, entry_id: str) -> None:
        """Delete a vector record by entry ID."""
        if not self._vector_enabled:
            return

        if self._vector_backend == "chromadb" and self._chroma_collection is not None:
            await asyncio.to_thread(self._chroma_collection.delete, ids=[entry_id])
            return

        async with self._vector_lock:
            await asyncio.to_thread(self._delete_sqlite_vector_record, entry_id)

    def _delete_sqlite_vector_record(self, entry_id: str) -> None:
        """Delete a vector record from sqlite store."""
        with sqlite3.connect(self._vector_db_path) as conn:
            conn.execute("DELETE FROM memory_vectors WHERE doc_id = ?", (entry_id,))

    async def semantic_search(
        self,
        query: str,
        user_id: str = "default",
        limit: int = 5,
    ) -> list[dict]:
        """Semantic search for relevant memories.

        Returns list of dicts compatible with MemoryManager.get_semantic_context().
        """
        if not query.strip():
            return []

        if self._vector_enabled:
            try:
                await self._backfill_missing_vector_records()
                if self._vector_backend == "chromadb" and self._chroma_collection is not None:
                    return await self._semantic_search_chromadb(query, user_id=user_id, limit=limit)
                return await self._semantic_search_sqlite(query, user_id=user_id, limit=limit)
            except Exception:
                logger.debug("Vector semantic search failed; using lexical fallback", exc_info=True)

        lexical = await self.search(query=query, limit=limit)
        return [
            {
                "id": entry.id,
                "memory": entry.content,
                "score": 0.0,
                "memory_type": entry.type.value,
            }
            for entry in lexical
        ]

    async def _backfill_missing_vector_records(self, max_items: int = 250) -> None:
        """Lazily vectorize historical in-memory entries not present in vector store."""
        if not self._vector_enabled:
            return

        # Chroma backend can upsert duplicates safely and handles id conflicts internally.
        if self._vector_backend == "chromadb" and self._chroma_collection is not None:
            pending = list(self._index.values())[:max_items]
            for entry in pending:
                await self._upsert_vector_record(entry)
            return

        def _get_existing_ids() -> set[str]:
            with sqlite3.connect(self._vector_db_path) as conn:
                rows = conn.execute("SELECT doc_id FROM memory_vectors").fetchall()
            return {str(row[0]) for row in rows}

        existing_ids = await asyncio.to_thread(_get_existing_ids)
        pending: list[MemoryEntry] = []
        for entry_id, entry in self._index.items():
            if entry_id not in existing_ids:
                pending.append(entry)
            if len(pending) >= max_items:
                break

        for entry in pending:
            await self._upsert_vector_record(entry)

    async def _semantic_search_chromadb(self, query: str, user_id: str, limit: int) -> list[dict]:
        """Semantic search using chromadb backend."""
        where = {"user_scope": user_id}
        result = await asyncio.to_thread(
            self._chroma_collection.query,
            query_texts=[query],
            n_results=limit,
            where=where,
        )

        ids = result.get("ids", [[]])[0] if result else []
        docs = result.get("documents", [[]])[0] if result else []
        metas = result.get("metadatas", [[]])[0] if result else []
        distances = result.get("distances", [[]])[0] if result else []

        rows: list[dict] = []
        for index, doc_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 1.0
            score = max(0.0, 1.0 - distance)
            meta = metas[index] if index < len(metas) and isinstance(metas[index], dict) else {}
            memory_text = docs[index] if index < len(docs) else ""
            rows.append(
                {
                    "id": doc_id,
                    "memory": memory_text,
                    "score": score,
                    "memory_type": meta.get("memory_type", "unknown"),
                }
            )

        return rows

    async def _semantic_search_sqlite(self, query: str, user_id: str, limit: int) -> list[dict]:
        """Semantic search using local sqlite vector table."""
        query_embedding = await self._embed_text(query)

        def _search_sync() -> list[dict]:
            with sqlite3.connect(self._vector_db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT doc_id, content, memory_type, embedding_json
                    FROM memory_vectors
                    WHERE user_scope = ?
                    """,
                    (user_id,),
                )
                rows = cursor.fetchall()

            scored: list[tuple[float, dict]] = []
            for doc_id, content, memory_type, embedding_json in rows:
                try:
                    embedding = json.loads(embedding_json)
                    if not isinstance(embedding, list):
                        continue
                    score = self._cosine_similarity(query_embedding, [float(v) for v in embedding])
                    scored.append(
                        (
                            score,
                            {
                                "id": doc_id,
                                "memory": content,
                                "score": score,
                                "memory_type": memory_type,
                            },
                        )
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue

            scored.sort(key=lambda item: item[0], reverse=True)
            return [payload for _, payload in scored[:limit]]

        return await asyncio.to_thread(_search_sync)

    # =========================================================================
    # Session Index
    # =========================================================================

    @property
    def _index_path(self) -> Path:
        """Path to the session index file."""
        return self.sessions_path / "_index.json"

    def _load_session_index(self) -> dict:
        """Read session index from disk. Returns empty dict if missing/corrupt."""
        if not self._index_path.exists():
            return {}
        try:
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_session_index(self, index: dict) -> None:
        """Atomic write of session index (write to .tmp then rename)."""
        tmp = self._index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, indent=2), encoding="utf-8")
        tmp.replace(self._index_path)

    # =========================================================================
    # Session Aliases
    # =========================================================================

    @property
    def _aliases_path(self) -> Path:
        """Path to the session aliases file."""
        return self.sessions_path / "_aliases.json"

    def _load_aliases(self) -> dict[str, str]:
        """Read session aliases from disk. Returns empty dict if missing/corrupt."""
        if not self._aliases_path.exists():
            return {}
        try:
            return json.loads(self._aliases_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_aliases(self, aliases: dict[str, str]) -> None:
        """Atomic write of aliases file (write to .tmp then rename)."""
        tmp = self._aliases_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(aliases, indent=2), encoding="utf-8")
        tmp.replace(self._aliases_path)

    async def resolve_session_alias(self, session_key: str) -> str:
        """Resolve a session key through the alias table.

        Returns the aliased target key if one exists, otherwise the original key.
        """
        async with self._alias_lock:
            aliases = self._load_aliases()
        return aliases.get(session_key, session_key)

    async def set_session_alias(self, source_key: str, target_key: str) -> None:
        """Set or overwrite a session alias (source_key -> target_key)."""
        async with self._alias_lock:
            aliases = self._load_aliases()
            aliases[source_key] = target_key
            self._save_aliases(aliases)

    async def remove_session_alias(self, source_key: str) -> bool:
        """Remove a session alias. Returns True if it existed."""
        async with self._alias_lock:
            aliases = self._load_aliases()
            if source_key not in aliases:
                return False
            del aliases[source_key]
            self._save_aliases(aliases)
            return True

    async def get_session_keys_for_chat(self, source_key: str) -> list[str]:
        """Return all session keys associated with this source key.

        Includes the current alias target (if any) plus all historical
        target keys where source matches.
        """
        async with self._alias_lock:
            aliases = self._load_aliases()

        keys: list[str] = []
        for src, tgt in aliases.items():
            if src == source_key:
                keys.append(tgt)

        # Also include the source_key itself (the default/unaliased session)
        safe_default = source_key.replace(":", "_").replace("/", "_")
        default_file = self.sessions_path / f"{safe_default}.json"
        if default_file.exists() and source_key not in keys:
            keys.append(source_key)

        return keys

    async def _update_session_index(
        self, session_key: str, entry: MemoryEntry, session_data: list[dict]
    ) -> None:
        """Update a single entry in the session index after a message save."""
        async with self._session_index_lock:
            index = self._load_session_index()
            safe_key = session_key.replace(":", "_").replace("/", "_")

            # Extract channel from session_key (format: "channel:uuid")
            parts = session_key.split(":", 1)
            channel = parts[0] if len(parts) > 1 else "unknown"

            # Find first user message for title
            title = ""
            for msg in session_data:
                if msg.get("role") == "user" and msg.get("content", "").strip():
                    title = msg["content"].strip()[:80]
                    break
            if not title:
                title = "New Chat"

            # Last message preview
            last_msg = session_data[-1] if session_data else {}
            preview = last_msg.get("content", "")[:120]

            # Timestamps
            first_msg = session_data[0] if session_data else {}
            created = first_msg.get("timestamp", datetime.now(tz=UTC).isoformat())
            last_activity = last_msg.get("timestamp", datetime.now(tz=UTC).isoformat())

            # Preserve existing title if user renamed it
            existing = index.get(safe_key, {})
            if existing.get("user_title"):
                title = existing["user_title"]

            index[safe_key] = {
                "title": title,
                "channel": channel,
                "created": existing.get("created", created),
                "last_activity": last_activity,
                "message_count": len(session_data),
                "preview": preview,
            }
            # Preserve user_title flag if set
            if existing.get("user_title"):
                index[safe_key]["user_title"] = existing["user_title"]

            self._save_session_index(index)

    def rebuild_session_index(self) -> dict:
        """Full directory scan to build index from all session files."""
        index: dict = {}
        for session_file in self.sessions_path.glob("*.json"):
            if session_file.name.startswith("_") or session_file.name.endswith("_compaction.json"):
                continue

            safe_key = session_file.stem
            try:
                data = json.loads(session_file.read_text(encoding="utf-8"))
                if not data or not isinstance(data, list):
                    continue

                # Derive channel from safe_key (format: "channel_uuid")
                parts = safe_key.split("_", 1)
                channel = parts[0] if len(parts) > 1 else "unknown"

                # First user message as title
                title = "New Chat"
                for msg in data:
                    if msg.get("role") == "user" and msg.get("content", "").strip():
                        title = msg["content"].strip()[:80]
                        break

                first_msg = data[0]
                last_msg = data[-1]

                index[safe_key] = {
                    "title": title,
                    "channel": channel,
                    "created": first_msg.get("timestamp", ""),
                    "last_activity": last_msg.get("timestamp", ""),
                    "message_count": len(data),
                    "preview": last_msg.get("content", "")[:120],
                }
            except (json.JSONDecodeError, KeyError, OSError):
                continue

        self._save_session_index(index)
        return index

    async def delete_session(self, session_key: str) -> bool:
        """Delete a session file, compaction cache, and index entry."""
        safe_key = session_key.replace(":", "_").replace("/", "_")
        session_file = self.sessions_path / f"{safe_key}.json"
        compaction_file = self.sessions_path / f"{safe_key}_compaction.json"

        if not session_file.exists():
            return False

        entry_ids: list[str] = []
        try:
            raw = await asyncio.to_thread(lambda: session_file.read_text(encoding="utf-8"))
            data = json.loads(raw)
            if isinstance(data, list):
                entry_ids = [str(item.get("id", "")) for item in data if item.get("id")]
        except (OSError, json.JSONDecodeError):
            entry_ids = []

        session_file.unlink()
        if compaction_file.exists():
            compaction_file.unlink()

        for entry_id in entry_ids:
            await self._delete_vector_record(entry_id)
            self._index.pop(entry_id, None)

        # Remove from index (protected by lock to prevent lost updates)
        async with self._session_index_lock:
            index = self._load_session_index()
            index.pop(safe_key, None)
            self._save_session_index(index)

        # Clean up write lock
        self._session_write_locks.pop(session_key, None)

        return True

    async def update_session_title(self, session_key: str, title: str) -> bool:
        """Update the title of a session in the index."""
        safe_key = session_key.replace(":", "_").replace("/", "_")
        async with self._session_index_lock:
            index = self._load_session_index()
            if safe_key not in index:
                return False
            index[safe_key]["title"] = title
            index[safe_key]["user_title"] = title  # Mark as user-renamed
            self._save_session_index(index)
        return True

    async def search_sessions(self, query: str, limit: int = 20) -> list[dict]:
        """Search session files for messages matching *query*.

        All blocking I/O (glob, read_text, json.loads) runs inside
        ``asyncio.to_thread`` so the event loop is never blocked.
        """
        if not query or not query.strip():
            return []

        query_lower = query.lower()
        sessions_path = self.sessions_path
        index_path = self._index_path

        def _search_sync() -> list[dict]:
            # Load index inside the thread so its file I/O doesn't block
            # the event loop either.
            try:
                index_snapshot = json.loads(index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, FileNotFoundError):
                index_snapshot = {}
            results: list[dict] = []
            for session_file in sessions_path.glob("*.json"):
                if session_file.name.startswith("_") or session_file.name.endswith(
                    "_compaction.json"
                ):
                    continue
                try:
                    data = json.loads(session_file.read_text(encoding="utf-8"))
                    for msg in data:
                        if query_lower in msg.get("content", "").lower():
                            safe_key = session_file.stem
                            meta = index_snapshot.get(safe_key, {})
                            results.append(
                                {
                                    "id": safe_key,
                                    "title": meta.get("title", "Untitled"),
                                    "channel": meta.get("channel", "unknown"),
                                    "match": msg["content"][:200],
                                    "match_role": msg.get("role", ""),
                                    "last_activity": meta.get("last_activity", ""),
                                }
                            )
                            break
                except (json.JSONDecodeError, OSError):
                    continue
                if len(results) >= limit:
                    break
            return results

        return await asyncio.to_thread(_search_sync)

    def _load_index(self) -> None:
        """Load existing memories into index."""
        # Load long-term memories (root = owner/default)
        if self.long_term_file.exists():
            self._parse_markdown_file(self.long_term_file, MemoryType.LONG_TERM)

        # Load per-user long-term memories
        users_dir = self.base_path / "users"
        if users_dir.exists():
            for user_mem in users_dir.glob("*/MEMORY.md"):
                self._parse_markdown_file(user_mem, MemoryType.LONG_TERM)

        # Load ALL daily files (not just today's)
        for daily_file in sorted(
            self.base_path.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].md")
        ):
            self._parse_markdown_file(daily_file, MemoryType.DAILY)

        self._inv_dirty = True

    def _rebuild_inverted(self) -> None:
        """Build/rebuild the inverted index from _index. Resets _inv_dirty."""
        inv: dict[str, set[str]] = {}
        for eid, entry in self._index.items():
            words = _tokenize(entry.content)
            header = entry.metadata.get("header", "")
            if header:
                words |= _tokenize(header)
            for w in words:
                inv.setdefault(w, set()).add(eid)
        self._inverted = inv
        self._inv_dirty = False

    def _parse_markdown_file(self, path: Path, memory_type: MemoryType) -> None:
        """Parse a markdown file into memory entries."""
        content = path.read_text(encoding="utf-8")

        # Derive user_id from path for per-user memory files
        user_id = "default"
        users_dir = self.base_path / "users"
        try:
            if path.is_relative_to(users_dir):
                # e.g. .../users/abc123/MEMORY.md → user_id = "abc123"
                user_id = path.parent.name
        except (TypeError, ValueError):
            pass

        # Split by headers (## or ###)
        sections = re.split(r"\n(?=##+ )", content)

        for section in sections:
            if not section.strip():
                continue

            # Extract header and content
            lines = section.strip().split("\n")
            header = lines[0].lstrip("#").strip()
            body = "\n".join(lines[1:]).strip()

            if body:
                entry_id = _make_deterministic_id(path, header, body)
                metadata = {"header": header, "source": str(path)}
                if user_id != "default":
                    metadata["user_id"] = user_id
                self._index[entry_id] = MemoryEntry(
                    id=entry_id,
                    type=memory_type,
                    content=body,
                    tags=self._extract_tags(body),
                    metadata=metadata,
                )

    def _extract_tags(self, content: str) -> list[str]:
        """Extract #tags from content."""
        return re.findall(r"#(\w+)", content)

    def _get_user_memory_file(self, user_id: str = "default") -> Path:
        """Get the MEMORY.md path for a given user.

        - "default" → root MEMORY.md (owner / single-user)
        - Others → users/{user_id}/MEMORY.md (auto-create dir)
        """
        if user_id == "default":
            return self.long_term_file
        user_dir = self.base_path / "users" / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / "MEMORY.md"

    def _get_daily_file(self, d: date) -> Path:
        """Get the path for a daily notes file."""
        return self.base_path / f"{d.isoformat()}.md"

    def _get_session_file(self, session_key: str) -> Path:
        """Get the path for a session file."""
        safe_key = session_key.replace(":", "_").replace("/", "_")
        return self.sessions_path / f"{safe_key}.json"

    # =========================================================================
    # MemoryStoreProtocol Implementation
    # =========================================================================

    async def save(self, entry: MemoryEntry) -> str:
        """Save a memory entry."""
        if entry.type == MemoryType.SESSION:
            # Session entries use random UUIDs (no collision issue)
            if not entry.id:
                entry.id = str(uuid.uuid4())
            entry.updated_at = datetime.now(tz=UTC)
            self._index[entry.id] = entry
            await self._save_session_entry(entry)
            await self._upsert_vector_record(entry)
            return entry.id

        # For LONG_TERM and DAILY: compute deterministic ID from content
        header = entry.metadata.get("header", "Memory")
        if entry.type == MemoryType.LONG_TERM:
            user_id = entry.metadata.get("user_id", "default")
            target_path = self._get_user_memory_file(user_id)
        else:
            target_path = self._get_daily_file(date.today())

        det_id = _make_deterministic_id(target_path, header, entry.content)

        # Dedup: if this exact content already exists, skip
        if det_id in self._index:
            return det_id

        entry.id = det_id
        entry.metadata["source"] = str(target_path)
        entry.updated_at = datetime.now(tz=UTC)
        self._index[entry.id] = entry
        self._inv_dirty = True

        # Persist to markdown
        await self._append_to_markdown(target_path, entry)
        await self._upsert_vector_record(entry)

        return entry.id

    async def _append_to_markdown(self, path: Path, entry: MemoryEntry) -> None:
        """Append a memory entry to a markdown file."""
        header = entry.metadata.get("header", datetime.now(tz=UTC).strftime("%H:%M"))
        tags_str = " ".join(f"#{t}" for t in entry.tags) if entry.tags else ""

        section = f"\n\n## {header}\n\n{entry.content}"
        if tags_str:
            section += f"\n\n{tags_str}"

        with open(path, "a", encoding="utf-8") as f:
            f.write(section)

    async def _save_session_entry(self, entry: MemoryEntry) -> None:
        """Save a session memory entry."""
        if not entry.session_key:
            return

        # Per-session lock to prevent concurrent read-modify-write corruption
        if entry.session_key not in self._session_write_locks:
            self._session_write_locks[entry.session_key] = asyncio.Lock()

        async with self._session_write_locks[entry.session_key]:
            session_file = self._get_session_file(entry.session_key)

            # Run blocking file I/O in a thread to avoid freezing the event loop
            def _read_and_append_once() -> list[dict[str, object]]:
                session_data = []
                if session_file.exists():
                    try:
                        session_data = json.loads(session_file.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as exc:
                        logger.warning("Discarding corrupt session file %s: %s", session_file, exc)
                session_data.append(
                    {
                        "id": entry.id,
                        "role": entry.role,
                        "content": entry.content,
                        "timestamp": entry.created_at.isoformat(),
                        "metadata": entry.metadata,
                    }
                )
                # Atomic write: tmp file + replace to prevent corruption on crash
                tmp = session_file.with_suffix(".tmp")
                tmp.write_text(json.dumps(session_data, indent=2), encoding="utf-8")
                # On Windows, os.replace can fail with PermissionError if another
                # process briefly holds the file handle.
                tmp.replace(session_file)
                return session_data

            session_data: list[dict[str, object]] | None = None
            for _attempt in range(5):
                try:
                    session_data = await asyncio.to_thread(_read_and_append_once)
                    break
                except PermissionError:
                    if _attempt == 4:
                        raise
                    await asyncio.sleep(0.01 * (2**_attempt))

            if session_data is None:
                return

            # Update session index
            await self._update_session_index(entry.session_key, entry, session_data)

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Get a memory entry by ID."""
        return self._index.get(entry_id)

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry and rewrite source file."""
        if entry_id not in self._index:
            return False

        entry = self._index.pop(entry_id)
        self._inv_dirty = True

        # Rewrite the source markdown file without this entry
        source = entry.metadata.get("source")
        if source:
            self._rewrite_markdown(Path(source))

        await self._delete_vector_record(entry_id)

        return True

    def _rewrite_markdown(self, path: Path) -> None:
        """Reconstruct a markdown file from remaining index entries for that file."""
        source_str = str(path)
        entries = [e for e in self._index.values() if e.metadata.get("source") == source_str]

        if not entries:
            # No entries left — remove file
            if path.exists():
                path.unlink()
            return

        parts = []
        for e in entries:
            header = e.metadata.get("header", "Memory")
            tags_str = " ".join(f"#{t}" for t in e.tags) if e.tags else ""
            section = f"## {header}\n\n{e.content}"
            if tags_str:
                section += f"\n\n{tags_str}"
            parts.append(section)

        path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")

    async def search(
        self,
        query: str | None = None,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search memories using word-overlap scoring."""
        candidates: list[tuple[float, MemoryEntry]] = []
        query_words = _tokenize(query) if query else set()

        if query_words:
            # Rebuild inverted index if dirty
            if self._inv_dirty:
                self._rebuild_inverted()

            # Narrow candidates to entries sharing at least one query word
            candidate_ids: set[str] = set()
            for w in query_words:
                candidate_ids |= self._inverted.get(w, set())

            for eid in candidate_ids:
                entry = self._index.get(eid)
                if not entry:
                    continue

                # Type filter
                if memory_type and entry.type != memory_type:
                    continue

                # Tag filter
                if tags and not any(t in entry.tags for t in tags):
                    continue

                content_words = _tokenize(entry.content)
                header = entry.metadata.get("header", "")
                if header:
                    content_words |= _tokenize(header)

                overlap = query_words & content_words
                if not overlap:
                    continue
                score = len(overlap) / len(query_words)
                candidates.append((score, entry))
        else:
            # No query — apply type/tag filters across all entries
            for entry in self._index.values():
                if memory_type and entry.type != memory_type:
                    continue
                if tags and not any(t in entry.tags for t in tags):
                    continue
                candidates.append((0.0, entry))

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        return [entry for _, entry in candidates[:limit]]

    async def get_by_type(
        self, memory_type: MemoryType, limit: int = 100, **kwargs
    ) -> list[MemoryEntry]:
        """Get all memories of a specific type.

        For LONG_TERM type, accepts optional user_id kwarg to scope retrieval.
        """
        user_id = kwargs.get("user_id")
        results = []
        for e in self._index.values():
            if e.type != memory_type:
                continue
            # Scope LONG_TERM to user_id if provided
            if user_id and memory_type == MemoryType.LONG_TERM:
                entry_uid = e.metadata.get("user_id", "default")
                if entry_uid != user_id:
                    continue
            results.append(e)
            if len(results) >= limit:
                break
        return results

    async def get_session(self, session_key: str) -> list[MemoryEntry]:
        """Get session history."""
        session_file = self._get_session_file(session_key)

        if not session_file.exists():
            return []

        try:
            raw = await asyncio.to_thread(lambda: session_file.read_text(encoding="utf-8"))
            data = json.loads(raw)
            return [
                MemoryEntry(
                    id=item["id"],
                    type=MemoryType.SESSION,
                    content=item["content"],
                    role=item.get("role"),
                    session_key=session_key,
                    created_at=_ensure_utc(datetime.fromisoformat(item["timestamp"])),
                    metadata=item.get("metadata", {}),
                )
                for item in data
            ]
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Could not parse session file %s: %s", session_file, exc)
            return []

    async def clear_session(self, session_key: str) -> int:
        """Clear session history."""
        session_file = self._get_session_file(session_key)

        def _clear() -> tuple[int, list[str]]:
            if session_file.exists():
                try:
                    data = json.loads(session_file.read_text(encoding="utf-8"))
                    count = len(data)
                    ids = [str(item.get("id", "")) for item in data if item.get("id")]
                    session_file.unlink()
                    return count, ids
                except json.JSONDecodeError as exc:
                    logger.warning("Corrupt session file removed %s: %s", session_file, exc)
                    session_file.unlink()
                    return 0, []
            return 0, []

        count, entry_ids = await asyncio.to_thread(_clear)
        for entry_id in entry_ids:
            await self._delete_vector_record(entry_id)
            self._index.pop(entry_id, None)
        return count
