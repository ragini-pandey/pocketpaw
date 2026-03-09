"""Agent Router — registry-based backend selection.

Uses the backend registry to lazily discover and instantiate the
configured agent backend. Falls back to ``claude_agent_sdk`` when
the requested backend is unavailable.
"""

import logging
from collections.abc import AsyncIterator

from pocketpaw.agents.backend import BackendInfo
from pocketpaw.agents.protocol import AgentEvent
from pocketpaw.agents.registry import get_backend_class
from pocketpaw.config import Settings

logger = logging.getLogger(__name__)


class AgentRouter:
    """Routes agent requests to the selected backend via the registry."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._backend = None
        self._active_backend_name: str | None = None

        # NEW: fallback backend list (optional)
        self._fallback_backends: list[str] = getattr(settings, "fallback_backends", [])

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Initialize the selected agent backend from the registry."""
        backend_name = self.settings.agent_backend

        cls = get_backend_class(backend_name)
        if cls is None:
            logger.warning(
                "Backend '%s' unavailable — falling back to claude_agent_sdk",
                backend_name,
            )
            cls = get_backend_class("claude_agent_sdk")
            backend_name = "claude_agent_sdk"

        if cls is None:
            logger.error("No agent backend could be loaded")
            self._active_backend_name = None
            return

        try:
            self._backend = cls(self.settings)
            self._active_backend_name = backend_name
            info = cls.info()
            logger.info("🚀 Backend: %s", info.display_name)
        except Exception as exc:
            logger.error("Failed to initialize '%s' backend: %s", backend_name, exc)
            self._active_backend_name = None

    async def run(
        self,
        message: str,
        *,
        system_prompt: str | None = None,
        history: list[dict] | None = None,
        session_key: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent with simple backend fallback."""
        backend_chain: list[str] = []
        # primary backend
        if self._active_backend_name:
            backend_chain.append(self._active_backend_name)
        # always include Claude fallback
        backend_chain=[]
        if self._active_backend_name:
            backend_chain.append(self._active_backend_name)
        backend_chain.extend(self._fallback_backends)
        if "claude_agent_sdk" not in backend_chain:
            backend_chain.append("claude_agent_sdk")
        last_error: str | None = None
        for backend_name in backend_chain:
            cls = get_backend_class(backend_name)
            if cls is None:
                logger.warning("Backend '%s' not available", backend_name)
                continue
            try:
                backend = cls(self.settings)
                logger.info("Attempting backend: %s", backend_name)
                async for event in backend.run(
                    message,
                    system_prompt=system_prompt,
                    history=history,
                    session_key=session_key,
                ):
                    if event.type == "error":
                        # capture error but try next backend
                        last_error = str(event.content)
                        logger.warning(
                            "Backend '%s' returned error: %s",
                            backend_name,
                            event.content,
                        )
                        break
                    yield event
                else:
                    # backend completed successfully
                    return
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Backend '%s' raised exception: %s",
                    backend_name,
                    exc,
                )
                continue
        yield AgentEvent(
            type="error",
            content=last_error or "All configured backends failed",
        )
        yield AgentEvent(type="done", content="")

    async def stop(self) -> None:
        """Stop the agent."""
        if self._backend:
            await self._backend.stop()

    def get_backend_info(self) -> BackendInfo | None:
        """Return metadata about the active backend."""
        if self._backend is None:
            return None
        return self._backend.info()