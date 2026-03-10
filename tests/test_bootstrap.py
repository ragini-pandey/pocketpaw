# Tests for Bootstrap System
# Created: 2026-02-02


import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from pocketpaw.bootstrap.context_builder import AgentContextBuilder
from pocketpaw.bootstrap.default_provider import DefaultBootstrapProvider
from pocketpaw.bootstrap.protocol import BootstrapContext
from pocketpaw.bus.events import Channel


@pytest.fixture
def temp_identity_path():
    """Create a temporary directory for identity files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestBootstrapContext:
    """Tests for BootstrapContext."""

    def test_to_system_prompt(self):
        ctx = BootstrapContext(
            name="TestAgent",
            identity="I am a test agent.",
            soul="I exist to test.",
            style="Be terse.",
            knowledge=["Fact 1", "Fact 2"],
        )

        prompt = ctx.to_system_prompt()
        assert "# Identity: TestAgent" in prompt
        assert "I am a test agent." in prompt
        assert "I exist to test." in prompt
        assert "Be terse." in prompt
        assert "# Key Knowledge" in prompt
        assert "- Fact 1" in prompt

    def test_to_system_prompt_has_identity_xml_tags(self):
        """Identity block is wrapped in <identity> XML tags for LLM emphasis."""
        ctx = BootstrapContext(
            name="TestAgent",
            identity="I am a test agent.",
            soul="Soul text.",
            style="Style text.",
        )
        prompt = ctx.to_system_prompt()
        assert "<identity>" in prompt
        assert "</identity>" in prompt
        # Core identity content must be inside the XML block
        open_tag = prompt.index("<identity>")
        close_tag = prompt.index("</identity>")
        identity_block = prompt[open_tag:close_tag]
        assert "# Identity: TestAgent" in identity_block
        assert "I am a test agent." in identity_block
        assert "Soul text." in identity_block
        assert "Style text." in identity_block

    def test_instructions_outside_identity_xml_block(self):
        """Instructions (tool docs) appear after the closing </identity> tag."""
        ctx = BootstrapContext(
            name="Agent",
            identity="ID",
            soul="Soul",
            style="Style",
            instructions="TOOL_DOCS_HERE",
        )
        prompt = ctx.to_system_prompt()
        close_tag = prompt.index("</identity>")
        instr_pos = prompt.index("TOOL_DOCS_HERE")
        assert instr_pos > close_tag, "Instructions must come after </identity>"


class TestDefaultBootstrapProvider:
    """Tests for DefaultBootstrapProvider."""

    @pytest.mark.asyncio
    async def test_defaults_creation(self, temp_identity_path):
        provider = DefaultBootstrapProvider(base_path=temp_identity_path)

        # Check files created
        assert (temp_identity_path / "IDENTITY.md").exists()
        assert (temp_identity_path / "SOUL.md").exists()
        assert (temp_identity_path / "STYLE.md").exists()

        # Check content loading
        ctx = await provider.get_context()
        assert ctx.name == "PocketPaw"
        assert "You are PocketPaw" in ctx.identity

    @pytest.mark.asyncio
    async def test_custom_content(self, temp_identity_path):
        # Create provider (makes defaults)
        provider = DefaultBootstrapProvider(base_path=temp_identity_path)

        # Modify files
        (temp_identity_path / "IDENTITY.md").write_text("I am CustomAgent")

        # Reload
        ctx = await provider.get_context()
        assert ctx.identity == "I am CustomAgent"

    @pytest.mark.asyncio
    async def test_get_context_uses_cache(self, temp_identity_path):
        """Second call returns cached content without re-reading from disk."""
        from pocketpaw.bootstrap import default_provider as dp

        dp._identity_file_cache.clear()
        provider = DefaultBootstrapProvider(base_path=temp_identity_path)

        ctx1 = await provider.get_context()
        cached_snapshot = dict(dp._identity_file_cache)
        assert len(cached_snapshot) > 0

        ctx2 = await provider.get_context()
        # Cache entries unchanged (same mtime → no re-read)
        assert dp._identity_file_cache == cached_snapshot
        assert ctx1.identity == ctx2.identity

    @pytest.mark.asyncio
    async def test_cache_invalidates_on_file_change(self, temp_identity_path):
        """Cache is invalidated when a file's mtime changes."""
        import os
        import time

        from pocketpaw.bootstrap import default_provider as dp

        dp._identity_file_cache.clear()
        provider = DefaultBootstrapProvider(base_path=temp_identity_path)

        ctx1 = await provider.get_context()
        assert "You are PocketPaw" in ctx1.identity

        identity = temp_identity_path / "IDENTITY.md"
        identity.write_text("Updated identity", encoding="utf-8")
        # Force mtime forward regardless of filesystem resolution
        future = time.time() + 10
        os.utime(identity, (future, future))

        ctx2 = await provider.get_context()
        assert ctx2.identity == "Updated identity"

    @pytest.mark.asyncio
    async def test_cache_handles_missing_file(self, temp_identity_path):
        """Cache returns empty string for missing files."""
        from pocketpaw.bootstrap import default_provider as dp

        dp._identity_file_cache.clear()
        provider = DefaultBootstrapProvider(base_path=temp_identity_path)

        # Remove USER.md
        (temp_identity_path / "USER.md").unlink()

        ctx = await provider.get_context()
        assert ctx.user_profile == ""

    @pytest.mark.asyncio
    async def test_mtime_cache_returns_same_object(self, temp_identity_path):
        """Repeated get_context() calls with unchanged files return the same instance."""
        provider = DefaultBootstrapProvider(base_path=temp_identity_path)
        ctx1 = await provider.get_context()
        ctx2 = await provider.get_context()
        assert ctx1 is ctx2  # same object — instance-level cache hit

    @pytest.mark.asyncio
    async def test_mtime_cache_invalidated_on_file_change(self, temp_identity_path):
        """Cache is invalidated when a file is modified."""
        provider = DefaultBootstrapProvider(base_path=temp_identity_path)
        ctx1 = await provider.get_context()

        # Modify a file (bumps its mtime)
        (temp_identity_path / "IDENTITY.md").write_text("Updated identity")

        ctx2 = await provider.get_context()
        assert ctx2 is not ctx1  # cache invalidated
        assert ctx2.identity == "Updated identity"

    @pytest.mark.asyncio
    async def test_missing_file_returns_empty_string(self, temp_identity_path):
        """A missing identity file yields an empty string rather than an exception."""
        provider = DefaultBootstrapProvider(base_path=temp_identity_path)
        # Remove a file that was just created by _ensure_defaults
        (temp_identity_path / "SOUL.md").unlink()
        ctx = await provider.get_context()
        assert ctx.soul == ""

    @pytest.mark.asyncio
    async def test_unreadable_file_returns_empty_string(self, temp_identity_path):
        """An unreadable file yields an empty string and logs a warning."""
        import unittest.mock as mock

        provider = DefaultBootstrapProvider(base_path=temp_identity_path)
        # Force an OSError (e.g. permissions) on STYLE.md
        with mock.patch("pathlib.Path.read_bytes", side_effect=OSError("permission denied")):
            ctx = await provider.get_context()
        assert ctx.style == ""


class TestAgentContextBuilder:
    """Tests for AgentContextBuilder."""

    @pytest.mark.asyncio
    async def test_build_full_prompt(self):
        # Mock provider
        mock_provider = MagicMock()
        mock_provider.get_context = AsyncMock(
            return_value=BootstrapContext(
                name="Test", identity="Identity", soul="Soul", style="Style"
            )
        )

        # Mock memory
        mock_memory = MagicMock()
        mock_memory.get_context_for_agent = AsyncMock(return_value="Memory Context")

        builder = AgentContextBuilder(
            bootstrap_provider=mock_provider,
            memory_manager=mock_memory,
        )

        prompt = await builder.build_system_prompt(include_memory=True)

        assert "Identity" in prompt
        assert "Memory Context" in prompt
        assert "# Memory Context" in prompt

    @pytest.mark.asyncio
    async def test_build_with_channel_hint(self):
        mock_provider = MagicMock()
        mock_provider.get_context = AsyncMock(
            return_value=BootstrapContext(
                name="Test", identity="Identity", soul="Soul", style="Style"
            )
        )
        mock_memory = MagicMock()
        mock_memory.get_context_for_agent = AsyncMock(return_value="")

        builder = AgentContextBuilder(bootstrap_provider=mock_provider, memory_manager=mock_memory)

        prompt = await builder.build_system_prompt(channel=Channel.WHATSAPP)
        assert "# Response Format" in prompt
        assert "WhatsApp" in prompt

    @pytest.mark.asyncio
    async def test_build_passthrough_channel_no_hint(self):
        mock_provider = MagicMock()
        mock_provider.get_context = AsyncMock(
            return_value=BootstrapContext(
                name="Test", identity="Identity", soul="Soul", style="Style"
            )
        )
        mock_memory = MagicMock()
        mock_memory.get_context_for_agent = AsyncMock(return_value="")

        builder = AgentContextBuilder(bootstrap_provider=mock_provider, memory_manager=mock_memory)

        prompt = await builder.build_system_prompt(channel=Channel.WEBSOCKET)
        assert "# Response Format" not in prompt

    @pytest.mark.asyncio
    async def test_build_no_memory(self):
        # Mock provider
        mock_provider = MagicMock()
        mock_provider.get_context = AsyncMock(
            return_value=BootstrapContext(
                name="Test", identity="Identity", soul="Soul", style="Style"
            )
        )

        builder = AgentContextBuilder(
            bootstrap_provider=mock_provider,
            memory_manager=MagicMock(),
        )

        prompt = await builder.build_system_prompt(include_memory=False)

        assert "Identity" in prompt
        assert "Memory Context" not in prompt
