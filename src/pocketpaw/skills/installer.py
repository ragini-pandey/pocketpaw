"""Shared skill installer -- clone GitHub repos and install SKILL.md directories.

Created: 2026-03-22
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

INSTALL_DIR = Path.home() / ".agents" / "skills"


async def install_skills_from_github(
    owner: str,
    repo: str,
    skill_name: str | None = None,
    prefix_filter: str | None = None,
    timeout: float = 60,
) -> list[str]:
    """Clone a GitHub repo and install SKILL.md directories.

    Args:
        owner: GitHub owner (e.g. "googleworkspace").
        repo: GitHub repo name (e.g. "cli").
        skill_name: Install only this specific skill (by directory name).
        prefix_filter: Only install skills whose directory name starts with this
            prefix (e.g. "gws-"). Ignored when *skill_name* is set.
        timeout: Git clone timeout in seconds.

    Returns:
        List of installed skill names.

    Raises:
        RuntimeError: If the clone fails or no skills are found.
    """
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "clone",
            "--depth=1",
            f"https://github.com/{owner}/{repo}.git",
            tmpdir,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"Clone failed: {err}")

        tmp = Path(tmpdir)
        skill_dirs: list[tuple[str, Path]] = []

        if skill_name:
            # Look for a specific skill by name
            for candidate in [tmp / skill_name, tmp / "skills" / skill_name]:
                if (candidate / "SKILL.md").exists():
                    skill_dirs.append((skill_name, candidate))
                    break
        else:
            # Scan for all skills
            for scan_dir in [tmp, tmp / "skills"]:
                if not scan_dir.is_dir():
                    continue
                for item in sorted(scan_dir.iterdir()):
                    if not item.is_dir() or not (item / "SKILL.md").exists():
                        continue
                    if prefix_filter and not item.name.startswith(prefix_filter):
                        continue
                    skill_dirs.append((item.name, item))

        if not skill_dirs:
            target = skill_name or f"{owner}/{repo}"
            raise RuntimeError(f"No SKILL.md found for '{target}'")

        installed: list[str] = []
        for name, src_dir in skill_dirs:
            dest = INSTALL_DIR / name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src_dir, dest)
            installed.append(name)

        logger.info("Installed %d skills from %s/%s", len(installed), owner, repo)
        return installed
