"""Shared skill-installation logic for dashboard and API endpoints."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

from pocketpaw.security.audit import AuditEvent, AuditSeverity, get_audit_logger
from pocketpaw.skills.loader import get_skill_loader

logger = logging.getLogger(__name__)


def _ignore_symlinks(src: str, names: list[str]) -> set[str]:
    """Return names that are symlinks so ``shutil.copytree`` skips them."""
    return {n for n in names if os.path.islink(os.path.join(src, n))}


class SkillInstallError(Exception):
    """Raised when skill installation fails."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


async def install_skill_from_source(source: str) -> list[str]:
    """Validate, clone, and install a skill from a GitHub source string.

    Args:
        source: GitHub source in format "owner/repo" or "owner/repo/skill_name"

    Returns:
        List of installed skill names.

    Raises:
        SkillInstallError: If validation, cloning, or installation fails.
    """
    source = source.strip()
    if not source:
        raise SkillInstallError("Missing 'source' field", 400)

    parts = source.split("/")
    if len(parts) < 2 or len(parts) > 3:
        raise SkillInstallError("Source must be owner/repo or owner/repo/skill", 400)

    owner, repo = parts[0], parts[1]
    skill_name = parts[2] if len(parts) == 3 else None

    # Whitelist owner/repo to GitHub's actual naming rules.
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", owner):
        raise SkillInstallError("Invalid owner format", 400)
    if not re.match(r"^[a-zA-Z0-9._-]+$", repo):
        raise SkillInstallError("Invalid repo format", 400)
    if skill_name:
        if skill_name in (".", ".."):
            raise SkillInstallError("Invalid skill name format", 400)
        if not re.match(r"^[a-zA-Z0-9._-]+$", skill_name):
            raise SkillInstallError("Invalid skill name format", 400)

    install_dir = Path.home() / ".agents" / "skills"
    install_dir.mkdir(parents=True, exist_ok=True)

    try:
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
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                logger.warning(
                    "git clone stderr for %s/%s: %s",
                    owner,
                    repo,
                    stderr.decode(errors="replace").strip(),
                )
                raise SkillInstallError("Skill clone failed", 500)

            tmp = Path(tmpdir)
            skill_dirs: list[tuple[str, Path]] = []

            if skill_name:
                for candidate in [tmp / skill_name, tmp / "skills" / skill_name]:
                    if (candidate / "SKILL.md").exists():
                        skill_dirs.append((skill_name, candidate))
                        break
            else:
                for scan_dir in [tmp, tmp / "skills"]:
                    if not scan_dir.is_dir():
                        continue
                    for item in sorted(scan_dir.iterdir()):
                        if item.is_dir() and (item / "SKILL.md").exists():
                            skill_dirs.append((item.name, item))

            if not skill_dirs:
                raise SkillInstallError(f"No SKILL.md found for '{skill_name or source}'", 404)

            installed = []
            for name, src_dir in skill_dirs:
                # Reject path traversal and validate discovered directory names.
                if name in (".", "..") or not re.match(r"^[a-zA-Z0-9._-]+$", name):
                    logger.warning("Skipping skill directory with invalid name: %r", name)
                    continue
                dest = install_dir / name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src_dir, dest, ignore=_ignore_symlinks)
                installed.append(name)

            if not installed:
                raise SkillInstallError("No valid skill directories found after validation", 400)

            get_audit_logger().log(
                AuditEvent.create(
                    severity=AuditSeverity.INFO,
                    actor="dashboard_user",
                    action="skill_install",
                    target=f"{owner}/{repo}",
                    status="success",
                    installed=installed,
                )
            )

            loader = get_skill_loader()
            loader.reload()
            return installed

    except TimeoutError:
        raise SkillInstallError("Clone timed out (30s)", 504)
    except SkillInstallError:
        raise
    except Exception:
        logger.exception("Skill install failed")
        raise SkillInstallError("Skill install failed", 500)
