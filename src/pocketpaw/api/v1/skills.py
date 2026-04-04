# Skills router — list, search, install, remove, reload.
# Created: 2026-02-20

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request

from pocketpaw.api.v1.schemas.common import StatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Skills"])


@router.get("/skills")
async def list_installed_skills():
    """List all installed user-invocable skills."""
    from pocketpaw.skills import get_skill_loader

    loader = get_skill_loader()
    loader.reload()
    return [
        {"name": s.name, "description": s.description, "argument_hint": s.argument_hint}
        for s in loader.get_invocable()
    ]


@router.get("/skills/search")
async def search_skills_library(q: str = "", limit: int = Query(30, ge=1, le=100)):
    """Proxy search to skills.sh API."""
    import httpx

    if not q:
        return {"skills": [], "count": 0}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://skills.sh/api/search",
                params={"q": q, "limit": limit},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        logger.warning("skills.sh search failed: %s", exc)
        return {"skills": [], "count": 0, "error": str(exc)}


@router.post("/skills/install")
async def install_skill(request: Request):
    """Install a skill by cloning its GitHub repo."""
    from pocketpaw.skills import get_skill_loader
    from pocketpaw.skills.installer import install_skills_from_github

    data = await request.json()
    source = data.get("source", "").strip()
    if not source:
        raise HTTPException(status_code=400, detail="Missing 'source' field")

    if ".." in source or ";" in source or "|" in source or "&" in source:
        raise HTTPException(status_code=400, detail="Invalid source format")

    parts = source.split("/")
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail="Source must be owner/repo or owner/repo/skill")

    owner, repo = parts[0], parts[1]
    skill_name = parts[2] if len(parts) >= 3 else None

    try:
        installed = await install_skills_from_github(
            owner=owner,
            repo=repo,
            skill_name=skill_name,
            timeout=30,
        )
        loader = get_skill_loader()
        loader.reload()
        return {"status": "ok", "installed": installed}

    except TimeoutError:
        raise HTTPException(status_code=504, detail="Clone timed out (30s)")
    except RuntimeError as exc:
        detail = str(exc)
        if "No SKILL.md" in detail:
            raise HTTPException(status_code=404, detail=detail)
        raise HTTPException(status_code=500, detail=detail)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Skill install failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/skills/remove")
async def remove_skill(request: Request):
    """Remove an installed skill by deleting its directory."""
    import shutil
    from pathlib import Path

    from pocketpaw.skills import get_skill_loader

    data = await request.json()
    name = data.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Missing 'name' field")

    if ".." in name or "/" in name or ";" in name or "|" in name or "&" in name:
        raise HTTPException(status_code=400, detail="Invalid name format")

    for base in [Path.home() / ".agents" / "skills", Path.home() / ".pocketpaw" / "skills"]:
        skill_dir = base / name
        if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
            shutil.rmtree(skill_dir)
            loader = get_skill_loader()
            loader.reload()
            return StatusResponse()

    raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")


@router.post("/skills/reload")
async def reload_skills():
    """Force reload skills from disk."""
    from pocketpaw.skills import get_skill_loader

    loader = get_skill_loader()
    skills = loader.reload()
    return {"status": "ok", "count": len([s for s in skills.values() if s.user_invocable])}
