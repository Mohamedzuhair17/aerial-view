from collections import Counter
from typing import Optional

import cv2
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

from app import db
from app.cv.render import render_overlay
from app.schemas import Building, JobStatus
from app.spatial.tools import DAMAGE_CLASSES, QUADRANTS, norm_damage
from app.storage import job_directory

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _job(job_id: str) -> dict:
    try:
        job_directory(job_id)
    except ValueError:
        raise HTTPException(404, "job not found")
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return j


def _done(job_id: str) -> dict:
    j = _job(job_id)
    if j["status"] != "done":
        raise HTTPException(409, f"job is {j['status']}")
    return j


@router.get("/{job_id}", response_model=JobStatus)
def status(job_id: str):
    j = _job(job_id)
    return JobStatus(job_id=job_id, **{k: j.get(k) for k in ("status", "error", "width", "height", "n_buildings", "model_mode", "alignment")})


@router.get("/{job_id}/buildings", response_model=list[Building])
def buildings(job_id: str, damage: Optional[str] = None, quadrant: Optional[str] = None):
    _done(job_id)
    rows = db.get_buildings(job_id)
    if damage:
        try:
            wanted = norm_damage([d.strip() for d in damage.split(",")])
        except ValueError as e:
            raise HTTPException(400, str(e))
        rows = [r for r in rows if r["damage"] in wanted]
    if quadrant:
        q = quadrant.strip().upper()
        if q not in QUADRANTS:
            raise HTTPException(400, f"quadrant must be one of {', '.join(QUADRANTS)}")
        rows = [r for r in rows if r["quadrant"] == q]
    return rows


@router.get("/{job_id}/summary")
def summary(job_id: str):
    j = _done(job_id)
    rows = db.get_buildings(job_id)
    by_damage = Counter(r["damage"] for r in rows)
    matrix = {q: {c: 0 for c in DAMAGE_CLASSES} for q in QUADRANTS}
    for r in rows:
        matrix[r["quadrant"]][r["damage"]] += 1
    return {"job_id": job_id, "total": len(rows), "width": j["width"], "height": j["height"],
            "model_mode": j["model_mode"], "alignment": j.get("alignment"),
            "by_damage": {c: by_damage.get(c, 0) for c in DAMAGE_CLASSES},
            "by_quadrant": matrix}


@router.get("/{job_id}/image/{kind}")
def image(job_id: str, kind: str):
    _job(job_id)
    if kind not in ("before", "after", "overlay"):
        raise HTTPException(404, "kind must be before | after | overlay")
    p = job_directory(job_id) / f"{kind}.png"
    if not p.exists():
        raise HTTPException(404, "image not ready")
    return FileResponse(p, media_type="image/png")


@router.get("/{job_id}/overlay")
def overlay(job_id: str, highlight: str = Query("", max_length=5000, description="comma-separated building IDs"), grid: bool = True):
    _done(job_id)
    after_path = job_directory(job_id) / "after.png"
    after = cv2.imread(str(after_path))
    if after is None:
        raise HTTPException(404, "post-disaster image is unavailable")
    ids = {i.strip().upper() for i in highlight.split(",") if i.strip()}
    ok, buf = cv2.imencode(".png", render_overlay(after, db.get_buildings(job_id), ids, grid))
    if not ok:
        raise HTTPException(500, "Could not encode overlay image")
    return Response(buf.tobytes(), media_type="image/png")
