import json
import logging
import uuid
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile

from app import db
from app.config import settings
from app.cv.geospatial import attach_geographic_geometries, read_geotiff, reproject_before_to_after
from app.cv.render import render_overlay
from app.storage import job_directory

router = APIRouter(tags=["analysis"])
log = logging.getLogger("analyze")


async def _read_limited(upload: UploadFile, name: str) -> bytes:
    limit = settings.max_upload_mb * 1024 * 1024
    chunks, total = [], 0
    while True:
        chunk = await upload.read(min(1024 * 1024, limit + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(413, f"{name} exceeds {settings.max_upload_mb} MB")
        chunks.append(chunk)
    if not chunks:
        raise HTTPException(400, f"{name} image is empty")
    return b"".join(chunks)


def _decode(data: bytes, name: str, filename: str | None = None):
    is_tiff = Path(filename or "").suffix.lower() in {".tif", ".tiff"}
    if is_tiff:
        if not settings.geospatial_enabled:
            raise HTTPException(415, "GeoTIFF uploads are disabled")
        try:
            image, metadata = read_geotiff(data, settings.max_image_pixels, settings.max_image_side)
            return image, metadata
        except RuntimeError as exc:
            raise HTTPException(415, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(413 if "pixel" in str(exc).lower() or "dimensions" in str(exc).lower() else 400, str(exc)) from exc
        except Exception as exc:
            log.exception("Could not decode %s GeoTIFF", name)
            raise HTTPException(400, f"{name} is not a valid readable GeoTIFF") from exc

    try:
        from PIL import Image
        with Image.open(BytesIO(data)) as probe:
            if probe.format not in {"PNG", "JPEG"}:
                raise HTTPException(400, f"{name} must be PNG/JPG or GeoTIFF")
            source_w, source_h = probe.size
            if source_w * source_h > settings.max_image_pixels * 4:
                raise HTTPException(413, f"{name} image dimensions exceed the supported pixel limit")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"{name} is not a readable PNG/JPG image") from exc

    max_side = max(source_w, source_h)
    if max_side > settings.max_image_side * 4:
        decode_flag = cv2.IMREAD_REDUCED_COLOR_8
    elif max_side > settings.max_image_side * 2:
        decode_flag = cv2.IMREAD_REDUCED_COLOR_4
    elif max_side > settings.max_image_side:
        decode_flag = cv2.IMREAD_REDUCED_COLOR_2
    else:
        decode_flag = cv2.IMREAD_COLOR
    img = cv2.imdecode(np.frombuffer(data, np.uint8), decode_flag)
    if img is None:
        raise HTTPException(400, f"{name} is not a readable PNG/JPG image; GeoTIFF files must use .tif/.tiff")
    h, w = img.shape[:2]
    scale = min(1.0, settings.max_image_side / max(h, w))
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if img.shape[0] * img.shape[1] > settings.max_image_pixels:
        raise HTTPException(413, f"{name} image exceeds MAX_IMAGE_PIXELS={settings.max_image_pixels}")
    return img, None


def _register_before(before: np.ndarray, after: np.ndarray):
    """Conservative ORB/RANSAC registration; return unverified rather than trusting weak transforms."""
    h, w = after.shape[:2]
    bh, bw = before.shape[:2]
    ar_before, ar_after = bw / max(1, bh), w / max(1, h)
    relative_delta = abs(ar_before - ar_after) / max(ar_after, 1e-9)
    if relative_delta > settings.max_aspect_ratio_delta:
        return before, {"status": "unverified", "reason": "aspect_ratio_mismatch", "matches": 0, "inliers": 0, "overlap": 0.0}
    if before.shape[:2] != (h, w):
        before = cv2.resize(before, (w, h), interpolation=cv2.INTER_AREA)
    if not settings.registration_enabled:
        return before, {"status": "disabled", "matches": 0, "inliers": 0, "overlap": None}

    g1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=8)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return before, {"status": "unverified", "reason": "insufficient_features", "matches": 0, "inliers": 0, "overlap": 0.0}

    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < settings.registration_min_matches:
        return before, {"status": "unverified", "reason": "insufficient_matches", "matches": len(good), "inliers": 0, "overlap": 0.0}

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    try:
        H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    except cv2.error:
        return before, {"status": "unverified", "reason": "homography_estimation_failed", "matches": len(good), "inliers": 0, "overlap": 0.0}
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    inlier_ratio = inliers / max(1, len(good))
    if H is None or not np.isfinite(H).all() or inliers < settings.registration_min_matches or inlier_ratio < settings.registration_min_inlier_ratio:
        return before, {"status": "unverified", "reason": "weak_homography", "matches": len(good), "inliers": inliers, "inlier_ratio": round(inlier_ratio, 3), "overlap": 0.0}

    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    try:
        projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    except cv2.error:
        return before, {"status": "unverified", "reason": "invalid_projection", "matches": len(good), "inliers": inliers, "inlier_ratio": round(inlier_ratio, 3), "overlap": 0.0}
    if not np.isfinite(projected).all():
        return before, {"status": "unverified", "reason": "invalid_projection", "matches": len(good), "inliers": inliers, "inlier_ratio": round(inlier_ratio, 3), "overlap": 0.0}
    projected_area = abs(cv2.contourArea(projected.astype(np.float32)))
    area_ratio = projected_area / max(1.0, float(w * h))
    if not 0.5 <= area_ratio <= 2.0:
        return before, {"status": "unverified", "reason": "implausible_transform", "matches": len(good), "inliers": inliers, "inlier_ratio": round(inlier_ratio, 3), "overlap": 0.0}

    source_valid = np.full((h, w), 255, np.uint8)
    try:
        warped_valid = cv2.warpPerspective(source_valid, H, (w, h), flags=cv2.INTER_NEAREST)
    except cv2.error:
        return before, {"status": "unverified", "reason": "warp_failed", "matches": len(good), "inliers": inliers, "inlier_ratio": round(inlier_ratio, 3), "overlap": 0.0}
    overlap = float(np.count_nonzero(warped_valid)) / float(h * w)
    if overlap < settings.registration_min_overlap:
        return before, {"status": "unverified", "reason": "low_overlap", "matches": len(good), "inliers": inliers, "inlier_ratio": round(inlier_ratio, 3), "overlap": round(overlap, 3)}

    try:
        aligned = cv2.warpPerspective(before, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    except cv2.error:
        return before, {"status": "unverified", "reason": "warp_failed", "matches": len(good), "inliers": inliers, "inlier_ratio": round(inlier_ratio, 3), "overlap": round(overlap, 3)}
    return aligned, {"status": "aligned", "matches": len(good), "inliers": inliers, "inlier_ratio": round(inlier_ratio, 3), "overlap": round(overlap, 3), "homography": H.tolist()}


def _run_job(pipe, job_id: str):
    d = job_directory(job_id)
    try:
        db.update_job(job_id, status="running")
        before, after = cv2.imread(str(d / "before.png")), cv2.imread(str(d / "after.png"))
        if before is None or after is None:
            raise RuntimeError("Stored analysis images could not be read")
        rows = pipe.run(before, after)
        metadata_path = d / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        rows = attach_geographic_geometries(rows, metadata.get("after_georef"))
        db.save_buildings(job_id, rows)
        overlay = render_overlay(after, rows)
        if not cv2.imwrite(str(d / "overlay.png"), overlay):
            raise RuntimeError("Could not write analysis overlay")
        db.update_job(job_id, status="done", n_buildings=len(rows), model_mode=pipe.mode)
    except Exception as exc:
        log.exception("job %s failed", job_id)
        try:
            db.update_job(job_id, status="failed", error=str(exc)[:500])
        except Exception:
            log.exception("could not persist failure state for job %s", job_id)


@router.post("/analyze", status_code=202)
async def analyze(request: Request, background: BackgroundTasks,
                  before: UploadFile = File(..., description="pre-disaster image"),
                  after: UploadFile = File(..., description="post-disaster image")):
    before_data = await _read_limited(before, "before")
    after_data = await _read_limited(after, "after")
    b, before_georef = _decode(before_data, "before", before.filename)
    a, after_georef = _decode(after_data, "after", after.filename)

    if before_georef and after_georef and before_georef.get("georeferenced") and after_georef.get("georeferenced"):
        try:
            b, overlap = reproject_before_to_after(b, before_georef, after_georef, a.shape[:2])
            if overlap < settings.registration_min_overlap:
                raise ValueError(f"GeoTIFF footprints overlap only {overlap:.1%}; minimum is {settings.registration_min_overlap:.1%}")
            alignment = {"status": "geospatially_aligned", "method": "rasterio_reproject", "overlap": round(overlap, 3)}
        except Exception as exc:
            raise HTTPException(422, f"Could not align GeoTIFFs using their CRS/affine transforms: {exc}") from exc
    else:
        b, alignment = _register_before(b, a)
        if b.shape[:2] != a.shape[:2]:
            raise HTTPException(422, "Before and after images have incompatible aspect ratios. Pre-align or crop them to the same footprint before upload.")

    if settings.require_alignment and alignment["status"] not in {"aligned", "geospatially_aligned"}:
        raise HTTPException(422, {"detail": "Before/after image alignment could not be verified", "alignment": alignment})

    job_id = uuid.uuid4().hex
    directory = job_directory(job_id)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        if not cv2.imwrite(str(directory / "before.png"), b) or not cv2.imwrite(str(directory / "after.png"), a):
            raise RuntimeError("Could not persist uploaded images")
        metadata = {"alignment": alignment, "before_georef": before_georef, "after_georef": after_georef}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
        db.create_job(job_id, a.shape[1], a.shape[0], alignment=alignment)
    except Exception:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)
        raise

    if settings.job_queue_backend == "rq":
        try:
            from app.worker import enqueue_job
            enqueue_job(job_id)
        except Exception as exc:
            db.update_job(job_id, status="failed", error=f"Could not enqueue job: {exc}"[:500])
            raise HTTPException(503, "Background worker queue is unavailable") from exc
    else:
        background.add_task(_run_job, request.app.state.pipe, job_id)

    return {"job_id": job_id, "status": "queued", "alignment": alignment}
