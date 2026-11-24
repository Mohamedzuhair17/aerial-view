"""Optional GeoTIFF helpers and pixel-polygon to EPSG:4326 conversion."""
from __future__ import annotations

from typing import Any

import numpy as np


def read_geotiff(data: bytes, max_pixels: int, max_side: int):
    """Return BGR uint8 image and georeferencing metadata from a GeoTIFF.

    Requires rasterio (requirements-geo.txt). Pixel values are percentile-stretched
    per band for visualization/model input; the affine transform is adjusted for
    any resize so vectorized pixel coordinates still map to the source footprint.
    """
    try:
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.enums import Resampling
        from rasterio.transform import Affine
    except ImportError as exc:
        raise RuntimeError("GeoTIFF support requires: pip install -r requirements-geo.txt") from exc

    with MemoryFile(data) as mem:
        with mem.open() as src:
            width, height = src.width, src.height
            if width * height > max_pixels * 4:
                raise ValueError(f"GeoTIFF dimensions exceed the supported pixel limit ({max_pixels * 4:,})")
            scale = min(1.0, max_side / max(width, height))
            out_w, out_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
            count = src.count
            if count >= 3:
                bands = src.read([1, 2, 3], out_shape=(3, out_h, out_w), resampling=Resampling.bilinear)
            elif count == 1:
                band = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
                bands = np.stack([band, band, band])
            else:
                raise ValueError("GeoTIFF must contain at least one raster band")

            rgb = np.empty((out_h, out_w, 3), dtype=np.uint8)
            from app.config import settings
            for i in range(3):
                raw_plane = np.asarray(bands[i])
                plane = raw_plane.astype(np.float32)
                finite = plane[np.isfinite(plane)]
                if finite.size == 0:
                    rgb[..., i] = 0
                    continue
                if raw_plane.dtype == np.uint8:
                    lo, hi = 0.0, 255.0
                else:
                    # Fixed shared stretch across both acquisitions avoids independent
                    # per-image histogram stretching creating artificial change signals.
                    lo, hi = settings.geotiff_value_min, settings.geotiff_value_max
                    if hi <= lo:
                        raise ValueError("GEOTIFF_VALUE_MAX must be greater than GEOTIFF_VALUE_MIN")
                scaled = np.clip((plane - lo) * (255.0 / (hi - lo)), 0, 255)
                scaled[~np.isfinite(scaled)] = 0
                rgb[..., i] = scaled.astype(np.uint8)

            # Model APIs use OpenCV BGR ordering.
            bgr = rgb[..., ::-1].copy()
            transform = src.transform * Affine.scale(width / out_w, height / out_h)
            metadata = {
                "crs": src.crs.to_string() if src.crs else None,
                "affine": list(transform)[:6],
                "width": out_w,
                "height": out_h,
                "source_width": width,
                "source_height": height,
                "georeferenced": bool(src.crs) and not src.transform.is_identity,
            }
            if out_w * out_h > max_pixels:
                raise ValueError(f"Decoded image exceeds MAX_IMAGE_PIXELS={max_pixels}")
            return bgr, metadata


def reproject_before_to_after(before: np.ndarray, before_meta: dict, after_meta: dict, out_shape: tuple[int, int]):
    """Warp a georeferenced pre-event image onto the post-event raster grid."""
    try:
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import Affine
    except ImportError as exc:
        raise RuntimeError("GeoTIFF reprojection requires rasterio; install requirements-geo.txt") from exc
    if not before_meta.get("crs") or not after_meta.get("crs") or not before_meta.get("georeferenced") or not after_meta.get("georeferenced"):
        raise ValueError("Both GeoTIFFs must define a CRS and a non-identity affine transform for geospatial alignment")
    src_transform = Affine(*before_meta["affine"])
    dst_transform = Affine(*after_meta["affine"])
    h, w = out_shape
    destination = np.zeros((h, w, 3), dtype=np.uint8)
    valid_source = np.ones(before.shape[:2], dtype=np.uint8)
    valid_destination = np.zeros((h, w), dtype=np.uint8)
    reproject(
        source=valid_source, destination=valid_destination,
        src_transform=src_transform, src_crs=before_meta["crs"],
        dst_transform=dst_transform, dst_crs=after_meta["crs"],
        resampling=Resampling.nearest, src_nodata=0, dst_nodata=0,
    )
    for band in range(3):
        reproject(
            source=before[..., band], destination=destination[..., band],
            src_transform=src_transform, src_crs=before_meta["crs"],
            dst_transform=dst_transform, dst_crs=after_meta["crs"],
            resampling=Resampling.bilinear, dst_nodata=0,
        )
    overlap = float(np.count_nonzero(valid_destination)) / float(h * w)
    return destination, overlap


def attach_geographic_geometries(rows: list[dict], georef: dict | None):
    """Attach EPSG:4326 geometry from segmentation masks and the after-image affine.

    Rasterio polygonization follows pixel-cell boundaries and preserves interior
    rings, unlike OpenCV external-contour approximations used for the pixel overlay.
    """
    if not georef or not georef.get("crs") or not georef.get("georeferenced"):
        for row in rows:
            row.pop("_mask", None)
        return rows
    try:
        from rasterio.features import shapes
        from rasterio.transform import Affine
        from rasterio.warp import transform_geom
    except ImportError as exc:
        raise RuntimeError("GeoTIFF vectorization requires rasterio; install requirements-geo.txt") from exc

    transform = Affine(*georef["affine"])
    for row in rows:
        mask = row.pop("_mask", None)
        geometry = None
        if mask is not None:
            x1, y1, x2, y2 = row["bbox"]
            local = np.asarray(mask[y1:y2, x1:x2], dtype=np.uint8)
            if local.size and local.any():
                local_transform = transform * Affine.translation(x1, y1)
                parts = [geom for geom, value in shapes(local, mask=local.astype(bool), transform=local_transform) if int(value) == 1]
                if len(parts) == 1:
                    geometry = parts[0]
                elif parts:
                    geometry = {"type": "MultiPolygon", "coordinates": [part["coordinates"] for part in parts if part["type"] == "Polygon"]}

        # Compatibility fallback for older records or callers without a mask.
        if geometry is None:
            polygons = row.get("polygons") or [row.get("polygon") or []]
            rings = []
            for polygon in polygons:
                if not isinstance(polygon, list) or len(polygon) < 3:
                    continue
                ring = []
                for point in polygon:
                    x, y = transform * (float(point[0]), float(point[1]))
                    ring.append([float(x), float(y)])
                if ring and ring[0] != ring[-1]:
                    ring.append(ring[0])
                if len(ring) >= 4:
                    rings.append([ring])
            if rings:
                geometry = {"type": "MultiPolygon", "coordinates": rings}

        row["geometry_geojson"] = transform_geom(georef["crs"], "EPSG:4326", geometry, precision=7) if geometry else None
    return rows
