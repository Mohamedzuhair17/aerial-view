
import math

import numpy as np

from app.config import settings
from app.models.base import BuildingInstance
from app.models.registry import Models
from app.spatial.tools import DAMAGE_CLASSES


def quadrant_of(cx: float, cy: float, w: int, h: int) -> str:
    """Determine which image quadrant contains the building."""
    return ("N" if cy < h / 2 else "S") + (
        "W" if cx < w / 2 else "E"
    )


class Pipeline:
    """
    Run the configured model stack and produce
    frontend-ready building damage predictions.
    """

    def __init__(self, models: Models):
        self.models = models

    @property
    def mode(self) -> str:
        return self.models.mode

    def run(
        self,
        before: np.ndarray,
        after: np.ndarray
    ) -> list[dict]:

        # --------------------------------------------------
        # 1. Validate input images
        # --------------------------------------------------

        if (
            before is None
            or after is None
            or before.ndim != 3
            or after.ndim != 3
        ):
            raise ValueError(
                "before and after must be decoded HxWx3 images"
            )

        if before.shape != after.shape:
            raise ValueError(
                "before and after images must have identical "
                "dimensions after alignment"
            )

        h, w = after.shape[:2]

        # --------------------------------------------------
        # 2. Run ChangeFormer + U-Net
        # --------------------------------------------------

        if hasattr(self.models.change, "predict_instances"):

            # ChangeFormer:
            # Detect individual buildings using paired images.
            detections = self.models.change.predict_instances(
                before,
                after
            )

            # U-Net:
            # Generate semantic damage regions from post-disaster image.
            semantic_regions = self.models.seg.predict(after)

            # Match ChangeFormer detections with U-Net regions.
            instances = self._refine_detections(
                detections,
                semantic_regions,
                h,
                w
            )

            # Fallback if ChangeFormer does not detect any buildings.
            if not instances:
                instances = semantic_regions

            # Calculate approximate pixel-level change.
            change = cv2_change_mask(before, after)

        else:

            # Generic change detection pipeline.
            change = np.asarray(
                self.models.change.predict(before, after),
                dtype=bool
            )

            if change.shape != (h, w):
                raise ValueError(
                    f"change detector returned {change.shape}; "
                    f"expected {(h, w)}"
                )

            instances = self.models.seg.predict(after)

        # --------------------------------------------------
        # 3. Generate building-level results
        # --------------------------------------------------

        rows: list[dict] = []

        for inst in instances:

            mask = np.asarray(inst.mask, dtype=bool)

            if mask.shape != (h, w):
                raise ValueError(
                    f"segmenter returned mask {mask.shape}; "
                    f"expected {(h, w)}"
                )

            # Building pixel area.
            area = int(mask.sum())

            if area < settings.min_building_area:
                continue

            # --------------------------------------------------
            # 4. Validate bounding box
            # --------------------------------------------------

            x1, y1, x2, y2 = (
                int(v) for v in inst.bbox
            )

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            local_mask = mask[y1:y2, x1:x2]

            if not local_mask.any():
                continue

            # --------------------------------------------------
            # 5. Calculate changed pixel ratio
            # --------------------------------------------------

            ratio = float(change[mask].mean())

            # --------------------------------------------------
            # 6. Extract before and after building crops
            # --------------------------------------------------

            before_crop = before[
                y1:y2, x1:x2
            ].copy()

            after_crop = after[
                y1:y2, x1:x2
            ].copy()

            # Fill non-building pixels with median building colour.
            for crop in (before_crop, after_crop):

                pixels = crop[local_mask]

                if len(pixels):
                    fill = np.median(
                        pixels,
                        axis=0
                    ).astype(crop.dtype)
                else:
                    fill = np.zeros(
                        3,
                        dtype=crop.dtype
                    )

                crop[~local_mask] = fill

            # --------------------------------------------------
            # 7. Determine damage class
            # --------------------------------------------------

            # If U-Net has already assigned a valid damage class,
            # use its prediction.
            if getattr(inst, "damage", None) in DAMAGE_CLASSES[:-1]:

                label = inst.damage
                confidence = inst.confidence

            else:

                # Otherwise use the paired-image classifier.
                label, confidence = self.models.cls.predict(
                    before_crop,
                    after_crop
                )

            # --------------------------------------------------
            # 8. Validate prediction confidence
            # --------------------------------------------------

            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.0

            if (
                not math.isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
            ):
                label = "Unclassified"
                confidence = 0.0

            if label not in DAMAGE_CLASSES[:-1]:
                label = "Unclassified"

            if confidence < settings.min_conf:
                label = "Unclassified"

            # --------------------------------------------------
            # 9. Extract building polygons
            # --------------------------------------------------

            polygons = (
                getattr(inst, "polygons", None)
                or [inst.polygon]
            )

            polygons = [
                p for p in polygons
                if isinstance(p, list) and len(p) >= 3
            ]

            if not polygons:
                continue

            primary = (
                inst.polygon
                if inst.polygon in polygons
                else polygons[0]
            )

            # --------------------------------------------------
            # 10. Prepare frontend result
            # --------------------------------------------------

            rows.append({

                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2
                ],

                "polygon": primary,

                "polygons": polygons,

                "damage": label,

                "conf": round(confidence, 4),

                "changed_ratio": round(ratio, 4),

                "area": area,

                "quadrant": quadrant_of(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    w,
                    h
                ),

                # Internal-only raster mask for GeoTIFF polygonization.
                "_mask": mask,
            })

        # --------------------------------------------------
        # 11. Sort buildings spatially
        # --------------------------------------------------

        rows.sort(
            key=lambda row: (
                row["bbox"][1] // 40,
                row["bbox"][0]
            )
        )

        # Generate building IDs.
        for index, row in enumerate(rows, 1):
            row["id"] = f"B{index:03d}"

        return rows

    # ======================================================
    # CHANGEFORMER + U-NET REFINEMENT
    # ======================================================

    @staticmethod
    def _refine_detections(
        detections,
        regions,
        height: int,
        width: int
    ):
        """
        Match ChangeFormer building detections with U-Net
        semantic regions.

        ChangeFormer:
            Supplies individual building geometry.

        U-Net:
            Supplies damage class and confidence.

        Important:
            ChangeFormer masks, bounding boxes and polygons
            are preserved.
        """

        refined = []

        # --------------------------------------------------
        # 1. Validate U-Net regions
        # --------------------------------------------------

        valid_regions = []

        for region in regions:

            region_mask = np.asarray(
                region.mask,
                dtype=bool
            )

            if region_mask.shape != (height, width):
                continue

            region_area = int(region_mask.sum())

            if region_area == 0:
                continue

            valid_regions.append({

                "region": region,

                "mask": region_mask,

                "area": region_area

            })

        # --------------------------------------------------
        # 2. Match each ChangeFormer detection
        # --------------------------------------------------

        for detection in detections:

            det_mask = np.asarray(
                detection.mask,
                dtype=bool
            )

            if det_mask.shape != (height, width):
                refined.append(detection)
                continue

            det_area = int(det_mask.sum())

            if det_area == 0:
                refined.append(detection)
                continue

            best_region = None
            best_score = 0.0

            # --------------------------------------------------
            # 3. Find the best overlapping U-Net region
            # --------------------------------------------------

            for item in valid_regions:

                region = item["region"]
                region_mask = item["mask"]
                region_area = item["area"]

                intersection = int(
                    np.logical_and(
                        det_mask,
                        region_mask
                    ).sum()
                )

                if intersection == 0:
                    continue

                union = (
                    det_area
                    + region_area
                    - intersection
                )

                # Intersection over Union.
                iou = intersection / max(union, 1)

                # Fraction of the building detection
                # covered by the U-Net region.
                detection_coverage = (
                    intersection / det_area
                )

                # Fraction of the U-Net region
                # covered by the building detection.
                region_coverage = (
                    intersection / region_area
                )

                # Accept sufficiently strong overlap.
                valid_match = (
                    iou >= 0.05
                    or detection_coverage >= 0.35
                    or region_coverage >= 0.60
                )

                if not valid_match:
                    continue

                # Matching score.
                score = max(
                    iou,
                    0.50 * detection_coverage,
                    0.50 * region_coverage
                )

                if score > best_score:

                    best_score = score
                    best_region = region

            # --------------------------------------------------
            # 4. No matching U-Net region
            # --------------------------------------------------

            if best_region is None:

                # Preserve original ChangeFormer prediction.
                refined.append(detection)
                continue

            # --------------------------------------------------
            # 5. Extract U-Net damage prediction
            # --------------------------------------------------

            damage = getattr(
                best_region,
                "damage",
                None
            )

            confidence = getattr(
                best_region,
                "confidence",
                0.0
            )

            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.0

            # Validate U-Net prediction.
            if (
                damage not in DAMAGE_CLASSES[:-1]
                or not math.isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
            ):

                refined.append(detection)
                continue

            # --------------------------------------------------
            # 6. Preserve ChangeFormer geometry
            # --------------------------------------------------

            refined.append(
                BuildingInstance(

                    mask=np.asarray(
                        detection.mask,
                        dtype=bool
                    ),

                    bbox=detection.bbox,

                    polygon=detection.polygon,

                    polygons=detection.polygons,

                    # Only damage information comes from U-Net.
                    damage=damage,

                    confidence=confidence
                )
            )

        return refined


# ==========================================================
# CHANGE MASK GENERATION
# ==========================================================

def cv2_change_mask(
    before: np.ndarray,
    after: np.ndarray
) -> np.ndarray:
    """
    Approximate changed-pixel ratio for display when
    ChangeFormer returns building detections instead of
    a complete pixel-level change mask.
    """

    import cv2

    diff = cv2.absdiff(
        before,
        after
    ).astype(np.float32)

    diff = diff.mean(axis=2)

    return diff > 20.0
