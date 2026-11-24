"""Dataset helpers for the paired xBD-style aerial dataset."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["No Damage", "Minor", "Major", "Destroyed"]
CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}
SUBTYPE_MAP = {
    "no-damage": "No Damage",
    "minor-damage": "Minor",
    "major-damage": "Major",
    "destroyed": "Destroyed",
}
IMAGE_STEM = re.compile(r"^(.*)_(pre|post)(_aug\d+)?$", re.IGNORECASE)
COORDINATE = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


@dataclass
class Scene:
    scene_id: str
    before_path: Path
    after_path: Path
    label_path: Path


@dataclass
class Building:
    polygon: np.ndarray  # Nx2 float xy coordinates
    damage: str | None


def parse_polygon(wkt: str) -> np.ndarray:
    """Parse the first outer ring of a POLYGON WKT into image xy points."""
    # xBD building geometries are single POLYGONs. Ignore any interior rings;
    # they do not affect the building footprint used by this pipeline.
    text = wkt.strip()
    if text.upper().startswith("POLYGON"):
        start = text.find("((")
        end = text.find("))", start + 2)
        if start < 0 or end < 0:
            return np.empty((0, 2), np.float32)
        text = text[start + 2:end]
    elif text.upper().startswith("MULTIPOLYGON"):
        start = text.find("(((")
        end = text.find(")))", start + 3)
        if start < 0 or end < 0:
            return np.empty((0, 2), np.float32)
        text = text[start + 3:end]
    else:
        return np.empty((0, 2), np.float32)
    points = [(float(x), float(y)) for x, y in COORDINATE.findall(text)]
    if len(points) >= 4 and points[0] == points[-1]:
        points.pop()
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def list_scenes(dataset_root: Path, split: str) -> list[Scene]:
    """Find complete pre/post/label triplets for one split."""
    split_root = dataset_root / split
    image_dir, label_dir = split_root / "images", split_root / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and labels/ under {split_root}")
    files: dict[str, dict[str, Path]] = {}
    for path in image_dir.iterdir():
        match = IMAGE_STEM.match(path.stem)
        if match and path.is_file():
            prefix, date, aug = match.groups()
            scene_id = prefix + (aug or "")
            files.setdefault(scene_id, {})[date.lower()] = path
    scenes: list[Scene] = []
    incomplete: list[str] = []
    for scene_id, pair in sorted(files.items()):
        label_path = label_dir / f"{scene_id}.json"
        if "pre" not in pair or "post" not in pair or not label_path.is_file():
            incomplete.append(scene_id)
            continue
        scenes.append(Scene(scene_id, pair["pre"], pair["post"], label_path))
    if incomplete:
        raise ValueError(f"{split} contains {len(incomplete)} incomplete pairs, e.g. {incomplete[:5]}")
    if not scenes:
        raise ValueError(f"No complete scenes found in {split_root}")
    return scenes


def read_scene(scene: Scene) -> tuple[np.ndarray, np.ndarray, list[Building]]:
    """Return BGR before/after images and usable building polygons."""
    before = cv2.imread(str(scene.before_path), cv2.IMREAD_COLOR)
    after = cv2.imread(str(scene.after_path), cv2.IMREAD_COLOR)
    if before is None or after is None:
        raise ValueError(f"Could not decode image pair for {scene.scene_id}")
    if before.shape != after.shape:
        raise ValueError(f"Image pair dimensions differ for {scene.scene_id}: {before.shape} vs {after.shape}")
    data = json.loads(scene.label_path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    expected = (int(metadata.get("height", after.shape[0])), int(metadata.get("width", after.shape[1])))
    if after.shape[:2] != expected:
        raise ValueError(f"Label dimensions {expected} do not match image {after.shape[:2]} for {scene.scene_id}")
    h, w = after.shape[:2]
    buildings: list[Building] = []
    for feature in data.get("features", {}).get("xy", []):
        props = feature.get("properties", {})
        if props.get("feature_type") != "building":
            continue
        polygon = parse_polygon(str(feature.get("wkt", "")))
        if len(polygon) < 3:
            continue
        polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)
        if abs(cv2.contourArea(polygon.astype(np.float32))) < 1:
            continue
        subtype = str(props.get("subtype", "")).strip().lower()
        buildings.append(Building(polygon, SUBTYPE_MAP.get(subtype)))
    return before, after, buildings


def instance_masks(
    buildings: list[Building], height: int, width: int,
    output_height: int | None = None, output_width: int | None = None,
) -> np.ndarray:
    """Rasterize polygons directly into one compact contiguous target array."""
    out_h = output_height or height
    out_w = output_width or width
    storage = np.zeros((len(buildings), out_h, out_w), dtype=np.uint8)
    count = 0
    scale = np.array([out_w / width, out_h / height], dtype=np.float32)
    for building in buildings:
        polygon = np.rint(building.polygon * scale).astype(np.int32)
        cv2.fillPoly(storage[count], [polygon], 1)
        if int(storage[count].sum()) >= 4:
            count += 1
    return storage[:count]


def masked_crop(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop = image[y1:y2, x1:x2].copy()
    local_mask = mask[y1:y2, x1:x2].astype(bool)
    pixels = crop[local_mask]
    if not len(pixels):
        return None
    crop[~local_mask] = np.median(pixels, axis=0).astype(crop.dtype)
    return crop
