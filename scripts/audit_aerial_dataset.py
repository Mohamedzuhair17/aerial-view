"""Audit extracted paired aerial imagery and xBD-style building labels.

Expected layout::

    DATASET_ROOT/
      train/{images,labels}/
      train_aug/{images,labels}/   # optional
      validation/{images,labels}/ # optional

Images are named <scene>_pre.jpg and <scene>_post.jpg. Label JSON follows
the xBD ``features.xy`` format and contains building polygons in WKT.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
WKT_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
IMAGE_STEM = re.compile(r"^(.*)_(pre|post)(_aug\d+)?$", re.IGNORECASE)
CLASS_MAP = {
    "no-damage": "No Damage",
    "minor-damage": "Minor",
    "major-damage": "Major",
    "destroyed": "Destroyed",
    "un-classified": "Unclassified",
    "unclassified": "Unclassified",
}


def _image_stems(directory: Path) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {"pre": set(), "post": set()}
    if not directory.is_dir():
        return found
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = IMAGE_STEM.match(path.stem)
        if match:
            scene, date, augmentation = match.groups()
            found[date.lower()].add(scene + (augmentation or ""))
    return found


def audit_split(root: Path, split: str) -> dict:
    split_root = root / split
    images_dir, labels_dir = split_root / "images", split_root / "labels"
    images = _image_stems(images_dir)
    scenes = images["pre"] & images["post"]
    label_paths = sorted(labels_dir.glob("*.json")) if labels_dir.is_dir() else []
    label_stems = {p.stem for p in label_paths}
    class_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    disaster_counts: Counter[str] = Counter()
    building_count = 0
    malformed: list[str] = []
    coordinate_bounds: list[tuple[float, float, float, float]] = []

    for path in label_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            metadata = data.get("metadata", {})
            if metadata.get("disaster"):
                disaster_counts[str(metadata["disaster"])] += 1
            features = data.get("features", {}).get("xy", [])
            if not isinstance(features, list):
                raise ValueError("features.xy is not a list")
            for feature in features:
                props = feature.get("properties", {})
                if props.get("feature_type") != "building":
                    continue
                raw = str(props.get("subtype", "unknown")).strip().lower()
                subtype_counts[raw] += 1
                class_counts[CLASS_MAP.get(raw, "Unknown")] += 1
                building_count += 1
                nums = [float(v) for v in WKT_NUMBER.findall(str(feature.get("wkt", "")))]
                if len(nums) >= 6 and len(nums) % 2 == 0:
                    xs, ys = nums[::2], nums[1::2]
                    coordinate_bounds.append((min(xs), min(ys), max(xs), max(ys)))
        except Exception as exc:  # report every bad annotation rather than aborting the audit
            malformed.append(f"{path.name}: {exc}")

    return {
        "split": split,
        "images": {
            "pre": len(images["pre"]), "post": len(images["post"]),
            "paired_scenes": len(scenes),
            "missing_pre_count": len(images["post"] - images["pre"]),
            "missing_pre_examples": sorted(images["post"] - images["pre"])[:20],
            "missing_post_count": len(images["pre"] - images["post"]),
            "missing_post_examples": sorted(images["pre"] - images["post"])[:20],
        },
        "labels": {
            "files": len(label_paths),
            "without_image_pair_count": len(label_stems - scenes),
            "without_image_pair_examples": sorted(label_stems - scenes)[:20],
            "without_label_count": len(scenes - label_stems),
            "without_label_examples": sorted(scenes - label_stems)[:20],
            "malformed": malformed,
        },
        "buildings": building_count,
        "damage_counts": dict(sorted(class_counts.items())),
        "raw_subtype_counts": dict(sorted(subtype_counts.items())),
        "disaster_scene_counts": dict(sorted(disaster_counts.items())),
        "polygon_bounds": {
            "min_x": min((b[0] for b in coordinate_bounds), default=None),
            "min_y": min((b[1] for b in coordinate_bounds), default=None),
            "max_x": max((b[2] for b in coordinate_bounds), default=None),
            "max_y": max((b[3] for b in coordinate_bounds), default=None),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Extracted dataset directory")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    root = args.dataset_root
    split_names = [name for name in ("train", "train_aug", "validation", "test") if (root / name).exists()]
    if not split_names:
        raise SystemExit(f"No expected split directories found under {root}")
    report = {"dataset_root": str(root), "splits": [audit_split(root, name) for name in split_names]}
    if "train" in split_names and "train_aug" in split_names:
        train = _image_stems(root / "train" / "images")
        aug = _image_stems(root / "train_aug" / "images")
        overlap = sorted((train["pre"] & train["post"]) & (aug["pre"] & aug["post"]))
        report["augmentation_overlap"] = {
            "paired_scene_name_overlap": len(overlap),
            "same_stem_examples": overlap[:20],
            "note": "Matching names only detects unrenamed copies; visually transformed copies need perceptual-hash review.",
        }
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
