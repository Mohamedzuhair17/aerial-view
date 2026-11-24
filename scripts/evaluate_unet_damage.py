#!/usr/bin/env python
"""Evaluate the saved U-Net on the validation split and save mask previews."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as torch_functional
from torch.utils.data import DataLoader

# 1. IMPORT LIBRARIES
# ================================================================

import argparse
import os
import json
import random
import time
import copy
import csv

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from shapely import wkt
from shapely.geometry import shape, Polygon, MultiPolygon, GeometryCollection


# ================================================================


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "dataset v1"
SPLIT_DIR = SOURCE_DIR / "validation"  # The dataset has no independent test split.
CHECKPOINT_PATH = PROJECT_DIR / "runs" / "unet_train_only" / "checkpoints" / "unet_best.pth"
OUTPUT_DIR = PROJECT_DIR / "runs" / "unet_test_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_SIZE = 256
NUM_CLASSES = 5
IGNORE_INDEX = 255
NUM_DISPLAY = 5
BATCH_SIZE = 1
NUM_WORKERS = 0
CLASS_NAMES = ["Background", "No Damage", "Minor Damage", "Major Damage", "Destroyed"]
CLASS_COLORS = np.array([
    [0, 0, 0], [0, 180, 0], [255, 220, 0], [255, 120, 0], [220, 0, 0]
], dtype=np.uint8)
CLASS_IDS = {"no-damage": 1, "minor-damage": 2, "major-damage": 3, "destroyed": 4}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5. DAMAGE LABEL NORMALIZATION
# ================================================================

def normalize_damage_label(label):

    if label is None:
        return None

    label = str(label).strip().lower()

    label = label.replace("_", "-")
    label = label.replace(" ", "-")

    aliases = {

        "no-damage": "no-damage",
        "nodamage": "no-damage",

        "minor-damage": "minor-damage",
        "minor": "minor-damage",

        "major-damage": "major-damage",
        "major": "major-damage",

        "destroyed": "destroyed",
        "destroy": "destroyed"

    }

    return aliases.get(label)


# ================================================================
# 6. EXTRACT FEATURES FROM JSON
# ================================================================

def extract_features(annotation):

    features = annotation.get("features", [])

    # Your annotation format may store features inside features.xy

    if isinstance(features, dict):

        features = features.get("xy", [])

    if not isinstance(features, list):

        return []

    return features


# ================================================================
# 7. EXTRACT POLYGON GEOMETRY
# ================================================================

def extract_geometry(feature):

    geometry = None

    # WKT polygon

    wkt_string = feature.get("wkt")

    if wkt_string:

        try:

            geometry = wkt.loads(wkt_string)

        except Exception:

            return None

    # GeoJSON geometry

    if geometry is None and feature.get("geometry"):

        try:

            geometry = shape(feature["geometry"])

        except Exception:

            return None

    if geometry is None:
        return None

    if geometry.is_empty:
        return None

    if not geometry.is_valid:

        geometry = geometry.buffer(0)

    return geometry


# ================================================================
# 8. EXTRACT CLASS ID
# ================================================================

def extract_class_id(feature):

    properties = feature.get("properties", {})

    if not isinstance(properties, dict):

        properties = {}

    label = properties.get("subtype")

    if label is None:

        label = properties.get("damage")

    if label is None:

        label = feature.get("subtype")

    normalized = normalize_damage_label(label)

    if normalized is None:

        # Ignore un-classified and unsupported labels

        return None

    return CLASS_IDS[normalized]


# ================================================================
# 9. GET POLYGONS
# ================================================================

def get_polygons(geometry):

    if geometry is None or geometry.is_empty:

        return []

    if isinstance(geometry, Polygon):

        return [geometry]

    if isinstance(geometry, MultiPolygon):

        return list(geometry.geoms)

    if isinstance(geometry, GeometryCollection):

        polygons = []

        for item in geometry.geoms:

            polygons.extend(get_polygons(item))

        return polygons

    return []


# ================================================================
# 10. CONVERT JSON POLYGONS INTO SEGMENTATION MASK
# ================================================================

def create_segmentation_mask(annotation, width, height):

    # Background = 0

    mask = Image.new(
        "L",
        (width, height),
        color=0
    )

    draw = ImageDraw.Draw(mask)

    features = extract_features(annotation)

    valid_polygon_count = 0

    for feature in features:

        class_id = extract_class_id(feature)

        if class_id is None:

            continue

        geometry = extract_geometry(feature)

        if geometry is None:

            continue

        polygons = get_polygons(geometry)

        for polygon in polygons:

            coordinates = []

            for x, y in polygon.exterior.coords:

                x = max(0, min(float(x), width - 1))

                y = max(0, min(float(y), height - 1))

                coordinates.append((x, y))

            if len(coordinates) < 3:

                continue

            draw.polygon(
                coordinates,
                fill=class_id
            )

            valid_polygon_count += 1

    return mask, valid_polygon_count


# ================================================================
# 11. FIND IMAGE PAIRS AND JSON ANNOTATIONS
# ================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff"
}


def build_dataset_records(split_dir):

    if not split_dir.exists():

        raise FileNotFoundError(
            f"Dataset directory not found: {split_dir}"
        )

    print("\nScanning:", split_dir)

    all_files = [
        p for p in split_dir.rglob("*")
        if p.is_file()
    ]

    # Index pre-disaster images by their base filename

    pre_index = {}

    for path in all_files:

        if path.suffix.lower() not in IMAGE_EXTENSIONS:

            continue

        if path.stem.lower().endswith("_pre"):

            base_name = path.stem[:-4].lower()

            pre_index.setdefault(base_name, []).append(path)

    # Index JSON annotations

    json_index = {}

    for path in all_files:

        if path.suffix.lower() == ".json":

            json_index.setdefault(
                path.stem.lower(), []
            ).append(path)

    # Find post-disaster images

    post_images = [

        path for path in all_files

        if path.suffix.lower() in IMAGE_EXTENSIONS

        and path.stem.lower().endswith("_post")

    ]

    records = []

    skipped = 0

    for post_path in sorted(post_images):

        base_name = post_path.stem[:-5].lower()

        # First try the same folder as the post image

        expected_pre = post_path.with_name(
            base_name + "_pre" + post_path.suffix
        )

        if expected_pre.exists():

            pre_path = expected_pre

        else:

            matches = pre_index.get(base_name, [])

            if len(matches) == 1:

                pre_path = matches[0]

            else:

                skipped += 1

                print(
                    "Missing or ambiguous pre image:",
                    post_path.name
                )

                continue

        # Find the matching JSON annotation

        expected_json = post_path.with_name(
            base_name + ".json"
        )

        if expected_json.exists():

            json_path = expected_json

        else:

            matches = json_index.get(base_name, [])

            if len(matches) == 1:

                json_path = matches[0]

            else:

                skipped += 1

                print(
                    "Missing or ambiguous JSON:",
                    post_path.name
                )

                continue

        records.append({

            "pre": pre_path,

            "post": post_path,

            "json": json_path

        })

    print("Post images found:", len(post_images))

    print("Valid image pairs:", len(records))

    print("Skipped:", skipped)

    if not records:

        raise RuntimeError(
            f"No valid image pairs found in {split_dir}"
        )

    return records


# ================================================================
# 12. PYTORCH DATASET
# ================================================================

class DisasterDamageDataset(Dataset):

    def __init__(
        self,
        records,
        image_size=256,
        augment=False
    ):

        self.records = records

        self.image_size = image_size

        self.augment = augment

    def __len__(self):

        return len(self.records)

    def load_json(self, json_path):

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    def image_to_tensor(self, image):

        image = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        return torch.from_numpy(
            np.ascontiguousarray(image)
        ).float()

    def __getitem__(self, index):

        record = self.records[index]

        pre_path = record["pre"]

        post_path = record["post"]

        json_path = record["json"]

        # ----------------------------------------------------
        # LOAD POST-DISASTER IMAGE
        # ----------------------------------------------------

        post_image = Image.open(
            post_path
        ).convert("RGB")

        original_width, original_height = post_image.size

        # ----------------------------------------------------
        # LOAD PRE-DISASTER IMAGE
        # ----------------------------------------------------

        pre_image = Image.open(
            pre_path
        ).convert("RGB")

        # ----------------------------------------------------
        # LOAD JSON ANNOTATION
        # ----------------------------------------------------

        annotation = self.load_json(json_path)

        # ----------------------------------------------------
        # GENERATE SEGMENTATION MASK
        # ----------------------------------------------------

        mask, polygon_count = create_segmentation_mask(
            annotation,
            original_width,
            original_height
        )

        # ----------------------------------------------------
        # RESIZE IMAGES AND MASK
        # ----------------------------------------------------

        resize_size = (
            self.image_size,
            self.image_size
        )

        post_image = post_image.resize(
            resize_size,
            Image.Resampling.BILINEAR
        )

        pre_image = pre_image.resize(
            resize_size,
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            resize_size,
            Image.Resampling.NEAREST
        )

        # ----------------------------------------------------
        # DATA AUGMENTATION
        # ----------------------------------------------------

        if self.augment:

            # Horizontal flip

            if random.random() < 0.5:

                post_image = post_image.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

                pre_image = pre_image.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

                mask = mask.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

            # Vertical flip

            if random.random() < 0.5:

                post_image = post_image.transpose(
                    Image.Transpose.FLIP_TOP_BOTTOM
                )

                pre_image = pre_image.transpose(
                    Image.Transpose.FLIP_TOP_BOTTOM
                )

                mask = mask.transpose(
                    Image.Transpose.FLIP_TOP_BOTTOM
                )

        # ----------------------------------------------------
        # CONVERT TO TENSORS
        # ----------------------------------------------------

        post_tensor = self.image_to_tensor(post_image)

        pre_tensor = self.image_to_tensor(pre_image)

        mask_array = np.asarray(
            mask,
            dtype=np.int64
        ).copy()

        mask_tensor = torch.from_numpy(
            mask_array
        ).long()

        # ----------------------------------------------------
        # VALIDATE MASK
        # ----------------------------------------------------

        if mask_tensor.min() < 0 or mask_tensor.max() >= NUM_CLASSES:

            raise ValueError(
                f"Invalid mask class in {post_path.name}"
            )

        return {

            "image": post_tensor,

            "mask": mask_tensor,

            "pre_image": pre_tensor,

            "name": post_path.name,

            "pre_name": pre_path.name,

            "polygon_count": polygon_count

        }


# ================================================================


# 15. STANDARD U-NET ARCHITECTURE
# ================================================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.block(x)


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = ConvBlock(
            in_channels,
            out_channels
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        features = self.conv(x)

        pooled = self.pool(features)

        return features, pooled


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = ConvBlock(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):

        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:

            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat(
            [x, skip],
            dim=1
        )

        return self.conv(x)


class UNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        num_classes=5
    ):

        super().__init__()

        # ENCODER

        self.encoder1 = EncoderBlock(
            in_channels, 64
        )

        self.encoder2 = EncoderBlock(
            64, 128
        )

        self.encoder3 = EncoderBlock(
            128, 256
        )

        self.encoder4 = EncoderBlock(
            256, 512
        )

        # BOTTLENECK

        self.bottleneck = ConvBlock(
            512, 1024
        )

        # DECODER

        self.decoder4 = DecoderBlock(
            1024, 512, 512
        )

        self.decoder3 = DecoderBlock(
            512, 256, 256
        )

        self.decoder2 = DecoderBlock(
            256, 128, 128
        )

        self.decoder1 = DecoderBlock(
            128, 64, 64
        )

        # SEGMENTATION HEAD

        self.segmentation_head = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # ENCODER

        e1, p1 = self.encoder1(x)

        e2, p2 = self.encoder2(p1)

        e3, p3 = self.encoder3(p2)

        e4, p4 = self.encoder4(p3)

        # BOTTLENECK

        b = self.bottleneck(p4)

        # DECODER

        d4 = self.decoder4(b, e4)

        d3 = self.decoder3(d4, e3)

        d2 = self.decoder2(d3, e2)

        d1 = self.decoder1(d2, e1)

        # OUTPUT LOGITS

        output = self.segmentation_head(d1)

        return output


# ================================================================


def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not SPLIT_DIR.is_dir():
        raise FileNotFoundError(f"Validation split not found: {SPLIT_DIR}")

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Evaluation split: {SPLIT_DIR} (validation; no separate test split is present)")

    records = build_dataset_records(SPLIT_DIR)
    dataset = DisasterDamageDataset(records, image_size=IMAGE_SIZE, augment=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda")

    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        state = checkpoint["model"]
    else:
        state = checkpoint
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    displayed = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(DEVICE, dtype=torch.float32, non_blocking=True)
            masks = batch["mask"].to(DEVICE, dtype=torch.long, non_blocking=True)
            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch_functional.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            predictions = logits.argmax(dim=1)
            gt = masks.cpu().numpy().ravel()
            pred = predictions.cpu().numpy().ravel()
            valid = ((gt != IGNORE_INDEX) & (gt >= 0) & (gt < NUM_CLASSES)
                     & (pred >= 0) & (pred < NUM_CLASSES))
            confusion += np.bincount(
                NUM_CLASSES * gt[valid] + pred[valid],
                minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
            if len(displayed) < NUM_DISPLAY:
                displayed.append((batch["image"][0].cpu(), masks[0].cpu(), predictions[0].cpu(), batch["name"][0]))

    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1e-10)
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-10)
    precision = tp / np.maximum(tp + fp, 1e-10)
    recall = tp / np.maximum(tp + fn, 1e-10)
    pixel_accuracy = tp.sum() / max(confusion.sum(), 1)
    present = (confusion.sum(axis=1)[1:] + confusion.sum(axis=0)[1:]) > 0
    miou = float(iou[1:][present].mean()) if present.any() else 0.0
    mean_dice = float(dice[1:][present].mean()) if present.any() else 0.0

    print("\nVALIDATION RESULTS")
    print(f"Pixel accuracy: {pixel_accuracy:.4f}")
    print(f"Foreground mIoU: {miou:.4f}")
    print(f"Foreground Dice: {mean_dice:.4f}")
    print("\nPER-CLASS METRICS")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<18} IoU={iou[i]:.4f} Dice={dice[i]:.4f} Precision={precision[i]:.4f} Recall={recall[i]:.4f}")
    print("\nCONFUSION MATRIX\n", confusion)

    results = {
        "split": "validation",
        "checkpoint": str(CHECKPOINT_PATH),
        "pixel_accuracy": float(pixel_accuracy),
        "foreground_miou": miou,
        "foreground_dice": mean_dice,
        "class_metrics": {
            name: {"iou": float(iou[i]), "dice": float(dice[i]),
                  "precision": float(precision[i]), "recall": float(recall[i])}
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion.tolist(),
    }
    (OUTPUT_DIR / "validation_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    def colorize(mask):
        mask = np.asarray(mask, dtype=np.int64)
        colors = np.zeros((*mask.shape, 3), dtype=np.uint8)
        valid_mask = (mask >= 0) & (mask < NUM_CLASSES)
        colors[valid_mask] = CLASS_COLORS[mask[valid_mask]]
        return colors

    for sample_no, (image_tensor, gt_tensor, pred_tensor, name) in enumerate(displayed, start=1):
        image = np.transpose(image_tensor.numpy(), (1, 2, 0)).clip(0, 1)
        gt_color, pred_color = colorize(gt_tensor.numpy()), colorize(pred_tensor.numpy())
        overlay = np.clip(0.55 * image + 0.45 * (pred_color.astype(np.float32) / 255), 0, 1)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, view, title in zip(axes, (image, gt_color, pred_color, overlay),
                                   ("Post image", "Ground truth", "Prediction", "Overlay")):
            ax.imshow(view)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"Validation sample {sample_no}: {name}")
        fig.tight_layout()
        output_path = OUTPUT_DIR / f"validation_sample_{sample_no}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", output_path)

    print("\nMetrics:", OUTPUT_DIR / "validation_metrics.json")
    print("Results directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
