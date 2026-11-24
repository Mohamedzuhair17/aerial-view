
# ================================================================
# STANDARD U-NET FOR DISASTER DAMAGE SEGMENTATION
# PyTorch | CPU / GPU
#
# INPUT:
#   Post-disaster image only
#
# TARGET:
#   Five-class segmentation mask generated from JSON polygons
#
# CLASSES:
#   0 = Background
#   1 = No damage
#   2 = Minor damage
#   3 = Major damage
#   4 = Destroyed
#
# DATASET:
#   dataset v1/
#       train/       (training only)
#       validation/  (checkpoint selection and final evaluation)
#
# IMAGE PAIR NAMING:
#   sample_pre.jpg
#   sample_post.jpg
#   sample.json
# ================================================================


# ================================================================
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
# 2. CONFIGURATION
# ================================================================

# Find the repository from the current Jupyter working directory.
PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "dataset v1"
OUTPUT_DIR = PROJECT_DIR / "runs" / "unet_train_only"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

TRAIN_DIR = SOURCE_DIR / "train"  # Regular training split only.
VAL_DIR = SOURCE_DIR / "validation"
TEST_DIR = VAL_DIR  # No independent test split exists; final metrics reuse validation.
assert TRAIN_DIR.name == "train" and TRAIN_DIR.name != "train_aug"
assert TRAIN_DIR.is_dir() and VAL_DIR.is_dir()


# Model configuration

IMAGE_SIZE = 256
IN_CHANNELS = 3
NUM_CLASSES = 5

CLASS_NAMES = [
    "Background",
    "No Damage",
    "Minor Damage",
    "Major Damage",
    "Destroyed"
]

CLASS_IDS = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4
}


# Training configuration (overridable from the command line).
parser = argparse.ArgumentParser(description="Train U-Net damage segmentation on dataset v1/train.")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--image-size", type=int, default=256)
parser.add_argument("--workers", type=int, default=0)
args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
IMAGE_SIZE = args.image_size
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
EARLY_STOPPING_PATIENCE = 10
NUM_WORKERS = args.workers
SEED = 42

# Confidence is not used for semantic segmentation.
# Every pixel receives one of the five class IDs.

# Set True only if pixels outside annotated polygons
# should be considered background.
UNANNOTATED_PIXELS_ARE_BACKGROUND = True


# ================================================================
# 3. REPRODUCIBILITY AND DEVICE
# ================================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PIN_MEMORY = DEVICE.type == "cuda"

USE_AMP = DEVICE.type == "cuda"
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True


# ================================================================
# 4. CREATE OUTPUT DIRECTORIES
# ================================================================

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
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
# 13. LOAD DATASETS
# ================================================================

print("\n" + "=" * 65)

print("LOADING DISASTER DAMAGE DATASET")

print("=" * 65)


train_records = build_dataset_records(TRAIN_DIR)

val_records = build_dataset_records(VAL_DIR)

test_records = build_dataset_records(TEST_DIR)


train_dataset = DisasterDamageDataset(
    train_records,
    image_size=IMAGE_SIZE,
    augment=True
)

val_dataset = DisasterDamageDataset(
    val_records,
    image_size=IMAGE_SIZE,
    augment=False
)

test_dataset = DisasterDamageDataset(
    test_records,
    image_size=IMAGE_SIZE,
    augment=False
)


print("\nDATASET SUMMARY")

print("Train:", len(train_dataset))

print("Validation:", len(val_dataset))

print("Final validation set:", len(test_dataset))


# ================================================================
# 14. CREATE DATALOADERS
# ================================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)


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
# 16. DICE LOSS
# ================================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        num_classes=5,
        smooth=1e-6
    ):

        super().__init__()

        self.num_classes = num_classes

        self.smooth = smooth

    def forward(self, logits, targets):

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        targets_one_hot = F.one_hot(
            targets,
            num_classes=self.num_classes
        )

        targets_one_hot = targets_one_hot.permute(
            0, 3, 1, 2
        ).float()

        # Ignore background when calculating foreground Dice

        probabilities = probabilities[:, 1:, :, :]

        targets_one_hot = targets_one_hot[:, 1:, :, :]

        intersection = (
            probabilities * targets_one_hot
        ).sum(dim=(0, 2, 3))

        denominator = (
            probabilities + targets_one_hot
        ).sum(dim=(0, 2, 3))

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice.mean()


# ================================================================
# 17. COMBINED LOSS
# ================================================================

class CombinedLoss(nn.Module):

    def __init__(self, num_classes=5):

        super().__init__()

        self.ce_loss = nn.CrossEntropyLoss()

        self.dice_loss = DiceLoss(
            num_classes=num_classes
        )

    def forward(self, logits, targets):

        ce = self.ce_loss(
            logits,
            targets
        )

        dice = self.dice_loss(
            logits,
            targets
        )

        total_loss = ce + dice

        return total_loss, ce.detach(), dice.detach()


# ================================================================
# 18. INITIALIZE MODEL, LOSS AND OPTIMIZER
# ================================================================

print("\n" + "=" * 65)

print("INITIALIZING U-NET")

print("=" * 65)


model = UNet(
    in_channels=IN_CHANNELS,
    num_classes=NUM_CLASSES
).to(DEVICE)


total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print("Device:", DEVICE)

print("Total parameters:", f"{total_parameters:,}")

print("Trainable parameters:", f"{trainable_parameters:,}")


criterion = CombinedLoss(
    num_classes=NUM_CLASSES
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP
)


# ================================================================
# 19. METRIC CALCULATION
# ================================================================

def calculate_metrics(confusion_matrix):

    confusion_matrix = confusion_matrix.astype(
        np.float64
    )

    true_positive = np.diag(confusion_matrix)

    actual = confusion_matrix.sum(axis=1)

    predicted = confusion_matrix.sum(axis=0)

    union = actual + predicted - true_positive

    iou = np.divide(
        true_positive,
        union,
        out=np.zeros_like(true_positive),
        where=union != 0
    )

    dice_denominator = actual + predicted

    dice = np.divide(
        2 * true_positive,
        dice_denominator,
        out=np.zeros_like(true_positive),
        where=dice_denominator != 0
    )

    pixel_accuracy = (
        true_positive.sum()
        / max(confusion_matrix.sum(), 1)
    )

    # Mean foreground metrics exclude background

    mean_iou = np.mean(iou[1:])

    mean_dice = np.mean(dice[1:])

    return {

        "pixel_accuracy": pixel_accuracy,

        "mean_iou": mean_iou,

        "mean_dice": mean_dice,

        "class_iou": iou,

        "class_dice": dice

    }


def update_confusion_matrix(
    confusion_matrix,
    predictions,
    targets
):

    predictions = predictions.detach().cpu().numpy().ravel()

    targets = targets.detach().cpu().numpy().ravel()

    valid = (
        (targets >= 0)
        & (targets < NUM_CLASSES)
    )

    combined = (
        NUM_CLASSES * targets[valid]
        + predictions[valid]
    )

    counts = np.bincount(
        combined,
        minlength=NUM_CLASSES ** 2
    )

    confusion_matrix += counts.reshape(
        NUM_CLASSES,
        NUM_CLASSES
    )

    return confusion_matrix


# ================================================================
# 20. TRAINING FUNCTION
# ================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler
):

    model.train()

    total_loss = 0.0

    total_ce = 0.0

    total_dice = 0.0

    for batch_index, batch in enumerate(loader):

        images = batch["image"].to(
            DEVICE,
            non_blocking=True
        )

        masks = batch["mask"].to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP
        ):

            outputs = model(images)

            loss, ce, dice = criterion(
                outputs,
                masks
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        total_loss += loss.item()

        total_ce += ce.item()

        total_dice += dice.item()

        if (batch_index + 1) % 25 == 0:

            print(
                f"Batch {batch_index + 1}/{len(loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    number_of_batches = max(len(loader), 1)

    return {

        "loss": total_loss / number_of_batches,

        "ce": total_ce / number_of_batches,

        "dice": total_dice / number_of_batches

    }


# ================================================================
# 21. VALIDATION FUNCTION
# ================================================================

def validate_one_epoch(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    confusion_matrix = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(DEVICE)

            masks = batch["mask"].to(DEVICE)

            with torch.autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP
            ):

                outputs = model(images)

                loss, _, _ = criterion(
                    outputs,
                    masks
                )

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            confusion_matrix = update_confusion_matrix(
                confusion_matrix,
                predictions,
                masks
            )

    average_loss = total_loss / max(len(loader), 1)

    metrics = calculate_metrics(
        confusion_matrix
    )

    return average_loss, metrics


# ================================================================
# 22. SAVE TRAINING HISTORY
# ================================================================

history = {

    "train_loss": [],

    "val_loss": [],

    "train_ce": [],

    "train_dice": [],

    "val_miou": [],

    "val_mdice": [],

    "val_pixel_accuracy": [],

    "learning_rate": []

}


# ================================================================
# 23. COMPLETE TRAINING LOOP
# ================================================================

print("\n" + "=" * 65)

print("STARTING U-NET TRAINING")

print("=" * 65)


best_val_loss = float("inf")

epochs_without_improvement = 0

best_checkpoint_path = (
    CHECKPOINT_DIR / "unet_best.pth"
)

last_checkpoint_path = (
    CHECKPOINT_DIR / "unet_last.pth"
)

training_start = time.time()


for epoch in range(EPOCHS):

    epoch_start = time.time()

    print("\n" + "=" * 65)

    print(
        f"EPOCH {epoch + 1}/{EPOCHS}"
    )

    print("=" * 65)

    # TRAIN

    train_results = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        scaler
    )

    # VALIDATE

    val_loss, val_metrics = validate_one_epoch(
        model,
        val_loader,
        criterion
    )

    # SCHEDULER

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    # HISTORY

    history["train_loss"].append(
        train_results["loss"]
    )

    history["val_loss"].append(
        val_loss
    )

    history["train_ce"].append(
        train_results["ce"]
    )

    history["train_dice"].append(
        train_results["dice"]
    )

    history["val_miou"].append(
        val_metrics["mean_iou"]
    )

    history["val_mdice"].append(
        val_metrics["mean_dice"]
    )

    history["val_pixel_accuracy"].append(
        val_metrics["pixel_accuracy"]
    )

    history["learning_rate"].append(
        current_lr
    )

    epoch_time = time.time() - epoch_start

    # PRINT RESULTS

    print("\nEPOCH RESULTS")

    print(
        f"Train Loss: {train_results['loss']:.4f}"
    )

    print(
        f"Cross Entropy: {train_results['ce']:.4f}"
    )

    print(
        f"Dice Loss: {train_results['dice']:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation mIoU: {val_metrics['mean_iou']:.4f}"
    )

    print(
        f"Validation Dice: {val_metrics['mean_dice']:.4f}"
    )

    print(
        f"Pixel Accuracy: {val_metrics['pixel_accuracy']:.4f}"
    )

    print(
        f"Learning Rate: {current_lr:.7f}"
    )

    print(
        f"Epoch Time: {epoch_time:.2f} seconds"
    )

    # SAVE LAST CHECKPOINT

    torch.save(

        {

            "epoch": epoch + 1,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "scheduler_state_dict": scheduler.state_dict(),

            "best_val_loss": best_val_loss,

            "history": history

        },

        last_checkpoint_path

    )

    # SAVE BEST MODEL

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(

            {

                "epoch": epoch + 1,

                "model_state_dict": model.state_dict(),

                "best_val_loss": best_val_loss,

                "class_names": CLASS_NAMES,

                "image_size": IMAGE_SIZE,

                "num_classes": NUM_CLASSES

            },

            best_checkpoint_path

        )

        print("\nBest model checkpoint saved!")

    else:

        epochs_without_improvement += 1

        print(
            "\nNo validation loss improvement:",
            epochs_without_improvement
        )

    # EARLY STOPPING

    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print("\nEarly stopping triggered.")

        break


training_time = time.time() - training_start


# ================================================================
# 24. SAVE TRAINING HISTORY
# ================================================================

history_path = OUTPUT_DIR / "training_history.json"

with open(
    history_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


print("\nTraining completed!")

print(
    "Total training time:",
    f"{training_time / 60:.2f} minutes"
)

print("Best validation loss:", best_val_loss)


# ================================================================
# 25. PLOT TRAINING AND VALIDATION LOSS
# ================================================================

epochs_completed = len(history["train_loss"])

epoch_range = range(
    1,
    epochs_completed + 1
)

plt.figure(figsize=(10, 5))

plt.plot(
    epoch_range,
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    epoch_range,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("U-Net Training and Validation Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "loss_curve.png",
    dpi=200
)

plt.show()


# ================================================================
# 26. PLOT VALIDATION METRICS
# ================================================================

plt.figure(figsize=(10, 5))

plt.plot(
    epoch_range,
    history["val_miou"],
    label="Validation mIoU"
)

plt.plot(
    epoch_range,
    history["val_mdice"],
    label="Validation Dice"
)

plt.xlabel("Epoch")

plt.ylabel("Score")

plt.title("Validation Segmentation Metrics")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "validation_metrics.png",
    dpi=200
)

plt.show()


# ================================================================
# 27. LOAD BEST MODEL
# ================================================================

print("\n" + "=" * 65)

print("LOADING BEST U-NET CHECKPOINT")

print("=" * 65)


if not best_checkpoint_path.exists():

    raise FileNotFoundError(
        "Best checkpoint was not created."
    )

checkpoint = torch.load(
    best_checkpoint_path,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)

model.eval()

print(
    "Best epoch:",
    checkpoint["epoch"]
)

print(
    "Best validation loss:",
    checkpoint["best_val_loss"]
)


# ================================================================
# 28. TEST EVALUATION
# ================================================================

print("\n" + "=" * 65)

print("FINAL VALIDATION EVALUATION (also used for checkpoint selection)")

print("=" * 65)


test_loss, test_metrics = validate_one_epoch(
    model,
    test_loader,
    criterion
)


print("\nFINAL VALIDATION RESULTS (same split used for checkpoint selection)")

print(
    f"Validation Loss: {test_loss:.4f}"
)

print(
    f"Validation Pixel Accuracy: "
    f"{test_metrics['pixel_accuracy']:.4f}"
)

print(
    f"Validation Mean IoU: "
    f"{test_metrics['mean_iou']:.4f}"
)

print(
    f"Validation Mean Dice: "
    f"{test_metrics['mean_dice']:.4f}"
)


# ================================================================
# 29. PER-CLASS TEST RESULTS
# ================================================================

print("\nPER-CLASS METRICS")

print("-" * 65)

for class_id, class_name in enumerate(CLASS_NAMES):

    class_iou = test_metrics["class_iou"][class_id]

    class_dice = test_metrics["class_dice"][class_id]

    print(
        f"{class_name:15s} "
        f"| IoU: {class_iou:.4f} "
        f"| Dice: {class_dice:.4f}"
    )


# ================================================================
# 30. SAVE TEST RESULTS
# ================================================================

test_results = {

    "test_loss": float(test_loss),

    "pixel_accuracy": float(
        test_metrics["pixel_accuracy"]
    ),

    "mean_iou": float(
        test_metrics["mean_iou"]
    ),

    "mean_dice": float(
        test_metrics["mean_dice"]
    ),

    "class_iou": {

        CLASS_NAMES[i]: float(
            test_metrics["class_iou"][i]
        )

        for i in range(NUM_CLASSES)

    },

    "class_dice": {

        CLASS_NAMES[i]: float(
            test_metrics["class_dice"][i]
        )

        for i in range(NUM_CLASSES)

    }

}


with open(
    OUTPUT_DIR / "validation_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        test_results,
        f,
        indent=4
    )


# ================================================================
# 31. SEGMENTATION COLOR MAP
# ================================================================

CLASS_COLORS = np.array(

    [

        [0, 0, 0],          # Background - Black

        [34, 197, 94],      # No damage - Green

        [234, 179, 8],      # Minor damage - Yellow

        [249, 115, 22],     # Major damage - Orange

        [239, 68, 68]       # Destroyed - Red

    ],

    dtype=np.uint8

)


def mask_to_color(mask):

    return CLASS_COLORS[mask]


# ================================================================
# 32. VISUALIZE 10 TEST SAMPLES
# ================================================================

print("\n" + "=" * 65)

print("GENERATING 10 TEST PREDICTIONS")

print("=" * 65)


def denormalize_image(tensor):

    image = tensor.detach().cpu().numpy()

    image = np.transpose(
        image,
        (1, 2, 0)
    )

    image = np.clip(
        image * 255.0,
        0,
        255
    ).astype(np.uint8)

    return image


model.eval()

number_of_samples = min(
    10,
    len(test_dataset)
)


for index in range(number_of_samples):

    sample = test_dataset[index]

    post_tensor = sample["image"]

    pre_tensor = sample["pre_image"]

    ground_truth = sample["mask"].numpy()

    image_name = sample["name"]

    # ----------------------------------------------------
    # MODEL PREDICTION
    # ----------------------------------------------------

    input_tensor = post_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP
        ):

            logits = model(input_tensor)

    predicted_mask = torch.argmax(
        logits,
        dim=1
    )[0].cpu().numpy()

    # ----------------------------------------------------
    # CONVERT IMAGES TO DISPLAY FORMAT
    # ----------------------------------------------------

    pre_image = denormalize_image(pre_tensor)

    post_image = denormalize_image(post_tensor)

    ground_truth_color = mask_to_color(
        ground_truth
    )

    predicted_color = mask_to_color(
        predicted_mask
    )

    # ----------------------------------------------------
    # CREATE VISUALIZATION
    # ----------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(22, 6)
    )

    axes[0].imshow(pre_image)

    axes[0].set_title(
        "Pre-disaster Image",
        fontsize=12
    )

    axes[1].imshow(post_image)

    axes[1].set_title(
        "Post-disaster Image",
        fontsize=12
    )

    axes[2].imshow(ground_truth_color)

    axes[2].set_title(
        "Ground Truth Mask",
        fontsize=12
    )

    axes[3].imshow(predicted_color)

    axes[3].set_title(
        "U-Net Prediction",
        fontsize=12
    )

    for ax in axes:

        ax.axis("off")

    fig.suptitle(
        f"Disaster Damage Segmentation - {image_name}",
        fontsize=15
    )

    plt.tight_layout()

    # ----------------------------------------------------
    # SAVE PREDICTION
    # ----------------------------------------------------

    safe_name = Path(image_name).stem

    save_path = (
        PREDICTION_DIR
        / f"{safe_name}_prediction.png"
    )

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    # ----------------------------------------------------
    # DISPLAY CLASS DISTRIBUTION
    # ----------------------------------------------------

    unique_classes, pixel_counts = np.unique(
        predicted_mask,
        return_counts=True
    )

    print("\nImage:", image_name)

    print("Predicted class distribution:")

    for class_id, pixel_count in zip(
        unique_classes,
        pixel_counts
    ):

        print(
            f"  {CLASS_NAMES[int(class_id)]}: "
            f"{pixel_count} pixels"
        )


# ================================================================
# 33. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)

print("STANDARD U-NET TRAINING AND TESTING COMPLETED")

print("=" * 70)

print("\nMODEL INFORMATION")

print("Architecture: Standard U-Net")

print("Input: Post-disaster image only")

print("Image size:", IMAGE_SIZE)

print("Number of classes:", NUM_CLASSES)

print("Device:", DEVICE)

print("\nDATASET")

print("Training images:", len(train_dataset))

print("Validation images:", len(val_dataset))

print("Final validation images:", len(test_dataset))

print("\nFINAL METRICS")

print(f"Validation Loss: {test_loss:.4f}")

print(
    f"Pixel Accuracy: "
    f"{test_metrics['pixel_accuracy']:.4f}"
)

print(
    f"Mean IoU: "
    f"{test_metrics['mean_iou']:.4f}"
)

print(
    f"Mean Dice: "
    f"{test_metrics['mean_dice']:.4f}"
)

print("\nSAVED FILES")

print("Best model:", best_checkpoint_path)

print("Last model:", last_checkpoint_path)

print("Training history:", history_path)

print("Validation results:", OUTPUT_DIR / "validation_results.json")

print("Loss curve:", OUTPUT_DIR / "loss_curve.png")

print("Predictions:", PREDICTION_DIR)

print("\nALL OPERATIONS COMPLETED SUCCESSFULLY!")
