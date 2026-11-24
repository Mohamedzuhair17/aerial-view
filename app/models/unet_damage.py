"""Adapter for the trained five-class U-Net damage-map checkpoint."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

from app.config import settings
from .base import BuildingInstance


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = _ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        features = self.conv(x)
        return features, self.pool(features)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv = _ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class _UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 5):
        super().__init__()
        self.encoder1 = _EncoderBlock(in_channels, 64)
        self.encoder2 = _EncoderBlock(64, 128)
        self.encoder3 = _EncoderBlock(128, 256)
        self.encoder4 = _EncoderBlock(256, 512)
        self.bottleneck = _ConvBlock(512, 1024)
        self.decoder4 = _DecoderBlock(1024, 512, 512)
        self.decoder3 = _DecoderBlock(512, 256, 256)
        self.decoder2 = _DecoderBlock(256, 128, 128)
        self.decoder1 = _DecoderBlock(128, 64, 64)
        self.segmentation_head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1, p1 = self.encoder1(x)
        e2, p2 = self.encoder2(p1)
        e3, p3 = self.encoder3(p2)
        e4, p4 = self.encoder4(p3)
        b = self.bottleneck(p4)
        d4 = self.decoder4(b, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        return self.segmentation_head(d1)


class UNetDamageSegmenter:
    """Return connected components from the U-Net's semantic damage map.

    This adapter cannot separate touching same-class buildings; the model predicts
    semantic classes, not building instances.
    """

    DAMAGE = {1: "No Damage", 2: "Minor", 3: "Major", 4: "Destroyed"}

    def __init__(self):
        path = settings.unet_weights
        if settings.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("DEVICE requests CUDA, but torch.cuda.is_available() is False")
        self.device = torch.device(settings.device)
        self.size = int(settings.unet_input_size)
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model = _UNet(in_channels=3, num_classes=5).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, after: np.ndarray) -> list[BuildingInstance]:
        h, w = after.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(after, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0).to(self.device)
        probs = self.model(tensor).softmax(dim=1)[0]
        class_map = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)
        conf_map = probs.cpu().numpy()
        class_map = cv2.resize(class_map, (w, h), interpolation=cv2.INTER_NEAREST)
        conf_map = np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in conf_map])

        instances: list[BuildingInstance] = []
        for class_id, damage in self.DAMAGE.items():
            count, labels, stats, _ = cv2.connectedComponentsWithStats((class_map == class_id).astype(np.uint8), 8)
            for component in range(1, count):
                x, y, bw, bh, area = (int(v) for v in stats[component])
                if area < settings.min_building_area:
                    continue
                mask = labels == component
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polygons = []
                for contour in contours:
                    if cv2.contourArea(contour) < 3:
                        continue
                    contour = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                    points = contour.reshape(-1, 2).astype(int).tolist()
                    if len(points) >= 3:
                        polygons.append(points)
                if not polygons:
                    continue
                confidence = float(conf_map[class_id][mask].mean())
                instance = BuildingInstance(mask, (x, y, x + bw, y + bh), polygons[0], polygons)
                instance.damage = damage
                instance.confidence = confidence
                instances.append(instance)
        return instances

