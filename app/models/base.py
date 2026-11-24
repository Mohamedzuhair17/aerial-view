"""Model contracts. Anything that implements these can be dropped into the pipeline."""
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

CLASSES = ["No Damage", "Minor", "Major", "Destroyed", "Unclassified"]


@dataclass
class BuildingInstance:
    mask: np.ndarray  # HxW bool
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    polygon: list[list[int]]
    polygons: list[list[list[int]]] = field(default_factory=list)
    # Semantic-damage models can supply a class directly with each component.
    damage: str | None = None
    confidence: float = 0.0


class ChangeDetector(Protocol):
    def predict(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """BGR uint8 images (HxWx3) -> HxW bool change mask."""
        ...


class BuildingSegmenter(Protocol):
    def predict(self, after: np.ndarray) -> list[BuildingInstance]:
        """BGR uint8 post-disaster image -> one instance per building."""
        ...


class DamageClassifier(Protocol):
    def predict(self, before_crop: np.ndarray, after_crop: np.ndarray) -> tuple[str, float]:
        """BGR crops of one building -> (damage class, confidence)."""
        ...
