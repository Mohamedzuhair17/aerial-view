import logging
import os
from dataclasses import dataclass

from app.config import settings

log = logging.getLogger("models")


@dataclass
class Models:
    change: object
    seg: object
    status: dict

    @property
    def mode(self) -> str:
        return "ChangeFormer detections + U-Net masks"


def load_models() -> Models:
    for label, path in (("U-Net", settings.unet_weights), ("ChangeFormer", settings.change_weights)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} checkpoint not found: {path}")

    # Import PyTorch before Paddle on Windows to avoid CUDA DLL load conflicts.
    from .unet_damage import UNetDamageSegmenter
    from .changeformer_paddle import PaddleChangeFormerDetector

    log.info("loading U-Net damage map from %s", settings.unet_weights)
    seg = UNetDamageSegmenter()
    log.info("loading ChangeFormer detector from %s", settings.change_weights)
    change = PaddleChangeFormerDetector(checkpoint_path=settings.change_weights)
    return Models(change, seg, {"change": "changeformer", "seg": "unet"})
