import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.vision.ops import roi_align, nms


# ============================================================
# 1. CONFIGURATION
# ============================================================

NUM_CLASSES = 4

CLASS_NAMES = {
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed"
}

CHANNELS = [32, 64, 160, 256]


# ============================================================
# 2. CONVOLUTION BLOCK
# ============================================================

class ConvBNReLU(nn.Layer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias_attr=False
            ),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# 3. OVERLAPPING PATCH EMBEDDING
# ============================================================

class OverlapPatchEmbed(nn.Layer):

    def __init__(self, in_channels, embed_dim,
                 kernel_size, stride):

        super().__init__()

        self.proj = nn.Conv2D(
            in_channels,
            embed_dim,
            kernel_size,
            stride,
            padding=kernel_size // 2
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):

        x = self.proj(x)

        B, C, H, W = x.shape

        x = x.flatten(2).transpose([0, 2, 1])
        x = self.norm(x)

        x = x.transpose([0, 2, 1]).reshape([B, C, H, W])

        return x


# ============================================================
# 4. SPATIAL REDUCTION ATTENTION
# ============================================================

class SpatialReductionAttention(nn.Layer):

    def __init__(self, dim, num_heads=4, sr_ratio=2):

        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio

        if sr_ratio > 1:

            self.sr = nn.Conv2D(
                dim,
                dim,
                kernel_size=sr_ratio,
                stride=sr_ratio
            )

            self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        B, C, H, W = x.shape

        tokens = x.flatten(2).transpose([0, 2, 1])
        N = tokens.shape[1]

        q = self.q(tokens)

        q = q.reshape([
            B, N, self.num_heads, self.head_dim
        ]).transpose([0, 2, 1, 3])

        if self.sr_ratio > 1:

            reduced = self.sr(x)
            reduced = reduced.flatten(2).transpose([0, 2, 1])
            reduced = self.norm(reduced)

        else:
            reduced = tokens

        kv = self.kv(reduced)

        kv = kv.reshape([
            B, -1, 2, self.num_heads, self.head_dim
        ]).transpose([2, 0, 3, 1, 4])

        k = kv[0]
        v = kv[1]

        attention = paddle.matmul(
            q, k, transpose_y=True
        ) * self.scale

        attention = F.softmax(attention, axis=-1)

        out = paddle.matmul(attention, v)

        out = out.transpose([0, 2, 1, 3]).reshape([
            B, N, C
        ])

        out = self.proj(out)

        out = out.transpose([0, 2, 1]).reshape([
            B, C, H, W
        ])

        return out


# ============================================================
# 5. MIX-FFN
# ============================================================

class MixFFN(nn.Layer):

    def __init__(self, dim, expansion=4):

        super().__init__()

        hidden_dim = dim * expansion

        self.fc1 = nn.Conv2D(dim, hidden_dim, 1)

        self.dwconv = nn.Conv2D(
            hidden_dim,
            hidden_dim,
            3,
            padding=1,
            groups=hidden_dim
        )

        self.fc2 = nn.Conv2D(hidden_dim, dim, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x


# ============================================================
# 6. TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Layer):

    def __init__(self, dim, num_heads, sr_ratio=2):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attention = SpatialReductionAttention(
            dim, num_heads, sr_ratio
        )

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MixFFN(dim)

    def forward(self, x):

        B, C, H, W = x.shape

        residual = x

        tokens = x.flatten(2).transpose([0, 2, 1])
        tokens = self.norm1(tokens)

        x = tokens.transpose([0, 2, 1]).reshape([
            B, C, H, W
        ])

        x = residual + self.attention(x)

        residual = x

        tokens = x.flatten(2).transpose([0, 2, 1])
        tokens = self.norm2(tokens)

        x = tokens.transpose([0, 2, 1]).reshape([
            B, C, H, W
        ])

        x = residual + self.mlp(x)

        return x


# ============================================================
# 7. HIERARCHICAL MIT ENCODER
# ============================================================

class MiTEncoder(nn.Layer):

    def __init__(self):

        super().__init__()

        dims = CHANNELS

        self.patch1 = OverlapPatchEmbed(3, dims[0], 7, 4)
        self.patch2 = OverlapPatchEmbed(dims[0], dims[1], 3, 2)
        self.patch3 = OverlapPatchEmbed(dims[1], dims[2], 3, 2)
        self.patch4 = OverlapPatchEmbed(dims[2], dims[3], 3, 2)

        self.block1 = nn.LayerList([
            TransformerBlock(dims[0], 1, 4),
            TransformerBlock(dims[0], 1, 4)
        ])

        self.block2 = nn.LayerList([
            TransformerBlock(dims[1], 2, 2),
            TransformerBlock(dims[1], 2, 2)
        ])

        self.block3 = nn.LayerList([
            TransformerBlock(dims[2], 5, 2),
            TransformerBlock(dims[2], 5, 2)
        ])

        self.block4 = nn.LayerList([
            TransformerBlock(dims[3], 8, 1),
            TransformerBlock(dims[3], 8, 1)
        ])

    def forward(self, x):

        features = []

        x = self.patch1(x)
        for block in self.block1:
            x = block(x)
        features.append(x)

        x = self.patch2(x)
        for block in self.block2:
            x = block(x)
        features.append(x)

        x = self.patch3(x)
        for block in self.block3:
            x = block(x)
        features.append(x)

        x = self.patch4(x)
        for block in self.block4:
            x = block(x)
        features.append(x)

        return features


# ============================================================
# 8. SIAMESE CHANGEFORMER
# ============================================================

class SiameseChangeFormer(nn.Layer):

    def __init__(self):

        super().__init__()

        # Shared encoder weights for both images.
        self.encoder = MiTEncoder()

    def forward(self, pre_image, post_image):

        pre_features = self.encoder(pre_image)
        post_features = self.encoder(post_image)

        return pre_features, post_features


# ============================================================
# 9. TEMPORAL FEATURE FUSION
# ============================================================

class TemporalFeatureFusion(nn.Layer):

    def __init__(self, channels):

        super().__init__()

        self.fusion_layers = nn.LayerList()

        for c in channels:

            self.fusion_layers.append(
                nn.Sequential(
                    nn.Conv2D(c * 3, c, 1, bias_attr=False),
                    nn.BatchNorm2D(c),
                    nn.ReLU(),

                    nn.Conv2D(c, c, 3, padding=1, bias_attr=False),
                    nn.BatchNorm2D(c),
                    nn.ReLU()
                )
            )

    def forward(self, pre_features, post_features):

        fused_features = []

        for i in range(len(pre_features)):

            difference = paddle.abs(
                post_features[i] - pre_features[i]
            )

            combined = paddle.concat([
                pre_features[i],
                post_features[i],
                difference
            ], axis=1)

            fused = self.fusion_layers[i](combined)

            fused_features.append(fused)

        return fused_features


# ============================================================
# 10. FEATURE PYRAMID NETWORK
# ============================================================

class FeaturePyramidNetwork(nn.Layer):

    def __init__(self, channels, out_channels=256):

        super().__init__()

        self.lateral = nn.LayerList([
            nn.Conv2D(c, out_channels, 1)
            for c in channels
        ])

        self.output_convs = nn.LayerList([
            ConvBNReLU(out_channels, out_channels, 3, 1, 1)
            for _ in channels
        ])

    def forward(self, features):

        laterals = [
            self.lateral[i](features[i])
            for i in range(len(features))
        ]

        for i in range(len(laterals) - 1, 0, -1):

            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="nearest"
            )

            laterals[i - 1] = laterals[i - 1] + upsampled

        outputs = []

        for i in range(len(laterals)):
            outputs.append(self.output_convs[i](laterals[i]))

        return outputs


# ============================================================
# 11. ANCHOR GENERATION
# ============================================================

def generate_anchors(
    feature_height,
    feature_width,
    stride=4,
    scales=(32, 64, 128),
    ratios=(0.5, 1.0, 2.0)
):

    y = (paddle.arange(feature_height, dtype="float32") + 0.5) * stride
    x = (paddle.arange(feature_width, dtype="float32") + 0.5) * stride

    yy, xx = paddle.meshgrid(y, x)

    centers = paddle.stack([xx, yy], axis=-1)

    anchors = []

    for scale in scales:

        for ratio in ratios:

            width = scale / math.sqrt(ratio)
            height = scale * math.sqrt(ratio)

            x1 = centers[:, :, 0] - width / 2
            y1 = centers[:, :, 1] - height / 2

            x2 = centers[:, :, 0] + width / 2
            y2 = centers[:, :, 1] + height / 2

            anchors.append(
                paddle.stack([x1, y1, x2, y2], axis=-1)
            )

    anchors = paddle.stack(anchors, axis=2)

    return anchors.reshape([-1, 4])


# ============================================================
# 12. REGION PROPOSAL NETWORK
# ============================================================

class RegionProposalNetwork(nn.Layer):

    def __init__(self, in_channels=256, num_anchors=9):

        super().__init__()

        self.conv = ConvBNReLU(
            in_channels, in_channels, 3, 1, 1
        )

        self.objectness = nn.Conv2D(
            in_channels, num_anchors, 1
        )

        self.bbox_regression = nn.Conv2D(
            in_channels, num_anchors * 4, 1
        )

    def forward(self, feature):

        x = self.conv(feature)

        objectness = self.objectness(x)
        bbox_deltas = self.bbox_regression(x)

        B, A, H, W = objectness.shape

        objectness = objectness.transpose([
            0, 2, 3, 1
        ]).reshape([B, -1])

        bbox_deltas = bbox_deltas.reshape([
            B, A, 4, H, W
        ]).transpose([
            0, 3, 4, 1, 2
        ]).reshape([B, -1, 4])

        return objectness, bbox_deltas


# ============================================================
# 13. BOUNDING BOX OPERATIONS
# ============================================================

def decode_boxes(anchors, deltas):

    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]

    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]

    dw = paddle.clip(deltas[:, 2], -4.0, 4.0)
    dh = paddle.clip(deltas[:, 3], -4.0, 4.0)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y

    pred_w = paddle.exp(dw) * widths
    pred_h = paddle.exp(dh) * heights

    x1 = pred_ctr_x - pred_w / 2
    y1 = pred_ctr_y - pred_h / 2

    x2 = pred_ctr_x + pred_w / 2
    y2 = pred_ctr_y + pred_h / 2

    return paddle.stack([x1, y1, x2, y2], axis=1)


def clip_boxes(boxes, height, width):

    x1 = paddle.clip(boxes[:, 0], 0, width - 1)
    y1 = paddle.clip(boxes[:, 1], 0, height - 1)

    x2 = paddle.clip(boxes[:, 2], 0, width - 1)
    y2 = paddle.clip(boxes[:, 3], 0, height - 1)

    return paddle.stack([x1, y1, x2, y2], axis=1)


# ============================================================
# 14. PROPOSAL GENERATION
# ============================================================

def generate_proposals(
    objectness,
    bbox_deltas,
    anchors,
    image_height,
    image_width,
    pre_nms_topk=1000,
    post_nms_topk=200,
    nms_threshold=0.7
):

    proposals_batch = []

    for b in range(objectness.shape[0]):

        scores = F.sigmoid(objectness[b])
        deltas = bbox_deltas[b]

        topk = min(pre_nms_topk, scores.shape[0])

        top_scores, top_indices = paddle.topk(scores, k=topk)

        selected_anchors = paddle.gather(anchors, top_indices)
        selected_deltas = paddle.gather(deltas, top_indices)

        boxes = decode_boxes(selected_anchors, selected_deltas)
        boxes = clip_boxes(boxes, image_height, image_width)

        valid = paddle.logical_and(
            boxes[:, 2] > boxes[:, 0] + 1,
            boxes[:, 3] > boxes[:, 1] + 1
        )

        valid_indices = paddle.nonzero(valid).flatten()

        boxes = paddle.gather(boxes, valid_indices)
        top_scores = paddle.gather(top_scores, valid_indices)

        if boxes.shape[0] == 0:
            proposals_batch.append(
                paddle.zeros([0, 4], dtype="float32")
            )
            continue

        keep = nms(
            boxes,
            iou_threshold=nms_threshold,
            scores=top_scores
        )

        keep = keep[:post_nms_topk]

        proposals_batch.append(paddle.gather(boxes, keep))

    return proposals_batch


# ============================================================
# 15. ROI ALIGN
# ============================================================

class MultiScaleRoIAlign(nn.Layer):

    def __init__(self, output_size=7):

        super().__init__()

        self.output_size = output_size
        self.spatial_scale = 0.25

    def forward(self, feature, proposals):

        pooled_features = []

        for batch_index, boxes in enumerate(proposals):

            if boxes.shape[0] == 0:
                continue

            feature_map = feature[
                batch_index:batch_index + 1
            ]

            boxes_num = paddle.to_tensor(
                [boxes.shape[0]], dtype="int32"
            )

            pooled = roi_align(
                feature_map,
                boxes,
                boxes_num,
                output_size=self.output_size,
                spatial_scale=self.spatial_scale,
                sampling_ratio=2,
                aligned=True
            )

            pooled_features.append(pooled)

        if len(pooled_features) == 0:

            return paddle.zeros(
                [0, feature.shape[1],
                 self.output_size, self.output_size],
                dtype="float32"
            )

        return paddle.concat(pooled_features, axis=0)


# ============================================================
# 16. DAMAGE CLASSIFICATION HEAD
# ============================================================

class DamageClassificationHead(nn.Layer):

    def __init__(self, in_channels=256, num_classes=4):

        super().__init__()

        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)

        # Background + four damage classes.
        self.classifier = nn.Linear(512, num_classes + 1)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.flatten(1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        return self.classifier(x)


# ============================================================
# 17. BOUNDING BOX REGRESSION HEAD
# ============================================================

class BoundingBoxRegressionHead(nn.Layer):

    def __init__(self, in_channels=256, num_classes=4):

        super().__init__()

        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.regressor = nn.Linear(512, num_classes * 4)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.flatten(1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        bbox_deltas = self.regressor(x)

        return bbox_deltas.reshape([-1, NUM_CLASSES, 4])


# ============================================================
# 18. COMPLETE CHANGEFORMER DAMAGE DETECTOR
# ============================================================

class ChangeFormerDamageDetector(nn.Layer):

    def __init__(self, num_classes=4):

        super().__init__()

        self.num_classes = num_classes

        self.siamese_encoder = SiameseChangeFormer()

        self.temporal_fusion = TemporalFeatureFusion(CHANNELS)

        self.fpn = FeaturePyramidNetwork(
            CHANNELS, out_channels=256
        )

        self.rpn = RegionProposalNetwork(
            in_channels=256,
            num_anchors=9
        )

        self.roi_align = MultiScaleRoIAlign(output_size=7)

        self.classification_head = DamageClassificationHead(
            in_channels=256,
            num_classes=num_classes
        )

        self.bbox_head = BoundingBoxRegressionHead(
            in_channels=256,
            num_classes=num_classes
        )

    def forward(self, pre_image, post_image):

        B, _, H, W = post_image.shape

        # 1. Shared Siamese feature extraction.
        pre_features, post_features = self.siamese_encoder(
            pre_image, post_image
        )

        # 2. Temporal feature fusion.
        fused_features = self.temporal_fusion(
            pre_features, post_features
        )

        # 3. Multi-scale feature extraction.
        pyramid_features = self.fpn(fused_features)

        # P2: highest-resolution feature map.
        p2 = pyramid_features[0]

        # 4. Region proposals.
        objectness, rpn_deltas = self.rpn(p2)

        anchors = generate_anchors(
            feature_height=p2.shape[2],
            feature_width=p2.shape[3],
            stride=4
        )

        # 5. Candidate building boxes.
        proposals = generate_proposals(
            objectness,
            rpn_deltas,
            anchors,
            H,
            W
        )

        # 6. RoI feature extraction.
        roi_features = self.roi_align(p2, proposals)

        # 7. Damage classification.
        class_logits = self.classification_head(roi_features)

        class_probabilities = F.softmax(
            class_logits, axis=1
        )

        # 8. Bounding-box regression.
        bbox_deltas = self.bbox_head(roi_features)

        return {
            "class_logits": class_logits,
            "class_probabilities": class_probabilities,
            "bbox_deltas": bbox_deltas,
            "proposals": proposals,
            "rpn_objectness": objectness,
            "rpn_bbox_deltas": rpn_deltas,
            "anchors": anchors,
            "pyramid_features": pyramid_features
        }



# Inference adapter used by the Aerial backend.
import logging
import pickle

import cv2
import numpy as np
from app.config import settings
from app.models.base import BuildingInstance

logger = logging.getLogger("models.changeformer")


class PaddleChangeFormerDetector:
    """Load the supplied Paddle checkpoint and return damage-labeled boxes."""

    def __init__(self, checkpoint_path: str | None = None):
        self.path = checkpoint_path or settings.change_weights
        if settings.device.startswith("cuda"):
            if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
                raise RuntimeError("DEVICE=cuda requires PaddlePaddle GPU with a visible CUDA device")
            paddle.set_device("gpu:0")
            self.device = "gpu:0"
        else:
            paddle.set_device("cpu")
            self.device = "cpu"
        self.size = int(settings.change_input_size)
        if self.size % 32:
            raise ValueError("CHANGE_INPUT_SIZE must be divisible by 32")
        self.score_threshold = float(settings.change_score_thresh)
        self.nms_threshold = float(settings.change_nms_iou)
        self.model = ChangeFormerDamageDetector(num_classes=4)
        try:
            state = paddle.load(self.path)
        except Exception as paddle_error:
            # These project-generated .pdparams files are raw Python pickles of
            # NumPy arrays rather than Paddle's wrapped serialization container.
            try:
                with open(self.path, "rb") as handle:
                    state = pickle.load(handle, encoding="latin1")
            except Exception:
                raise RuntimeError(f"Could not read ChangeFormer checkpoint {self.path}: {paddle_error}") from paddle_error
        if not isinstance(state, dict):
            raise ValueError("ChangeFormer checkpoint must contain a parameter dictionary")
        state.pop("StructuredToParameterName@@", None)
        state = {key: paddle.to_tensor(value) if isinstance(value, np.ndarray) else value
                 for key, value in state.items()}
        self.model.set_state_dict(state)
        self.model.eval()

    @staticmethod
    def _decode_for_class(boxes, deltas, height, width):
        decoded = decode_boxes(boxes, deltas)
        return clip_boxes(decoded, height, width)

    @paddle.no_grad()
    def predict_instances(self, before: np.ndarray, after: np.ndarray) -> list[BuildingInstance]:
        h, w = after.shape[:2]
        target = (self.size, self.size)
        pre_rgb = cv2.cvtColor(cv2.resize(before, target, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        post_rgb = cv2.cvtColor(cv2.resize(after, target, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        pre = paddle.to_tensor(pre_rgb.transpose(2, 0, 1)[None].astype("float32") / 255.0)
        post = paddle.to_tensor(post_rgb.transpose(2, 0, 1)[None].astype("float32") / 255.0)
        output = self.model(pre, post)
        proposals = output["proposals"][0]
        class_probs = output["class_probabilities"]
        box_deltas = output["bbox_deltas"]
        if proposals.shape[0] == 0:
            return []

        detections: list[BuildingInstance] = []
        scale_x, scale_y = w / self.size, h / self.size
        for class_id, label in CLASS_NAMES.items():
            scores = class_probs[:, class_id]
            deltas = box_deltas[:, class_id - 1, :]
            boxes = self._decode_for_class(proposals, deltas, self.size, self.size)
            valid = paddle.logical_and(
                paddle.logical_and(boxes[:, 2] > boxes[:, 0] + 1, boxes[:, 3] > boxes[:, 1] + 1),
                scores >= self.score_threshold,
            )
            indices = paddle.nonzero(valid).flatten()
            if indices.shape[0] == 0:
                continue
            boxes = paddle.gather(boxes, indices)
            scores = paddle.gather(scores, indices)
            keep = nms(boxes, iou_threshold=self.nms_threshold, scores=scores)
            boxes = paddle.gather(boxes, keep)
            scores = paddle.gather(scores, keep)
            boxes_np = boxes.numpy()
            scores_np = scores.numpy()
            for box, confidence in zip(boxes_np, scores_np):
                x1 = max(0, min(w - 1, int(round(box[0] * scale_x))))
                y1 = max(0, min(h - 1, int(round(box[1] * scale_y))))
                x2 = max(x1 + 1, min(w, int(round(box[2] * scale_x))))
                y2 = max(y1 + 1, min(h, int(round(box[3] * scale_y))))
                mask = np.zeros((h, w), dtype=bool)
                mask[y1:y2, x1:x2] = True
                polygon = [[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]]
                detections.append(BuildingInstance(
                    mask, (x1, y1, x2, y2), polygon, [polygon], label, float(confidence)
                ))
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections
