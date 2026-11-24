# Training the aerial models

The extracted xBD-style data is expected at `dataset v1/`. Training defaults to
the augmented training split and the supplied validation split. Do not combine
`train` and `train_aug`: `train_aug` already contains the original scenes as well
as augmented variants. Keep validation pairs out of training.

Run these commands from the repository root in the `chaiathon` Conda environment.
The installed environment should include PyTorch, TorchVision, OpenCV and NumPy;
the current environment has CUDA-enabled PyTorch/TorchVision. Pretrained ImageNet
or COCO weights are used by default and may be downloaded on first run. Pass
`--no-pretrained` to train from random initialization without downloading weights.

## 1. Train building instance segmentation

```powershell
conda run -n chaiathon python scripts/train_building_segmenter.py `
  --dataset "dataset v1" --device cuda --epochs 10 --batch-size 1
```

This trains a two-class Mask R-CNN on post-disaster images and polygon-derived
building masks. Inputs and target masks are jointly downsampled to 512px before
TorchVision's standard resize, keeping the effective object scale consistent with
serving while bounding memory for dense scenes. Empty-annotation scenes are used
as background examples. It writes `weights/maskrcnn_buildings.pth`, which the
current `RealSegmenter` wrapper can load.

## 2. Train damage classification

```powershell
conda run -n chaiathon python scripts/train_damage_classifier.py `
  --dataset "dataset v1" --device cuda --epochs 12 --batch-size 64
```

The classifier uses paired before/after crops, six channels in before-then-after
order, and the backend's ImageNet normalization. `Unclassified` polygons are kept
for segmentation but omitted from classifier loss. The loss uses softened
inverse-frequency class weights. The script selects a checkpoint by validation
macro-F1, fits one temperature on validation logits, saves the compatible
checkpoint at `weights/effnet_b0_damage.pth`, and writes validation logits beside
it for audit/recalibration.

The 66 validation pairs contain few examples of some damage classes. Treat these
metrics as an initial check, not a final estimate for unseen disasters. Keep a
separate event-level test set for final reporting if more labeled scenes become
available.

## 3. Use the checkpoints

The default output paths match the backend settings. Set `MODEL_MODE=real` and
`DEVICE=cuda` in `.env` on a GPU host (or `DEVICE=cpu` for CPU inference), then
start the worker/API. The training scripts do not train ChangeFormer because the
dataset has no pixel-level change masks.

Both scripts accept `--help` for options such as split names, output path, batch
size, epoch count, workers, and learning rate. Reduce classifier batch size or
keep segmentation at batch size 1 if GPU memory is limited.
