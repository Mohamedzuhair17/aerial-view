import cv2
import torch
import numpy as np
from app.models.unet_damage import UNetDamageSegmenter

IMAGE_PATH = r"C:\Users\irfan\Downloads\post.jpeg"

print("\n========== U-NET DEBUG ==========")

model = UNetDamageSegmenter()

print("Device:", model.device)
print("Input size:", model.size)
print("Model loaded: SUCCESS")

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

print("Original image:", image.shape)

resized = cv2.resize(image, (model.size, model.size))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

tensor = torch.from_numpy(rgb.copy())
tensor = tensor.permute(2, 0, 1).float().div(255)
tensor = tensor.unsqueeze(0).to(model.device)

model.model.eval()

with torch.inference_mode():
    logits = model.model(tensor)
    probabilities = torch.softmax(logits, dim=1)
    prediction = probabilities.argmax(dim=1)[0]

mask = prediction.cpu().numpy().astype(np.uint8)

print("\n========== SEGMENTATION RESULTS ==========")
print("Input tensor:", tensor.shape)
print("Output logits:", logits.shape)
print("Mask shape:", mask.shape)
print("Unique classes:", np.unique(mask))

names = {
    0: "Background",
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed"
}

for class_id, name in names.items():
    count = int((mask == class_id).sum())
    percentage = count / mask.size * 100
    print(f"{name}: {count} pixels ({percentage:.2f}%)")

# Resize prediction to original image dimensions
h, w = image.shape[:2]
mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

colors = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (0, 255, 255),
    3: (0, 165, 255),
    4: (0, 0, 255)
}

colored = np.zeros((h, w, 3), dtype=np.uint8)

for class_id, color in colors.items():
    colored[mask_full == class_id] = color

overlay = cv2.addWeighted(image, 0.6, colored, 0.4, 0)

cv2.imwrite("unet_debug_mask.png", mask_full)
cv2.imwrite("unet_debug_colored.png", colored)
cv2.imwrite("unet_debug_overlay.png", overlay)

print("\n========== SAVED OUTPUTS ==========")
print("unet_debug_mask.png")
print("unet_debug_colored.png")
print("unet_debug_overlay.png")
print("\n========== DEBUG COMPLETED ==========")
