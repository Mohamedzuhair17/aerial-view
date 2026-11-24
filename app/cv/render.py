import cv2
import numpy as np

# BGR
COLORS = {
    "No Damage": (94, 197, 34), "Minor": (8, 179, 234), "Major": (22, 115, 249),
    "Destroyed": (68, 68, 239), "Unclassified": (175, 163, 156),
}


def render_overlay(image: np.ndarray, buildings: list[dict], highlight: set[str] | None = None, grid: bool = True) -> np.ndarray:
    h, w = image.shape[:2]
    highlight = highlight or set()
    base, hi = image.copy(), image.copy()
    for b in buildings:
        for polygon in b.get("polygons") or [b["polygon"]]:
            pts = np.array(polygon, np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(hi if b["id"] in highlight else base, [pts], COLORS.get(b["damage"], (200, 200, 200)))
    a_base = 0.22 if highlight else 0.38
    out = cv2.addWeighted(base, a_base, image, 1 - a_base, 0)
    if highlight:
        mask = np.zeros((h, w), np.uint8)
        for b in buildings:
            if b["id"] in highlight:
                for polygon in b.get("polygons") or [b["polygon"]]:
                    if len(polygon) >= 3: cv2.fillPoly(mask, [np.array(polygon, np.int32)], 255)
        blended = cv2.addWeighted(hi, 0.7, out, 0.3, 0)
        out = np.where(mask[..., None] > 0, blended, out)
    for b in buildings:
        col = COLORS.get(b["damage"], (200, 200, 200))
        for polygon in b.get("polygons") or [b["polygon"]]:
            pts = np.array(polygon, np.int32)
            if len(pts) < 3: continue
            if b["id"] in highlight: cv2.polylines(out, [pts], True, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.polylines(out, [pts], True, col, 2 if b["id"] in highlight else 1, cv2.LINE_AA)
    if grid:
        for p1, p2 in (((w // 2, 0), (w // 2, h)), ((0, h // 2), (w, h // 2))):
            cv2.line(out, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)
        for txt, org in (("NW", (8, 22)), ("NE", (w - 40, 22)), ("SW", (8, h - 10)), ("SE", (w - 40, h - 10))):
            cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out
