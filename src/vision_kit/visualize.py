"""Clean PIL bounding-box renderer for Detection objects."""
from __future__ import annotations

from typing import Sequence
from PIL import Image, ImageDraw
from .onnx_engine import Detection

PALETTE = ["#38bdf8", "#34d399", "#f472b6", "#fbbf24", "#a78bfa", "#fb7185", "#22d3ee"]


def draw_detections(image: Image.Image, detections: Sequence[Detection]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for det in detections:
        color = PALETTE[det.class_id % len(PALETTE)]
        x1, y1, x2, y2 = det.xyxy
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{det.class_name} {det.confidence:.2f}"
        draw.rectangle([x1, max(0, y1 - 18), x1 + len(label) * 7 + 8, y1], fill=color)
        draw.text((x1 + 4, max(0, y1 - 15)), label, fill="#0f172a")
    return canvas
