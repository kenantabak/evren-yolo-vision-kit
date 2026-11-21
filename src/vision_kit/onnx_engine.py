"""Pure NumPy + ONNX Runtime YOLO26/YOLO11 object detection engine (no Ultralytics dependency)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: Tuple[float, float, float, float]


def letterbox(
    image: Image.Image,
    target_size: Tuple[int, int] = (640, 640),
    fill_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize and pad an image while preserving aspect ratio."""
    orig_w, orig_h = image.size
    tw, th = target_size
    scale = min(tw / orig_w, th / orig_h)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (tw, th), fill_color)
    pad_w = (tw - new_w) / 2.0
    pad_h = (th - new_h) / 2.0
    canvas.paste(resized, (int(round(pad_w - 0.1)), int(round(pad_h - 0.1))))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # (1, 3, H, W)
    return np.ascontiguousarray(arr), scale, (pad_w, pad_h)


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one [x1, y1, x2, y2] box and an array of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = np.maximum(1e-9, area_a + area_b - inter)
    return inter / union


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> List[int]:
    """Greedy NMS returning kept indices sorted by descending confidence."""
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores)
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = compute_iou(boxes[i], boxes[order[1:]])
        remaining = np.where(ious <= iou_threshold)[0]
        order = order[remaining + 1]
    return keep


def decode_yolo_output(
    raw_output: np.ndarray,
    orig_size: Tuple[int, int],
    scale: float,
    pad: Tuple[float, float],
    class_names: Sequence[str],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> List[Detection]:
    """Decode standard (1, 4+C, N) or end2end (1, N, 6) ONNX YOLO output."""
    out = np.squeeze(raw_output, axis=0)
    orig_w, orig_h = orig_size
    pad_w, pad_h = pad

    # Case A: Standard [4 + num_classes, num_anchors] -> transpose to [num_anchors, 4 + num_classes]
    expected_ch = 4 + len(class_names) if class_names else None
    is_standard_ch_first = (
        expected_ch is not None
        and out.ndim == 2
        and out.shape[0] == expected_ch
        and not (out.shape[1] == 6 and np.all(out[:, 5] == np.floor(out[:, 5])) and np.all(out[:, 4] <= 1.0))
    )

    # Case B: End-to-end exported YOLO output [N, 6] -> x1, y1, x2, y2, score, cls
    if out.ndim == 2 and out.shape[1] == 6 and not is_standard_ch_first:
        detections: List[Detection] = []
        for x1, y1, x2, y2, conf, cls_idx in out:
            if conf < conf_threshold:
                continue
            c = int(cls_idx)
            ux1 = float(np.clip((x1 - pad_w) / scale, 0, orig_w))
            uy1 = float(np.clip((y1 - pad_h) / scale, 0, orig_h))
            ux2 = float(np.clip((x2 - pad_w) / scale, 0, orig_w))
            uy2 = float(np.clip((y2 - pad_h) / scale, 0, orig_h))
            name = class_names[c] if 0 <= c < len(class_names) else str(c)
            detections.append(Detection(c, name, float(conf), (ux1, uy1, ux2, uy2)))
        return detections

    if is_standard_ch_first or (out.ndim == 2 and out.shape[0] < out.shape[1]):
        out = out.T  # -> [num_anchors, 4 + num_classes]

    xywh = out[:, :4]
    cls_scores = out[:, 4:]
    cls_ids = np.argmax(cls_scores, axis=1)
    confs = cls_scores[np.arange(len(cls_ids)), cls_ids]

    mask = confs >= conf_threshold
    if not np.any(mask):
        return []

    xywh = xywh[mask]
    confs = confs[mask]
    cls_ids = cls_ids[mask]

    cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = np.clip(((cx - w / 2.0) - pad_w) / scale, 0, orig_w)
    y1 = np.clip(((cy - h / 2.0) - pad_h) / scale, 0, orig_h)
    x2 = np.clip(((cx + w / 2.0) - pad_w) / scale, 0, orig_w)
    y2 = np.clip(((cy + h / 2.0) - pad_h) / scale, 0, orig_h)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    results: List[Detection] = []
    for c in np.unique(cls_ids):
        c_mask = cls_ids == c
        c_boxes = boxes[c_mask]
        c_scores = confs[c_mask]
        kept = non_max_suppression(c_boxes, c_scores, iou_threshold)
        c_int = int(c)
        name = class_names[c_int] if 0 <= c_int < len(class_names) else str(c_int)
        for k in kept:
            b = c_boxes[k]
            results.append(Detection(c_int, name, float(c_scores[k]), (float(b[0]), float(b[1]), float(b[2]), float(b[3]))))
    results.sort(key=lambda d: d.confidence, reverse=True)
    return results


class OnnxDetector:
    def __init__(self, weights_path: Path, class_names: Sequence[str], imgsz: int = 640):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(weights_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        h = shape[2] if len(shape) == 4 and isinstance(shape[2], int) else imgsz
        w = shape[3] if len(shape) == 4 and isinstance(shape[3], int) else imgsz
        self.target_size = (int(w), int(h))
        self.class_names = list(class_names)

    def predict(self, image: Image.Image, conf: float = 0.25, iou: float = 0.45) -> List[Detection]:
        rgb = image.convert("RGB")
        inp, scale, pad = letterbox(rgb, self.target_size)
        raw = self.session.run(None, {self.input_name: inp})[0]
        return decode_yolo_output(raw, rgb.size, scale, pad, self.class_names, conf, iou)
