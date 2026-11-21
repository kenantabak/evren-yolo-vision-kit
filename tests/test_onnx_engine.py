import numpy as np
from PIL import Image
from vision_kit.onnx_engine import compute_iou, decode_yolo_output, letterbox, non_max_suppression


def test_letterbox_preserves_aspect_ratio():
    img = Image.new("RGB", (1280, 720), (10, 20, 30))
    arr, scale, (pad_w, pad_h) = letterbox(img, (640, 640))
    assert arr.shape == (1, 3, 640, 640)
    assert abs(scale - 0.5) < 1e-6
    assert abs(pad_h - 140.0) < 1e-5
    assert abs(pad_w - 0.0) < 1e-5


def test_nms_suppresses_overlapping_box():
    boxes = np.array([
        [10.0, 10.0, 100.0, 100.0],
        [12.0, 12.0, 102.0, 102.0],
        [300.0, 300.0, 400.0, 400.0],
    ])
    scores = np.array([0.92, 0.85, 0.78])
    kept = non_max_suppression(boxes, scores, iou_threshold=0.45)
    assert kept == [0, 2]


def test_decode_standard_yolo_tensor():
    # (1, 6, 2) -> 4 box coords + 2 classes, 2 anchors
    raw = np.zeros((1, 6, 2), dtype=np.float32)
    raw[0, :, 0] = [320.0, 320.0, 100.0, 100.0, 0.90, 0.05]
    raw[0, :, 1] = [100.0, 100.0, 40.0, 40.0, 0.02, 0.10]  # below conf 0.25
    dets = decode_yolo_output(raw, (640, 640), 1.0, (0.0, 0.0), ["baret", "yelek"], conf_threshold=0.25)
    assert len(dets) == 1
    assert dets[0].class_name == "baret"
    assert abs(dets[0].confidence - 0.90) < 1e-5
