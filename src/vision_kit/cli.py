"""Command-line interface for evren-yolo-vision-kit."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from PIL import Image

from .hub import MODEL_REGISTRY, download_model_bundle
from .onnx_engine import OnnxDetector
from .visualize import draw_detections


def main() -> None:
    parser = argparse.ArgumentParser(prog="evren-vision", description="Run open YOLO26/YOLO11 weights from Hugging Face.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-models", help="List available pretrained models in the registry.")

    p_pred = sub.add_parser("predict", help="Run inference on an image.")
    p_pred.add_argument("--model", required=True, help="Registry alias (e.g. ppe-pro) or HF repo_id.")
    p_pred.add_argument("--source", required=True, help="Path to input image.")
    p_pred.add_argument("--conf", type=float, default=0.25)
    p_pred.add_argument("--iou", type=float, default=0.45)
    p_pred.add_argument("--save", default=None, help="Optional output image path.")

    args = parser.parse_args()
    if args.command == "list-models":
        for k, v in MODEL_REGISTRY.items():
            print(f"{k:18s} | {v['arch']:8s} | mAP@50={v['map50']:5.2f}% | https://huggingface.co/{v['repo_id']}")
        return

    if args.command == "predict":
        bundle = download_model_bundle(args.model, backend="onnx")
        det = OnnxDetector(bundle.weights_path, bundle.classes, imgsz=bundle.imgsz)
        img = Image.open(args.source)
        preds = det.predict(img, conf=args.conf, iou=args.iou)
        print(json.dumps([asdict(p) for p in preds], ensure_ascii=False, indent=2))
        if args.save:
            out_img = draw_detections(img, preds)
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            out_img.save(args.save)
            print(f"Saved annotated output to {args.save}")


if __name__ == "__main__":
    main()
