"""Model registry and weight downloader backed by Hugging Face Hub."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download

HF_OWNER = "kenantabak"

MODEL_REGISTRY: Dict[str, dict] = {
    "peanut-mold": {
        "repo_id": f"{HF_OWNER}/yolo26s-peanut-mold-quality-detection-tr",
        "arch": "yolo26s",
        "imgsz": 640,
        "map50": 99.50,
    },
    "indoor-furniture": {
        "repo_id": f"{HF_OWNER}/yolo26s-indoor-furniture-detection-tr",
        "arch": "yolo26s",
        "imgsz": 640,
        "map50": 99.49,
    },
    "traffic-nav": {
        "repo_id": f"{HF_OWNER}/yolo26s-traffic-sign-navigation-tr",
        "arch": "yolo26s",
        "imgsz": 768,
        "map50": 99.26,
    },
    "pcb-defect": {
        "repo_id": f"{HF_OWNER}/yolo11s-pcb-defect-inspection-6class-tr",
        "arch": "yolo11s",
        "imgsz": 640,
        "map50": 99.15,
    },
    "retail-produce": {
        "repo_id": f"{HF_OWNER}/yolo26s-retail-produce-detection-22class-tr",
        "arch": "yolo26s",
        "imgsz": 640,
        "map50": 98.94,
    },
    "ppe-pro": {
        "repo_id": f"{HF_OWNER}/yolo26x-industrial-ppe-safety-pro-tr",
        "arch": "yolo26x",
        "imgsz": 640,
        "map50": 98.48,
    },
    "hardhat-color": {
        "repo_id": f"{HF_OWNER}/yolo26s-color-coded-hardhat-safety-tr",
        "arch": "yolo26s",
        "imgsz": 640,
        "map50": 98.19,
    },
    "airport-apron": {
        "repo_id": f"{HF_OWNER}/yolo11s-airport-apron-ground-vehicles-tr",
        "arch": "yolo11s",
        "imgsz": 768,
        "map50": 98.03,
    },
    "thermal-sar": {
        "repo_id": f"{HF_OWNER}/yolo26s-thermal-human-canine-detection-tr",
        "arch": "yolo26s",
        "imgsz": 640,
        "map50": 97.92,
    },
    "circuit-symbols": {
        "repo_id": f"{HF_OWNER}/yolo26s-electrical-schematic-components-tr",
        "arch": "yolo26s",
        "imgsz": 640,
        "map50": 97.87,
    },
    "highway-signs": {
        "repo_id": f"{HF_OWNER}/yolo26s-highway-traffic-signs-23class-tr",
        "arch": "yolo26s",
        "imgsz": 768,
        "map50": 95.53,
    },
    "speed-signs": {
        "repo_id": f"{HF_OWNER}/yolo26s-autonomous-driving-speed-signs-tr",
        "arch": "yolo26s",
        "imgsz": 768,
        "map50": 94.95,
    },
    "wildfire-smoke": {
        "repo_id": f"{HF_OWNER}/yolo26s-wildfire-smoke-early-detection-tr",
        "arch": "yolo26s",
        "imgsz": 640,
        "map50": 93.96,
    },
    "heavy-vehicles": {
        "repo_id": f"{HF_OWNER}/yolo26s-highway-heavy-vehicle-truck-tr",
        "arch": "yolo26s",
        "imgsz": 640,
        "map50": 93.78,
    },
    "uav-vehicles": {
        "repo_id": f"{HF_OWNER}/yolo26s-uav-aerial-vehicle-detection-tr",
        "arch": "yolo26s",
        "imgsz": 640,
        "map50": 82.21,
    },
    "dental-xray": {
        "repo_id": f"{HF_OWNER}/yolo26s-dental-xray-pathology-detection-tr",
        "arch": "yolo26s",
        "imgsz": 640,
        "map50": 79.80,
    },
}


@dataclass
class ModelBundle:
    repo_id: str
    weights_path: Path
    classes: List[str]
    imgsz: int


def resolve_repo_id(model_key_or_repo: str) -> str:
    if model_key_or_repo in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_key_or_repo]["repo_id"]
    if "/" in model_key_or_repo:
        return model_key_or_repo
    return f"{HF_OWNER}/{model_key_or_repo}"


def download_model_bundle(
    model_key_or_repo: str,
    backend: str = "onnx",
    cache_dir: Optional[str] = None,
) -> ModelBundle:
    repo_id = resolve_repo_id(model_key_or_repo)
    filename = "best.onnx" if backend == "onnx" else "best.pt"
    weights = Path(hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir))
    cfg_path = Path(hf_hub_download(repo_id=repo_id, filename="evren_model_config.json", cache_dir=cache_dir))
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    imgsz = MODEL_REGISTRY.get(model_key_or_repo, {}).get("imgsz", 640)
    return ModelBundle(
        repo_id=repo_id,
        weights_path=weights,
        classes=cfg.get("classes") or [],
        imgsz=imgsz,
    )
