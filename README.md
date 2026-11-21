# evren-yolo-vision-kit

[![CI](https://github.com/kenantabak/evren-yolo-vision-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/kenantabak/evren-yolo-vision-kit/actions/workflows/ci.yml)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-16_Open_Models-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/kenantabak)

**[Hugging Face](https://huggingface.co/kenantabak)** ve **[EVREN](https://evren.ssyz.org.tr/users/u19979974710e)** üzerinde paylaşılan açık `YOLO26` (`yolo26s`, `yolo26x`) ve `YOLO11` (`yolo11s`) modellerini doğrudan çalıştırmak için yazılmış hafif PyTorch / saf **ONNX Runtime** çıkarım kütüphanesi, CLI aracı ve Gradio web arayüzü.

## Mimari Akış

```text
Hugging Face Hub (kenantabak/*)        evren-yolo-vision-kit
  ├── best.onnx / best.pt       ───▶   hub.download_model_bundle()
  └── evren_model_config.json   ───▶   OnnxDetector (NumPy + PIL + ONNXRuntime)
                                       └──▶ CLI / Python API / Gradio Demo
```

## Kurulum

```bash
git clone https://github.com/kenantabak/evren-yolo-vision-kit.git
cd evren-yolo-vision-kit
pip install -e ".[onnx,ui]"
```

## Hızlı Kullanım

### 1. CLI ile Model Listeleme ve Tespit

```bash
evren-vision list-models
evren-vision predict --model ppe-pro --source santiye.jpg --conf 0.25 --save out.jpg
```

### 2. Python SDK (Saf ONNX Runtime — GPU veya CPU)

```python
from PIL import Image
from vision_kit.hub import download_model_bundle
from vision_kit.onnx_engine import OnnxDetector
from vision_kit.visualize import draw_detections

bundle = download_model_bundle("ppe-pro", backend="onnx")
detector = OnnxDetector(bundle.weights_path, bundle.classes, imgsz=bundle.imgsz)

detections = detector.predict(Image.open("santiye.jpg"), conf=0.25)
draw_detections(Image.open("santiye.jpg"), detections).save("output.jpg")
```

### 3. Yerel Gradio Web Arayüzünü Başlatma

```bash
python app.py
```

## Testler

```bash
pytest -v
```
