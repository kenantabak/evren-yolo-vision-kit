"""Local Gradio demo for running all 16 Hugging Face .onnx models."""
import gradio as gr
from vision_kit.hub import MODEL_REGISTRY, download_model_bundle
from vision_kit.onnx_engine import OnnxDetector
from vision_kit.visualize import draw_detections

_CACHE = {}

def run_inference(image, model_key, conf):
    if image is None:
        return None, []
    if model_key not in _CACHE:
        b = download_model_bundle(model_key, backend="onnx")
        _CACHE[model_key] = OnnxDetector(b.weights_path, b.classes, imgsz=b.imgsz)
    det = _CACHE[model_key]
    preds = det.predict(image, conf=conf)
    vis = draw_detections(image, preds)
    rows = [[p.class_name, round(p.confidence, 3), [round(v, 1) for v in p.xyxy]] for p in preds]
    return vis, rows

with gr.Blocks(title="EVREN YOLO26/YOLO11 ONNX Demo") as demo:
    gr.Markdown("## EVREN & Hugging Face Açık YOLO26 / YOLO11 ONNX Çıkarım Arayüzü")
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Girdi Görseli")
            m_sel = gr.Dropdown(choices=list(MODEL_REGISTRY.keys()), value="ppe-pro", label="Model Seçimi")
            conf_sl = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Güven Eşiği (Confidence)")
            btn = gr.Button("Tespit Et", variant="primary")
        with gr.Column():
            img_out = gr.Image(type="pil", label="Tespit Çıktısı")
            tbl = gr.Dataframe(headers=["Sınıf", "Güven", "Koordinat [x1,y1,x2,y2]"])
    btn.click(run_inference, inputs=[img_in, m_sel, conf_sl], outputs=[img_out, tbl])

if __name__ == "__main__":
    demo.launch()
