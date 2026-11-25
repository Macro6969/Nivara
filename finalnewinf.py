import base64
import json
import os
import textwrap
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if not os.path.exists("/dev/nvidia0"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from ultralytics import YOLO

from nivara_langchain import OLLAMA_MODEL, generate_langchain_report

IMAGE_PATH = "test.jpeg"
OUTPUT_IMAGE_PATH = "test_nivara_annotated.jpeg"
OUTPUT_GRADCAM_PATH = "test_nivara_gradcam.jpeg"
OUTPUT_REPORT_PATH = "test_nivara_report.txt"

CIHMLC_WEIGHTS = Path("/home/voyager/Downloads/Nivara/CIHMLC/src/assets/models/default_model.keras")
YOLO26_PNEUMONIA_WEIGHTS = Path("/home/voyager/Downloads/Nivara/Nivara_Models/best_fine_tuned.pt")
YOLO26_THORACIC4_WEIGHTS = Path("/home/voyager/Downloads/Nivara/Nivara_Models/thoracic4_best.pt")

CIHMLC_PNEUMONIA_THRESHOLD = 0.05
CIHMLC_THORACIC4_THRESHOLD = 0.15
YOLO_IMG_SIZE = 640
YOLO_CONF_THRESHOLD = 0.10
THORACIC4_CONF_THRESHOLD = 0.20
YOLO_IOU_THRESHOLD = 0.50

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CIHMLC_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
    "Uncertain",
    "Abnormal",
    "Cardiac",
    "Fluid Accumulation",
    "Missing Lung Tissue",
    "Opacity",
    "Other",
]

NON_PATHOLOGY_LABELS = {"No Finding", "Support Devices", "Uncertain", "Abnormal", "Other"}
THORACIC4_TRIGGER_LABELS = ("Pneumothorax", "Pleural Effusion", "Cardiomegaly", "Lung Lesion")

# BGR colors matching YOLO_CLASS_COLORS in App.jsx
CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "pneumonia_opacity": (248, 189, 56),   # Sky blue
    "Pneumothorax": (94, 63, 244),         # Rose red
    "Pleural Effusion": (11, 158, 245),    # Amber
    "Cardiomegaly": (129, 185, 16),        # Emerald
    "Nodule/Mass": (247, 85, 168),         # Purple
}

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def load_chest_xray_rgb(image_path: str | Path) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    if path.suffix.lower() == ".dcm":
        import pydicom
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            arr = arr.max() - arr
        arr_min, arr_max = float(arr.min()), float(arr.max())
        arr = ((arr - arr_min) / (arr_max - arr_min) * 255.0) if arr_max > arr_min else np.zeros_like(arr)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")

    return Image.open(path).convert("RGB")


def prepare_cihmlc_input(pil_img: Image.Image, model: tf.keras.Model) -> np.ndarray:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    img_h = input_shape[1] or 224
    img_w = input_shape[2] or 224
    channels = input_shape[3] or 3

    mode = "L" if channels == 1 else "RGB"
    resized = pil_img.convert(mode).resize((img_w, img_h), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    if channels == 1 and arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)

    has_rescaling = any(isinstance(layer, tf.keras.layers.Rescaling) for layer in model.layers)
    if not has_rescaling:
        arr = arr / 255.0

    return np.expand_dims(arr, axis=0)


def find_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
    )
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if isinstance(layer, conv_types):
            return layer
        if isinstance(layer, tf.keras.Model):
            try:
                if len(layer.output.shape) == 4:
                    return layer
            except Exception:
                continue

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        try:
            if len(layer.output.shape) == 4:
                return layer
        except Exception:
            continue

    raise ValueError("No 4D convolutional feature layer found in CIHMLC model for Grad-CAM.")


def select_gradcam_target(label_scores: dict[str, float]) -> tuple[str, int, float]:
    candidates = [
        (lbl, score)
        for lbl, score in label_scores.items()
        if lbl not in NON_PATHOLOGY_LABELS
    ]
    if not candidates:
        p_score = label_scores.get("Pneumonia", 0.0)
        return "Pneumonia", CIHMLC_LABELS.index("Pneumonia"), p_score

    top_label, top_score = max(candidates, key=lambda item: item[1])
    if top_score < 0.15 and "Pneumonia" in label_scores:
        return "Pneumonia", CIHMLC_LABELS.index("Pneumonia"), label_scores["Pneumonia"]
    return top_label, CIHMLC_LABELS.index(top_label), top_score


def estimate_anatomical_zone(cx: float, cy: float, class_name: str) -> str:
    # Standard PA/AP radiological convention: image left = patient right
    side = "Right" if cx < 0.46 else ("Left" if cx > 0.54 else "Central")
    if class_name == "Cardiomegaly":
        return "Cardiomediastinal silhouette"
    vertical = "upper" if cy < 0.38 else ("mid" if cy < 0.66 else "lower")
    return f"{side} {vertical} lung zone"


def encode_bgr_base64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def compute_gradcam_overlay(
    grad_model: tf.keras.Model,
    input_batch: np.ndarray,
    orig_bgr: np.ndarray,
    class_idx: int,
    alpha: float = 0.45,
) -> np.ndarray:
    input_tensor = tf.convert_to_tensor(input_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor, training=False)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        class_channel = predictions[:, class_idx]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return orig_bgr.copy()

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_map = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_map * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    heatmap_np = heatmap.numpy()
    orig_h, orig_w = orig_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
    heatmap_u8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(orig_bgr, 1.0 - alpha, heatmap_color, alpha, 0)


def draw_panel_header(img: np.ndarray, title: str, color: tuple[int, int, int] = (0, 220, 255)) -> np.ndarray:
    panel = img.copy()
    h, w = panel.shape[:2]
    font_scale = max(0.55, min(1.0, w / 900.0))
    thickness = max(1, int(round(font_scale * 2)))
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(panel, (0, 0), (min(w, tw + 24), th + 20), (15, 15, 15), -1)
    cv2.putText(
        panel,
        title,
        (12, th + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return panel


def render_report_panel(report_text: str, height: int, width: int = 720) -> np.ndarray:
    panel = np.full((height, width, 3), 22, dtype=np.uint8)
    cv2.putText(
        panel,
        f"RADIOLOGY REPORT (LangChain RAG + {OLLAMA_MODEL})",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.line(panel, (18, 50), (width - 18, 50), (70, 70, 70), 1)

    y = 78
    for raw_line in report_text.splitlines():
        if y > height - 20:
            break
        clean = raw_line.replace("**", "").replace("###", "").replace("##", "").strip()
        if not clean:
            y += 12
            continue
        for wrapped in textwrap.wrap(clean, width=74):
            if y > height - 20:
                break
            cv2.putText(
                panel,
                wrapped,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            y += 22
    return panel


def show_nivara_window(
    gradcam_bgr: np.ndarray,
    yolo_bgr: np.ndarray,
    report_text: str = "",
    window_name: str = "NIVARA: Grad-CAM, Dual YOLO26m & RAG Radiology Report",
) -> None:
    h1, w1 = gradcam_bgr.shape[:2]
    if yolo_bgr.shape[:2] != (h1, w1):
        yolo_bgr = cv2.resize(yolo_bgr, (w1, h1), interpolation=cv2.INTER_LINEAR)

    panels = [gradcam_bgr, yolo_bgr]
    if report_text:
        panels.append(render_report_panel(report_text, height=h1))

    combined = np.hstack(panels)
    ch, cw = combined.shape[:2]
    max_w, max_h = 1760, 820
    scale = min(max_w / cw, max_h / ch, 1.0)
    if scale < 1.0:
        combined = cv2.resize(combined, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, combined)
        close_keys = {27, ord("q"), ord("Q"), 13, 32}
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in close_keys:
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    except cv2.error as exc:
        print(f"[OpenCV display skipped: {exc}]")
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def _collect_yolo_detections(
    model: YOLO,
    source_bgr: np.ndarray,
    conf_threshold: float,
    default_name: str | None = None,
) -> list[dict]:
    results = model.predict(
        source=source_bgr,
        imgsz=YOLO_IMG_SIZE,
        conf=conf_threshold,
        iou=YOLO_IOU_THRESHOLD,
        device=DEVICE,
        augment=False,
        verbose=False,
    )
    res = results[0]
    items: list[dict] = []
    if res.boxes is None or len(res.boxes) == 0:
        return items

    for box in res.boxes:
        cls_id = int(box.cls.item())
        conf = round(float(box.conf.item()), 4)
        x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0].tolist()]
        cx, cy, nw, nh = [round(float(v), 6) for v in box.xywhn[0].tolist()]
        cls_name = default_name or res.names.get(cls_id, "pneumonia_opacity")
        items.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": conf,
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_xywh_norm": [cx, cy, nw, nh],
            "zone": estimate_anatomical_zone(cx, cy, cls_name),
        })
    return items


print(f"Loading CIHMLC Keras model from {CIHMLC_WEIGHTS}...")
if not CIHMLC_WEIGHTS.exists():
    raise FileNotFoundError(f"CIHMLC checkpoint not found: {CIHMLC_WEIGHTS}")
cihmlc_model = tf.keras.models.load_model(str(CIHMLC_WEIGHTS), compile=False)
last_conv_layer = find_last_conv_layer(cihmlc_model)
cihmlc_grad_model = tf.keras.Model(
    inputs=cihmlc_model.inputs,
    outputs=[last_conv_layer.output, cihmlc_model.output],
)

print(f"Loading Pneumonia YOLO26m detector from {YOLO26_PNEUMONIA_WEIGHTS} on {DEVICE}...")
if not YOLO26_PNEUMONIA_WEIGHTS.exists():
    raise FileNotFoundError(f"Pneumonia YOLO26m checkpoint not found: {YOLO26_PNEUMONIA_WEIGHTS}")
yolo26_pneumonia_model = YOLO(str(YOLO26_PNEUMONIA_WEIGHTS))

yolo26_thoracic4_model: YOLO | None = None
if YOLO26_THORACIC4_WEIGHTS.exists():
    print(f"Loading 4-Class Thoracic YOLO26m detector from {YOLO26_THORACIC4_WEIGHTS} on {DEVICE}...")
    yolo26_thoracic4_model = YOLO(str(YOLO26_THORACIC4_WEIGHTS))
else:
    print(f"[Info] 4-Class Thoracic YOLO26m weights not found at {YOLO26_THORACIC4_WEIGHTS}; running single-YOLO gate.")


def run_nivara_inference(
    image_path: str | Path,
    save_annotated_path: str | Path | None = OUTPUT_IMAGE_PATH,
    save_gradcam_path: str | Path | None = OUTPUT_GRADCAM_PATH,
    save_report_path: str | Path | None = OUTPUT_REPORT_PATH,
    display_window: bool = True,
    include_base64: bool = False,
) -> dict:
    pil_img = load_chest_xray_rgb(image_path)
    orig_w, orig_h = pil_img.size
    orig_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    t0_cihmlc = time.perf_counter()
    input_batch = prepare_cihmlc_input(pil_img, cihmlc_model)
    raw_preds = cihmlc_model.predict(input_batch, verbose=0)[0].astype(np.float64)
    if np.any(raw_preds < 0.0) or np.any(raw_preds > 1.0):
        probs = 1.0 / (1.0 + np.exp(-raw_preds))
    else:
        probs = raw_preds

    label_scores = {
        CIHMLC_LABELS[i]: round(float(probs[i]), 6)
        for i in range(min(len(CIHMLC_LABELS), len(probs)))
    }
    gradcam_label, gradcam_idx, gradcam_score = select_gradcam_target(label_scores)
    gradcam_bgr = compute_gradcam_overlay(cihmlc_grad_model, input_batch, orig_bgr, gradcam_idx)
    cihmlc_time = round(time.perf_counter() - t0_cihmlc, 4)

    pneumonia_score = label_scores["Pneumonia"]
    thoracic4_peak = max((label_scores.get(lbl, 0.0) for lbl in THORACIC4_TRIGGER_LABELS), default=0.0)

    gradcam_panel = draw_panel_header(
        gradcam_bgr,
        f"CIHMLC Grad-CAM | {gradcam_label}: {gradcam_score:.4f}",
        color=(0, 220, 255),
    )

    gradcam_saved_path = None
    if save_gradcam_path is not None:
        cv2.imwrite(str(save_gradcam_path), gradcam_panel)
        gradcam_saved_path = str(save_gradcam_path)

    run_pneumonia_yolo = pneumonia_score >= CIHMLC_PNEUMONIA_THRESHOLD
    run_thoracic4_yolo = (
        yolo26_thoracic4_model is not None and thoracic4_peak >= CIHMLC_THORACIC4_THRESHOLD
    )

    detections: list[dict] = []
    yolo_time = 0.0
    saved_path = None

    if not run_pneumonia_yolo and not run_thoracic4_yolo:
        routing_decision = "YOLO26_SKIPPED"
        yolo_panel = draw_panel_header(
            orig_bgr,
            f"YOLO26m Skipped (Pneumonia < {CIHMLC_PNEUMONIA_THRESHOLD})",
            color=(180, 180, 180),
        )
    else:
        if run_pneumonia_yolo and run_thoracic4_yolo:
            routing_decision = "DUAL_YOLO_ACTIVATED"
        elif run_thoracic4_yolo:
            routing_decision = "THORACIC4_YOLO_ACTIVATED"
        else:
            routing_decision = "PNEUMONIA_YOLO_ACTIVATED"

        t0_yolo = time.perf_counter()
        if run_pneumonia_yolo:
            detections.extend(
                _collect_yolo_detections(
                    yolo26_pneumonia_model,
                    orig_bgr,
                    conf_threshold=YOLO_CONF_THRESHOLD,
                    default_name="pneumonia_opacity",
                )
            )
        if run_thoracic4_yolo and yolo26_thoracic4_model is not None:
            detections.extend(
                _collect_yolo_detections(
                    yolo26_thoracic4_model,
                    orig_bgr,
                    conf_threshold=THORACIC4_CONF_THRESHOLD,
                )
            )
        yolo_time = round(time.perf_counter() - t0_yolo, 4)

        canvas = orig_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            color_bgr = CLASS_COLORS_BGR.get(cls_name, (0, 220, 255))

            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(canvas, p1, p2, color_bgr, 3)
            tag = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(
                canvas,
                (p1[0], max(0, p1[1] - th - 10)),
                (p1[0] + tw + 8, p1[1]),
                color_bgr,
                -1,
            )
            cv2.putText(
                canvas,
                tag,
                (p1[0] + 4, max(th + 4, p1[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (15, 15, 15),
                2,
                cv2.LINE_AA,
            )

        yolo_panel = draw_panel_header(
            canvas,
            f"{routing_decision}: {len(detections)} detections",
            color=(0, 255, 140),
        )
        if save_annotated_path is not None:
            cv2.imwrite(str(save_annotated_path), yolo_panel)
            saved_path = str(save_annotated_path)

    result = {
        "filename": str(image_path),
        "image_size": [orig_w, orig_h],
        "status": "success",
        "cihmlc_pneumonia_score": pneumonia_score,
        "gradcam_target_label": gradcam_label,
        "routing_threshold": CIHMLC_PNEUMONIA_THRESHOLD,
        "routing_decision": routing_decision,
        "yolo26_detection_count": len(detections),
        "yolo26_detections": detections,
        "cihmlc_time_seconds": cihmlc_time,
        "yolo26_time_seconds": yolo_time,
        "gradcam_image": gradcam_saved_path,
        "annotated_image": saved_path,
        "cihmlc_scores": label_scores,
    }

    if include_base64:
        result["gradcam_base64"] = encode_bgr_base64(gradcam_bgr)
        result["annotated_base64"] = encode_bgr_base64(yolo_panel)

    print(f"Generating RAG radiology report via LangChain + {OLLAMA_MODEL}...")
    report_text, grounding_audit = generate_langchain_report(result)
    result["radiology_report"] = report_text
    result["grounding_audit"] = grounding_audit

    if save_report_path is not None:
        Path(save_report_path).write_text(report_text, encoding="utf-8")

    print("\n" + "=" * 65)
    print("PRELIMINARY RADIOLOGY REPORT")
    print("=" * 65)
    print(report_text)
    print("=" * 65 + "\n")

    if display_window:
        show_nivara_window(gradcam_panel, yolo_panel, report_text)

    return result


if __name__ == "__main__":
    output = run_nivara_inference(
        IMAGE_PATH,
        save_annotated_path=OUTPUT_IMAGE_PATH,
        save_gradcam_path=OUTPUT_GRADCAM_PATH,
        save_report_path=OUTPUT_REPORT_PATH,
        display_window=True,
    )
    print(f"Image:                  {output['filename']}")
    print(f"CIHMLC Pneumonia Score: {output['cihmlc_pneumonia_score']:.6f} (Gate >= {CIHMLC_PNEUMONIA_THRESHOLD})")
    print(f"Grad-CAM Target:        {output['gradcam_target_label']}")
    print(f"Routing Decision:       {output['routing_decision']}")
    print(f"YOLO26 Detections:      {output['yolo26_detection_count']}")
    print(f"Latency (CIHMLC/YOLO):  {output['cihmlc_time_seconds']}s / {output['yolo26_time_seconds']}s")
    print(f"Grounding Audit Score:  {output['grounding_audit']['grounding_score']}")
    if output["gradcam_image"]:
        print(f"Grad-CAM Output Saved:  {output['gradcam_image']}")
    if output["annotated_image"]:
        print(f"Annotated Output Saved: {output['annotated_image']}")
    print(f"Report Output Saved:    {OUTPUT_REPORT_PATH}")
    print("-" * 65)
    print(json.dumps({k: v for k, v in output.items() if k != "radiology_report"}, indent=2))