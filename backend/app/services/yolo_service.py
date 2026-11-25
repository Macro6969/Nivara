
from ultralytics import YOLO


IMAGE_SIZE = 512
INFERENCE_CONFIDENCE = 0.001
NMS_IOU = 0.7
MAX_DETECTIONS = 300

PATIENT_THRESHOLD = 0.02420


def load_yolo(model_path):
    return YOLO(str(model_path))


def predict_yolo(model, image):
    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=INFERENCE_CONFIDENCE,
        iou=NMS_IOU,
        max_det=MAX_DETECTIONS,
        device="cpu",
        verbose=False
    )

    result = results[0]
    detections = []

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, score, class_id in zip(boxes, scores, classes):
            detections.append({
                "box_xyxy": [float(v) for v in box],
                "confidence": float(score),
                "class_id": int(class_id),
                "class_name": model.names[int(class_id)]
            })

    top_scores = sorted(
        [item["confidence"] for item in detections],
        reverse=True
    )[:10]

    top10_mean = (
        sum(top_scores) / len(top_scores)
        if top_scores else 0.0
    )

    return {
        "detections": detections,
        "detection_count": len(detections),
        "top10_mean": float(top10_mean),
        "patient_threshold": PATIENT_THRESHOLD,
        "threshold_exceeded": top10_mean >= PATIENT_THRESHOLD
    }
