
from typing import Any

import cv2
import numpy as np


# =====================================================
# CONFIGURATION
# =====================================================

IOU_MATCH_THRESHOLD = 0.30
SCORE_CHANGE_THRESHOLD = 0.05


# =====================================================
# YOLO BOX UTILITIES
# =====================================================

def normalize_box(box, width, height):
    """Convert pixel coordinates into normalized 0-1 coordinates."""

    x1, y1, x2, y2 = map(float, box)

    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")

    return [
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, x2 / width)),
        max(0.0, min(1.0, y2 / height)),
    ]


def calculate_iou(box_a, box_b):
    """Calculate intersection over union between two normalized boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = (
        max(0.0, ix2 - ix1) *
        max(0.0, iy2 - iy1)
    )

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def transform_box(box, transform_matrix, width, height):
    """
    Transform a previous-scan YOLO box into current-scan
    pixel coordinates.

    transform_matrix must map previous -> current coordinates.
    """

    x1, y1, x2, y2 = map(float, box)

    corners = np.array([
        [x1, y1, 1.0],
        [x2, y1, 1.0],
        [x1, y2, 1.0],
        [x2, y2, 1.0],
    ], dtype=np.float64)

    transformed = corners @ transform_matrix.T

    if not np.all(np.isfinite(transformed)):
        raise ValueError("Invalid transformed bounding box")

    xs = transformed[:, 0]
    ys = transformed[:, 1]

    new_box = [
        float(np.clip(xs.min(), 0, width)),
        float(np.clip(ys.min(), 0, height)),
        float(np.clip(xs.max(), 0, width)),
        float(np.clip(ys.max(), 0, height)),
    ]

    if new_box[2] <= new_box[0] or new_box[3] <= new_box[1]:
        return None

    return new_box


def get_registration_transform(registration_result):
    """
    ECC registration in our registration.py uses:

        findTransformECC(current, previous, matrix, ...)
        warpAffine(previous, ..., WARP_INVERSE_MAP)

    Therefore, the returned ECC matrix maps current -> previous.
    We invert it to obtain previous -> current for bounding boxes.
    """

    if not isinstance(registration_result, dict):
        return None

    if registration_result.get("status") != "estimated":
        return None

    matrix = registration_result.get("warp_matrix")

    if matrix is None:
        return None

    try:
        matrix = np.asarray(matrix, dtype=np.float64)

        if matrix.shape != (2, 3):
            return None

        if not np.all(np.isfinite(matrix)):
            return None

        homogeneous = np.eye(3, dtype=np.float64)
        homogeneous[:2, :] = matrix

        inverse = np.linalg.inv(homogeneous)

        if not np.all(np.isfinite(inverse)):
            return None

        return inverse[:2, :]

    except (ValueError, np.linalg.LinAlgError):
        return None


# =====================================================
# DENSENET COMPARISON
# =====================================================

def compare_densenet(previous, current):

    previous_findings = previous["densenet"]["findings"]
    current_findings = current["densenet"]["findings"]

    results = {}

    all_findings = sorted(
        set(previous_findings) | set(current_findings)
    )

    for finding in all_findings:

        old_score = float(previous_findings.get(finding, 0.0))
        new_score = float(current_findings.get(finding, 0.0))

        delta = new_score - old_score

        if abs(delta) < SCORE_CHANGE_THRESHOLD:
            direction = "relatively_stable"
        elif delta > 0:
            direction = "score_increased"
        else:
            direction = "score_decreased"

        results[finding] = {
            "previous_score": round(old_score, 5),
            "current_score": round(new_score, 5),
            "change": round(delta, 5),
            "direction": direction,
        }

    return results


# =====================================================
# YOLO DETECTION PREPARATION
# =====================================================

def prepare_detections(prediction, transform=None, target_dimensions=None):

    dimensions = prediction["image_dimensions"]

    width = int(dimensions["width"])
    height = int(dimensions["height"])

    prepared = []

    for detection in prediction["yolo"]["detections"]:

        pixel_box = list(map(float, detection["box_xyxy"]))

        if transform is not None:

            target_width, target_height = target_dimensions

            pixel_box = transform_box(
                pixel_box,
                transform,
                target_width,
                target_height
            )

            if pixel_box is None:
                continue

            box = normalize_box(
                pixel_box,
                target_width,
                target_height
            )

        else:
            box = normalize_box(pixel_box, width, height)

        prepared.append({
            "box": box,
            "confidence": float(detection["confidence"]),
            "class_name": detection.get("class_name", "unknown"),
        })

    return prepared


# =====================================================
# YOLO COMPARISON
# =====================================================

def compare_yolo(previous, current, registration_result=None):

    old_dimensions = previous["image_dimensions"]
    new_dimensions = current["image_dimensions"]

    old_width = int(old_dimensions["width"])
    old_height = int(old_dimensions["height"])

    new_width = int(new_dimensions["width"])
    new_height = int(new_dimensions["height"])

    transform = get_registration_transform(registration_result)

    registration_status = (
        registration_result.get("status", "unavailable")
        if isinstance(registration_result, dict)
        else "unavailable"
    )

    # Do not perform spatial matching if registration failed,
    # was skipped, or dimensions are incompatible.
    if (
        transform is None
        or old_width != new_width
        or old_height != new_height
    ):

        old_detections = prepare_detections(previous)
        new_detections = prepare_detections(current)

        return {
            "previous_detection_count": len(old_detections),
            "current_detection_count": len(new_detections),

            "registration_status": registration_status,

            "matching_status": "not_performed_registration_unavailable",

            "spatial_matching_reliable": False,

            "matched_regions": [],
            "newly_detected_regions": [],
            "previous_regions_not_matched": [],

            "previous_detections": old_detections,
            "current_detections": new_detections,

            "matching_iou_threshold": IOU_MATCH_THRESHOLD,

            "notice": (
                "Spatial matching was not performed because "
                "a valid registration transformation was unavailable "
                "or image dimensions differed."
            ),
        }

    # Transform previous detections into current image coordinates.
    old_detections = prepare_detections(
        previous,
        transform=transform,
        target_dimensions=(new_width, new_height)
    )

    new_detections = prepare_detections(current)

    candidates = []

    for old_index, old in enumerate(old_detections):

        for new_index, new in enumerate(new_detections):

            if old["class_name"] != new["class_name"]:
                continue

            iou = calculate_iou(old["box"], new["box"])

            if iou >= IOU_MATCH_THRESHOLD:
                candidates.append((iou, old_index, new_index))

    # Greedy one-to-one matching, highest IoU first.
    candidates.sort(reverse=True)

    matched_old = set()
    matched_new = set()

    matched_regions = []

    for iou, old_index, new_index in candidates:

        if old_index in matched_old or new_index in matched_new:
            continue

        old = old_detections[old_index]
        new = new_detections[new_index]

        matched_old.add(old_index)
        matched_new.add(new_index)

        matched_regions.append({
            "previous_box_aligned": old["box"],
            "current_box": new["box"],

            "previous_confidence": round(old["confidence"], 5),
            "current_confidence": round(new["confidence"], 5),

            "confidence_change": round(
                new["confidence"] - old["confidence"], 5
            ),

            "iou": round(iou, 5),
            "class_name": old["class_name"],
        })

    newly_detected = [
        {
            "box": detection["box"],
            "confidence": round(detection["confidence"], 5),
            "class_name": detection["class_name"],
        }
        for index, detection in enumerate(new_detections)
        if index not in matched_new
    ]

    no_longer_matched = [
        {
            "box_aligned": detection["box"],
            "confidence": round(detection["confidence"], 5),
            "class_name": detection["class_name"],
        }
        for index, detection in enumerate(old_detections)
        if index not in matched_old
    ]

    return {
        "previous_detection_count": len(old_detections),
        "current_detection_count": len(new_detections),

        "registration_status": registration_status,

        "matching_status": "completed",
        "spatial_matching_reliable": True,

        "matched_regions": matched_regions,
        "newly_detected_regions": newly_detected,
        "previous_regions_not_matched": no_longer_matched,

        "matching_iou_threshold": IOU_MATCH_THRESHOLD,

        "coordinate_system": "current_scan",
    }


# =====================================================
# MAIN COMPARISON FUNCTION
# =====================================================

def compare_predictions(
    previous,
    current,
    registration_result=None
):

    if not isinstance(previous, dict) or not isinstance(current, dict):
        raise ValueError("Both predictions must be dictionaries")

    if "densenet" not in previous or "densenet" not in current:
        raise ValueError("DenseNet results are missing")

    if "yolo" not in previous or "yolo" not in current:
        raise ValueError("YOLO results are missing")

    densenet_comparison = compare_densenet(previous, current)

    yolo_comparison = compare_yolo(
        previous,
        current,
        registration_result
    )

    old_top10 = float(previous["yolo"]["top10_mean"])
    new_top10 = float(current["yolo"]["top10_mean"])

    old_threshold = float(previous["yolo"]["patient_threshold"])
    new_threshold = float(current["yolo"]["patient_threshold"])

    return {
        "project": "NIVARA",

        "comparison_type": "previous_vs_current",

        "densenet_comparison": densenet_comparison,

        "yolo_comparison": yolo_comparison,

        "patient_score_comparison": {
            "previous_top10_mean": round(old_top10, 6),
            "current_top10_mean": round(new_top10, 6),

            "change": round(new_top10 - old_top10, 6),

            "previous_threshold_exceeded": (
                old_top10 >= old_threshold
            ),

            "current_threshold_exceeded": (
                new_top10 >= new_threshold
            ),
        },

        "interpretation_notice": (
            "These are changes in model outputs and detected regions, "
            "not proof of clinical improvement or deterioration. "
            "Registration and model outputs are experimental and "
            "require visual and expert review. Unmatched detections "
            "do not establish newly developed or resolved abnormalities."
        ),
    }
