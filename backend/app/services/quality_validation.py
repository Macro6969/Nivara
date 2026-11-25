
import cv2
import numpy as np


def assess_xray_quality(gray):
    """
    Perform basic technical quality checks on a preprocessed
    grayscale chest X-ray.

    These are heuristic checks, not clinical image-quality grading.
    """

    blocking_issues = []
    warnings = []

    # -------------------------------------------------
    # 1. Validate array
    # -------------------------------------------------

    if gray is None:
        return {
            "usable": False,
            "blocking_issues": ["Image could not be loaded."],
            "warnings": [],
            "metrics": {}
        }

    image = np.asarray(gray)

    if image.ndim != 2:
        return {
            "usable": False,
            "blocking_issues": [
                "Expected a single-channel grayscale image."
            ],
            "warnings": [],
            "metrics": {}
        }

    if image.size == 0:
        return {
            "usable": False,
            "blocking_issues": ["Image contains no pixels."],
            "warnings": [],
            "metrics": {}
        }

    if not np.isfinite(image).all():
        return {
            "usable": False,
            "blocking_issues": [
                "Image contains invalid or non-finite pixel values."
            ],
            "warnings": [],
            "metrics": {}
        }

    height, width = image.shape

    # -------------------------------------------------
    # 2. Validate dimensions
    # -------------------------------------------------

    if width < 128 or height < 128:
        blocking_issues.append(
            "Image resolution is too small for reliable analysis."
        )

    elif width < 512 or height < 512:
        warnings.append(
            "Image resolution is relatively low."
        )

    # -------------------------------------------------
    # 3. Normalize image for quality measurements
    # -------------------------------------------------

    image_float = image.astype(np.float32)

    p1 = float(np.percentile(image_float, 1))
    p99 = float(np.percentile(image_float, 99))

    dynamic_range = p99 - p1

    if dynamic_range <= 1e-6:
        blocking_issues.append(
            "Image is almost completely uniform or blank."
        )

        return {
            "usable": False,
            "blocking_issues": blocking_issues,
            "warnings": warnings,
            "metrics": {
                "width": int(width),
                "height": int(height),
                "dynamic_range": float(dynamic_range)
            }
        }

    normalized = np.clip(
        (image_float - p1) / dynamic_range,
        0,
        1
    )

    normalized_uint8 = (
        normalized * 255
    ).astype(np.uint8)

    # -------------------------------------------------
    # 4. Contrast assessment
    # -------------------------------------------------

    contrast_std = float(np.std(normalized_uint8))

    if contrast_std < 15:
        warnings.append(
            "Image has relatively low contrast."
        )

    # -------------------------------------------------
    # 5. Blur assessment
    # -------------------------------------------------

    laplacian = cv2.Laplacian(
        normalized_uint8,
        cv2.CV_64F
    )

    blur_score = float(laplacian.var())

    if blur_score < 5:
        warnings.append(
            "Image may be significantly blurred."
        )

    elif blur_score < 20:
        warnings.append(
            "Image may have reduced sharpness."
        )

    # -------------------------------------------------
    # 6. Pixel clipping assessment
    # -------------------------------------------------

    low_fraction = float(
        np.mean(normalized_uint8 <= 1)
    )

    high_fraction = float(
        np.mean(normalized_uint8 >= 254)
    )

    clipped_fraction = low_fraction + high_fraction

    if clipped_fraction > 0.25:
        warnings.append(
            "A substantial portion of pixels is near "
            "the intensity limits."
        )

    # -------------------------------------------------
    # 7. Final result
    # -------------------------------------------------

    usable = len(blocking_issues) == 0

    return {
        "usable": usable,

        "blocking_issues": blocking_issues,

        "warnings": warnings,

        "metrics": {
            "width": int(width),
            "height": int(height),
            "dynamic_range": round(dynamic_range, 2),
            "contrast_std": round(contrast_std, 2),
            "blur_score": round(blur_score, 2),
            "low_intensity_fraction": round(low_fraction, 4),
            "high_intensity_fraction": round(high_fraction, 4),
            "clipped_fraction": round(clipped_fraction, 4)
        }
    }


def validate_xray_pair(previous_gray, current_gray):

    previous_quality = assess_xray_quality(previous_gray)

    current_quality = assess_xray_quality(current_gray)

    blocking_issues = []
    warnings = []

    if not previous_quality["usable"]:
        blocking_issues.append(
            "Previous scan failed image quality checks."
        )

    if not current_quality["usable"]:
        blocking_issues.append(
            "Current scan failed image quality checks."
        )

    if previous_quality["warnings"]:
        warnings.append({
            "scan": "previous",
            "issues": previous_quality["warnings"]
        })

    if current_quality["warnings"]:
        warnings.append({
            "scan": "current",
            "issues": current_quality["warnings"]
        })

    return {
        "can_process": len(blocking_issues) == 0,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "previous_quality": previous_quality,
        "current_quality": current_quality
    }
