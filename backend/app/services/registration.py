
import cv2
import numpy as np


def _normalize_for_registration(image):
    """Convert a grayscale image into a normalized float32 image."""

    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 2 or image.size == 0:
        raise ValueError("Registration requires a non-empty grayscale image.")

    if not np.isfinite(image).all():
        raise ValueError("Image contains invalid pixel values.")

    p1, p99 = np.percentile(image, [1, 99])

    if p99 - p1 < 1e-6:
        raise ValueError("Image has insufficient intensity variation.")

    image = np.clip((image - p1) / (p99 - p1), 0, 1)

    return image.astype(np.float32)


def register_images(previous_gray, current_gray):
    """
    Align the previous scan to the current scan using OpenCV ECC.

    Initial model: Euclidean transformation (rotation + translation).

    Returns:
        aligned_previous: aligned grayscale image, or None
        report: JSON-compatible registration diagnostics

    This is an experimental technical alignment, not clinical validation.
    """

    # -------------------------------------------------
    # 1. Validate image dimensions
    # -------------------------------------------------

    if previous_gray is None or current_gray is None:
        return None, {
            "status": "skipped",
            "reason": "One or both images are missing."
        }

    if previous_gray.ndim != 2 or current_gray.ndim != 2:
        return None, {
            "status": "skipped",
            "reason": "Registration requires grayscale images."
        }

    if previous_gray.shape != current_gray.shape:
        return None, {
            "status": "skipped",
            "reason": (
                "Image dimensions differ. "
                "Automatic resizing is not enabled in this version."
            ),
            "previous_shape": list(previous_gray.shape),
            "current_shape": list(current_gray.shape)
        }

    height, width = current_gray.shape

    try:
        previous = _normalize_for_registration(previous_gray)
        current = _normalize_for_registration(current_gray)

        # Light smoothing reduces sensitivity to individual pixel noise.
        previous = cv2.GaussianBlur(previous, (5, 5), 0)
        current = cv2.GaussianBlur(current, (5, 5), 0)

        # -------------------------------------------------
        # 2. Estimate transformation
        # -------------------------------------------------

        warp_matrix = np.eye(2, 3, dtype=np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            150,
            1e-6
        )

        ecc_score, warp_matrix = cv2.findTransformECC(
            current,
            previous,
            warp_matrix,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            5
        )

        # -------------------------------------------------
        # 3. Extract transformation parameters
        # -------------------------------------------------

        rotation_degrees = np.degrees(
            np.arctan2(
                warp_matrix[1, 0],
                warp_matrix[0, 0]
            )
        )

        translation_x = float(warp_matrix[0, 2])
        translation_y = float(warp_matrix[1, 2])

        # -------------------------------------------------
        # 4. Reject implausibly large transformations
        # -------------------------------------------------

        max_translation_x = width * 0.10
        max_translation_y = height * 0.10

        if abs(rotation_degrees) > 12:
            return None, {
                "status": "rejected",
                "reason": "Estimated rotation is too large.",
                "ecc_score": float(ecc_score),
                "rotation_degrees": float(rotation_degrees)
            }

        if (
            abs(translation_x) > max_translation_x
            or abs(translation_y) > max_translation_y
        ):
            return None, {
                "status": "rejected",
                "reason": "Estimated translation is too large.",
                "ecc_score": float(ecc_score),
                "translation_x": translation_x,
                "translation_y": translation_y
            }

        # -------------------------------------------------
        # 5. Warp previous scan into current scan coordinates
        # -------------------------------------------------

        aligned_previous = cv2.warpAffine(
            previous_gray,
            warp_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # -------------------------------------------------
        # 6. Calculate valid overlapping image area
        # -------------------------------------------------

        valid_mask = cv2.warpAffine(
            np.ones((height, width), dtype=np.uint8),
            warp_matrix,
            (width, height),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        overlap_fraction = float(np.mean(valid_mask))

        if overlap_fraction < 0.75:
            return None, {
                "status": "rejected",
                "reason": "Too little image area overlaps after alignment.",
                "ecc_score": float(ecc_score),
                "overlap_fraction": overlap_fraction
            }

        # -------------------------------------------------
        # 7. Return diagnostics
        # -------------------------------------------------

        report = {
            "status": "estimated",
            "method": "OpenCV ECC",
            "transformation": "Euclidean",
            "ecc_score": float(ecc_score),
            "rotation_degrees": float(rotation_degrees),
            "translation_x": translation_x,
            "translation_y": translation_y,
            "overlap_fraction": overlap_fraction,
            "warp_matrix": warp_matrix.tolist(),
            "note": (
                "Experimental alignment only. "
                "Successful registration does not establish "
                "anatomical correspondence or clinical change."
            )
        }

        return aligned_previous, report

    except cv2.error as exc:

        return None, {
            "status": "failed",
            "reason": "OpenCV ECC registration did not converge.",
            "error": str(exc)[:300]
        }

    except (ValueError, TypeError) as exc:

        return None, {
            "status": "failed",
            "reason": str(exc)
        }
