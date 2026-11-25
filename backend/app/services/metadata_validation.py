
from io import BytesIO
from datetime import datetime

import pydicom


# Chest radiography modalities supported by the current NIVARA pipeline.
XRAY_MODALITIES = {"CR", "DX", "DR"}

FRONTAL_VIEWS = {"PA", "AP", "AP_SUPINE", "AP_SEMI"}
LATERAL_VIEWS = {"LATERAL", "LL", "RL"}


def _get_value(dataset, attribute):
    value = getattr(dataset, attribute, None)

    if value is None:
        return None

    value = str(value).strip()
    return value if value else None


def _get_float_list(dataset, attribute):
    value = getattr(dataset, attribute, None)

    if value is None:
        return None

    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None


def _valid_study_date(value):
    """Return a real date, or None for missing/anonymized dates."""
    if not value or value in {"19010101", "19000101", "00000000"}:
        return None

    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return None


def extract_comparison_metadata(file_bytes):
    """Extract only metadata needed for scan comparison."""
    try:
        ds = pydicom.dcmread(
            BytesIO(file_bytes),
            stop_before_pixels=True
        )
    except Exception as exc:
        raise ValueError(f"Could not read DICOM metadata: {exc}")

    rows = getattr(ds, "Rows", None)
    columns = getattr(ds, "Columns", None)

    return {
        "study_date": _get_value(ds, "StudyDate"),
        "modality": _get_value(ds, "Modality"),
        "view_position": _get_value(ds, "ViewPosition"),
        "patient_position": _get_value(ds, "PatientPosition"),
        "rows": int(rows) if rows is not None else None,
        "columns": int(columns) if columns is not None else None,
        "pixel_spacing": _get_float_list(ds, "PixelSpacing"),
        "image_orientation": _get_float_list(
            ds, "ImageOrientationPatient"
        ),
    }


def validate_scan_pair(previous, current):
    """
    Check whether two scans appear suitable for comparison.

    This is a technical compatibility check, not clinical validation.
    """

    blocking_issues = []
    warnings = []

    # 1. Modality compatibility
    prev_modality = previous["modality"]
    curr_modality = current["modality"]

    if not prev_modality or not curr_modality:
        warnings.append("Modality metadata is missing from one or both scans.")

    elif (
        prev_modality not in XRAY_MODALITIES
        or curr_modality not in XRAY_MODALITIES
    ):
        blocking_issues.append(
            "Both scans must be chest radiographs (CR, DX or DR)."
        )

    elif prev_modality != curr_modality:
        warnings.append(
            f"Different imaging modalities: "
            f"{prev_modality} vs {curr_modality}."
        )

    # 2. Study date
    prev_date = _valid_study_date(previous["study_date"])
    curr_date = _valid_study_date(current["study_date"])

    if prev_date is None or curr_date is None:
        warnings.append(
            "A reliable StudyDate is unavailable. "
            "The chronological order cannot be verified from metadata."
        )
    elif prev_date > curr_date:
        blocking_issues.append(
            "The previous scan has a later StudyDate than the current scan."
        )
    elif prev_date == curr_date:
        warnings.append("Both scans have the same StudyDate.")

    # 3. View position
    prev_view = previous["view_position"]
    curr_view = current["view_position"]

    if not prev_view or not curr_view:
        warnings.append("ViewPosition is missing from one or both scans.")

    elif prev_view != curr_view:
        prev_is_frontal = prev_view in FRONTAL_VIEWS
        curr_is_frontal = curr_view in FRONTAL_VIEWS

        prev_is_lateral = prev_view in LATERAL_VIEWS
        curr_is_lateral = curr_view in LATERAL_VIEWS

        if (
            (prev_is_frontal and curr_is_lateral)
            or (prev_is_lateral and curr_is_frontal)
        ):
            blocking_issues.append(
                "One scan is frontal and the other is lateral. "
                "These views are not directly comparable."
            )
        else:
            warnings.append(
                f"Different X-ray views: {prev_view} vs {curr_view}."
            )

    # 4. Patient position
    prev_position = previous["patient_position"]
    curr_position = current["patient_position"]

    if prev_position and curr_position:
        if prev_position != curr_position:
            warnings.append(
                f"Different patient positions: "
                f"{prev_position} vs {curr_position}."
            )

    # 5. Image dimensions
    prev_rows, prev_cols = previous["rows"], previous["columns"]
    curr_rows, curr_cols = current["rows"], current["columns"]

    if all((prev_rows, prev_cols, curr_rows, curr_cols)):
        row_ratio = max(prev_rows, curr_rows) / min(prev_rows, curr_rows)
        col_ratio = max(prev_cols, curr_cols) / min(prev_cols, curr_cols)

        if row_ratio > 1.20 or col_ratio > 1.20:
            warnings.append(
                "Image dimensions differ substantially. "
                "Check cropping and image scaling."
            )
    else:
        warnings.append("Image dimensions are missing from metadata.")

    # 6. Pixel spacing
    prev_spacing = previous["pixel_spacing"]
    curr_spacing = current["pixel_spacing"]

    if prev_spacing and curr_spacing:
        if len(prev_spacing) == 2 and len(curr_spacing) == 2:
            for a, b in zip(prev_spacing, curr_spacing):
                if min(a, b) <= 0:
                    warnings.append("Invalid PixelSpacing value found.")
                    break

                ratio = max(a, b) / min(a, b)

                if ratio > 1.20:
                    warnings.append(
                        "Pixel spacing differs substantially "
                        "between scans."
                    )
                    break
    else:
        warnings.append(
            "PixelSpacing is missing from one or both scans."
        )

    # 7. Image orientation
    prev_orientation = previous["image_orientation"]
    curr_orientation = current["image_orientation"]

    if prev_orientation and curr_orientation:
        if len(prev_orientation) == 6 and len(curr_orientation) == 6:
            for start in (0, 3):
                dot = sum(
                    prev_orientation[start + i]
                    * curr_orientation[start + i]
                    for i in range(3)
                )

                if dot < 0.95:
                    warnings.append(
                        "Image orientations differ. "
                        "Registration may be more difficult."
                    )
                    break
    else:
        warnings.append(
            "ImageOrientationPatient is missing from one or both scans."
        )

    return {
        "can_compare": len(blocking_issues) == 0,
        "blocking_issues": blocking_issues,
        "warnings": list(dict.fromkeys(warnings)),
        "previous_metadata": previous,
        "current_metadata": current,
    }
