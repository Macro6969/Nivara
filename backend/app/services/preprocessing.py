
from io import BytesIO

import cv2
import numpy as np
import pydicom


def load_dicom(file_bytes):
    """Decode a DICOM using the established NIVARA preprocessing."""

    ds = pydicom.dcmread(BytesIO(file_bytes))
    image = ds.pixel_array.astype(np.float32)

    if image.ndim != 2:
        raise ValueError(f"Expected 2D X-ray, got {image.shape}")

    if not np.isfinite(image).all():
        raise ValueError("Image contains invalid pixel values")

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image

    low, high = np.percentile(image, [0.5, 99.5])

    if high <= low:
        raise ValueError("Invalid image intensity range")

    image = np.clip(image, low, high)

    image = (
        (image - low) / (high - low) * 255
    ).astype(np.uint8)

    return image


def prepare_yolo_image(gray):
    """Match the JPEG round-trip used in validation evaluation."""

    success, encoded = cv2.imencode(
        ".jpg",
        gray,
        [cv2.IMWRITE_JPEG_QUALITY, 92]
    )

    if not success:
        raise ValueError("JPEG encoding failed")

    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("JPEG decoding failed")

    return image


def prepare_densenet_image(gray, center_crop, resizer):
    """TorchXRayVision preprocessing for DenseNet121."""

    import torchxrayvision as xrv

    image = xrv.datasets.normalize(
        gray.astype(np.float32),
        255
    )

    image = image[None, :, :]
    image = center_crop(image)
    image = resizer(image)

    return image
