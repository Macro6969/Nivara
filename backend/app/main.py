
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from io import BytesIO

import torch
import torchxrayvision as xrv
import pydicom

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

# NIVARA services
from app.services.preprocessing import (
    load_dicom,
    prepare_yolo_image,
    prepare_densenet_image
)

from app.services.densenet_service import (
    load_densenet,
    predict_densenet
)

from app.services.yolo_service import (
    load_yolo,
    predict_yolo
)

from app.services.comparison import compare_predictions

from app.services.metadata_validation import (
    extract_comparison_metadata,
    validate_scan_pair
)

from app.services.quality_validation import (
    validate_xray_pair
)

from app.services.registration import (
    register_images
)


# =====================================================
# CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

YOLO_PATH = PROJECT_ROOT / "Nivara_Models" / "best.pt"

MAX_UPLOAD_SIZE = 50 * 1024 * 1024


# =====================================================
# MODEL LIFECYCLE
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\nStarting NIVARA...")

    device = torch.device("cpu")

    print("Inference device:", device)

    print("Loading DenseNet121...")

    app.state.densenet = load_densenet(device)

    print("Loading YOLOv8n...")

    app.state.yolo = load_yolo(YOLO_PATH)

    app.state.center_crop = xrv.datasets.XRayCenterCrop()

    app.state.resizer = xrv.datasets.XRayResizer(224)

    app.state.device = device

    app.state.inference_lock = Lock()

    print("Both models loaded successfully.")

    print("NIVARA is ready!\n")

    yield

    print("NIVARA shutting down.")


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="NIVARA API",
    description="AI-assisted chest X-ray analysis",
    version="1.3.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =====================================================
# SHARED HELPERS
# =====================================================

def read_upload(file: UploadFile) -> bytes:

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is missing"
        )

    if not file.filename.lower().endswith(".dcm"):
        raise HTTPException(
            status_code=400,
            detail=f"{file.filename}: only DICOM (.dcm) files are supported"
        )

    content = file.file.read(MAX_UPLOAD_SIZE + 1)

    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"{file.filename}: file exceeds 50 MB limit"
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail=f"{file.filename}: uploaded file is empty"
        )

    return content


def prepare_xray(content: bytes):

    gray = load_dicom(content)

    yolo_image = prepare_yolo_image(gray)

    dense_image = prepare_densenet_image(
        gray,
        app.state.center_crop,
        app.state.resizer
    )

    return gray, yolo_image, dense_image


def run_models(gray, yolo_image, dense_image, filename):

    with app.state.inference_lock:

        densenet_scores = predict_densenet(
            app.state.densenet,
            dense_image,
            app.state.device
        )

        yolo_results = predict_yolo(
            app.state.yolo,
            yolo_image
        )

    return {
        "project": "NIVARA",
        "status": "success",
        "filename": filename,

        "image_dimensions": {
            "width": int(gray.shape[1]),
            "height": int(gray.shape[0])
        },

        "densenet": {
            "model": "DenseNet121",
            "weights": "densenet121-res224-all",
            "score_interpretation": (
                "Model output scores; not calibrated probabilities"
            ),
            "findings": densenet_scores
        },

        "yolo": {
            "model": "NIVARA YOLOv8n",
            **yolo_results
        },

        "clinical_notice": (
            "Research prototype only. Model outputs are not "
            "a medical diagnosis."
        )
    }


# =====================================================
# PATIENT ID EXTRACTION
# =====================================================

def extract_patient_id(file_bytes: bytes):

    try:

        ds = pydicom.dcmread(
            BytesIO(file_bytes),
            stop_before_pixels=True
        )

        patient_id = str(
            getattr(ds, "PatientID", "")
        ).strip()

        return patient_id or None

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail="Could not read DICOM patient metadata."
        ) from exc


# =====================================================
# BASIC ROUTES
# =====================================================

@app.get("/")
def root():

    return {
        "project": "NIVARA",
        "status": "running",
        "version": "1.3.0"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "device": str(app.state.device),
        "densenet_loaded": app.state.densenet is not None,
        "yolo_loaded": app.state.yolo is not None
    }


# =====================================================
# SINGLE X-RAY PREDICTION
# =====================================================

@app.post("/predict")
def predict(file: UploadFile = File(...)):

    try:

        content = read_upload(file)

        gray, yolo_image, dense_image = prepare_xray(content)

        return run_models(
            gray,
            yolo_image,
            dense_image,
            file.filename
        )

    except HTTPException:
        raise

    except Exception as error:

        print("Prediction error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="X-ray processing failed. Check backend logs."
        )


# =====================================================
# PREVIOUS VS CURRENT COMPARISON
# =====================================================

@app.post("/compare")
def compare(

    previous_file: UploadFile = File(...),

    current_file: UploadFile = File(...),

    same_patient_confirmed: bool = Form(False)

):

    try:

        # -------------------------------------------------
        # STEP 1: READ BOTH FILES
        # -------------------------------------------------

        previous_content = read_upload(previous_file)

        current_content = read_upload(current_file)


        # -------------------------------------------------
        # STEP 2: PATIENT IDENTITY VERIFICATION
        # -------------------------------------------------

        previous_patient_id = extract_patient_id(
            previous_content
        )

        current_patient_id = extract_patient_id(
            current_content
        )

        if previous_patient_id and current_patient_id:

            if previous_patient_id != current_patient_id:

                raise HTTPException(
                    status_code=400,
                    detail="Patient ID mismatch. These scans cannot be compared."
                )

            identity_status = "verified_match"

        else:

            if not same_patient_confirmed:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Patient identity could not be verified "
                        "from DICOM metadata. Confirm that both "
                        "scans belong to the same patient."
                    )
                )

            identity_status = "user_confirmed_unverified"


        # -------------------------------------------------
        # STEP 3: METADATA EXTRACTION
        # -------------------------------------------------

        previous_metadata = extract_comparison_metadata(
            previous_content
        )

        current_metadata = extract_comparison_metadata(
            current_content
        )


        # -------------------------------------------------
        # STEP 4: METADATA VALIDATION
        # -------------------------------------------------

        metadata_validation = validate_scan_pair(
            previous_metadata,
            current_metadata
        )

        if not metadata_validation["can_compare"]:

            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Scans are not compatible for comparison.",
                    "metadata_validation": metadata_validation
                }
            )


        # -------------------------------------------------
        # STEP 5: IMAGE PREPROCESSING
        # -------------------------------------------------

        (
            previous_gray,
            previous_yolo,
            previous_dense
        ) = prepare_xray(previous_content)

        (
            current_gray,
            current_yolo,
            current_dense
        ) = prepare_xray(current_content)


        # -------------------------------------------------
        # STEP 6: IMAGE QUALITY VALIDATION
        # -------------------------------------------------

        quality_validation = validate_xray_pair(
            previous_gray,
            current_gray
        )

        if not quality_validation["can_process"]:

            raise HTTPException(
                status_code=422,
                detail={
                    "message": "One or both scans failed image quality checks.",
                    "quality_validation": quality_validation
                }
            )


        # -------------------------------------------------
        # STEP 7: IMAGE REGISTRATION
        # -------------------------------------------------

        aligned_previous_gray, registration_result = register_images(
            previous_gray,
            current_gray
        )


        # -------------------------------------------------
        # STEP 8: MODEL INFERENCE
        # -------------------------------------------------

        previous_prediction = run_models(
            previous_gray,
            previous_yolo,
            previous_dense,
            previous_file.filename
        )

        current_prediction = run_models(
            current_gray,
            current_yolo,
            current_dense,
            current_file.filename
        )


        # -------------------------------------------------
        # STEP 9: MODEL OUTPUT COMPARISON
        # -------------------------------------------------

        comparison = compare_predictions(
            previous_prediction,
            current_prediction,
            registration_result=registration_result
        )


        # -------------------------------------------------
        # STEP 10: FINAL RESPONSE
        # -------------------------------------------------

        return {

            "project": "NIVARA",

            "status": "success",

            "previous_filename": previous_file.filename,

            "current_filename": current_file.filename,

            "patient_identity": identity_status,

            "metadata_validation": metadata_validation,

            "quality_validation": quality_validation,

            "image_registration": registration_result,

            "previous_prediction": previous_prediction,

            "current_prediction": current_prediction,

            "comparison": comparison,

            "clinical_notice": (
                "Research prototype only. Changes in model scores "
                "and detections do not establish clinical improvement "
                "or deterioration. Registration is experimental and "
                "does not establish anatomical correspondence. "
                "Clinical review is required."
            )

        }


    except HTTPException:
        raise

    except Exception as error:

        print("Comparison error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="X-ray comparison failed. Check backend logs."
        )
