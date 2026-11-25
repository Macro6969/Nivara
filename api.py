import gc
import os
import re
import tempfile
from pathlib import Path

# Cap CPU threads before importing TensorFlow / PyTorch / OpenCV on 8 GB RAM
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

torch.set_num_threads(2)

from finalnewinf import run_nivara_inference
from nivara_langchain import build_rag_clinical_summary

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR / "fullstack" / "nivara" / "dist"

app = FastAPI(title="NIVARA Clinical Triage API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_report_sections(report_text: str) -> dict:
    """Converts MedGemma's 4-section Markdown output into structured fields for the React drawer."""
    clean = report_text.replace("**", "")

    def _grab(pattern: str, default: str = "") -> str:
        m = re.search(pattern, clean, re.IGNORECASE | re.DOTALL)
        return " ".join(m.group(1).strip().split()) if m else default

    lungs = _grab(
        r"Lungs\s*&\s*Pleura:\s*(.+?)(?=\n-\s*Cardiomediastinal|\nIMPRESSION:|$)",
        report_text,
    )
    cardiac = _grab(
        r"Cardiomediastinal(?:\s+Silhouette)?:\s*(.+?)(?=\n-\s*Support Devices|\nIMPRESSION:|$)",
        "Within normal limits.",
    )
    devices = _grab(
        r"Support Devices\s*&\s*Other:\s*(.+?)(?=\nIMPRESSION:|$)",
        "No support devices detected.",
    )

    imp_block = _grab(r"IMPRESSION:\s*(.+?)(?=\nRECOMMENDATIONS:|$)")
    impressions = [
        re.sub(r"^\d+\.\s*", "", line).strip()
        for line in re.split(r"\s+(?=\d+\.\s)", imp_block)
        if line.strip()
    ] or ["Preliminary AI-assisted impression generated."]

    rec_block = _grab(r"RECOMMENDATIONS:\s*(.+?)(?=\nAI DISCLAIMER:|$)")
    recommendations = [
        re.sub(r"^-\s*", "", line).strip()
        for line in re.split(r"\s+(?=-\s)", rec_block)
        if line.strip()
    ] or ["Routine clinical correlation as indicated."]

    return {
        "examination": "Chest Radiograph (PA/AP)",
        "findings": {
            "lungsAndPleura": lungs,
            "cardiomediastinal": cardiac,
            "devicesAndOther": devices,
        },
        "impression": impressions,
        "recommendations": recommendations,
    }


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/infer")
async def infer_radiograph(file: UploadFile = File(...)):
    suffix = Path(file.filename or "scan.jpeg").suffix.lower() or ".jpeg"
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        result = run_nivara_inference(
            tmp_path,
            save_annotated_path=None,
            save_gradcam_path=None,
            save_report_path=None,
            display_window=False,
            include_base64=True,
        )

        # Reuse guidelines already retrieved during report generation (with fallback)
        guidelines = result.get("retrieved_guidelines")
        if guidelines is None:
            summary = build_rag_clinical_summary(result)
            guidelines = (
                summary[1]
                if isinstance(summary, tuple)
                else summary.get("retrieved_pdf_guidelines", [])
            )

        result["filename"] = file.filename or result["filename"]
        result["retrieved_guidelines"] = [
            {
                "citation": g.get("citation", ""),
                "score": g.get("hybrid_score", g.get("score", 0.0)),
                "excerpt": g.get("excerpt", ""),
            }
            for g in guidelines
        ]
        result["parsed_report"] = _parse_report_sections(result.get("radiology_report", ""))
        return result

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"NIVARA inference failed: {exc}") from exc
    finally:
        await file.close()
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        gc.collect()


# Serve compiled React frontend on port 8000 (keep below all /api routes)
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")