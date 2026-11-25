# NIVARA

**NIVARA** is a locally deployable chest X-ray screening and reporting workstation developed by our team as a research-oriented hackathon project. It brings together a multi-label classification model, two gated YOLO26 detection pipelines, Grad-CAM visualisation, clinical guideline retrieval, and a locally hosted language model into a unified React-based interface.

The system accepts chest X-ray images and DICOM studies, analyses them for a range of thoracic abnormalities, conditionally routes suspicious findings through specialised detection models, and generates a structured preliminary report supported by retrieved clinical guideline references.

> **Clinical Disclaimer:** NIVARA is a research and demonstration system, not a diagnostic medical device. It is not intended to replace a radiologist or any qualified healthcare professional. All generated findings and reports should be treated as preliminary decision-support outputs and independently reviewed alongside the complete imaging study and relevant clinical context.

## What our team developed

- **React/Vite clinician workspace:** An interface for uploading studies, reviewing findings, comparing images, signing reports, and exporting report text.
- **FastAPI backend:** A backend serving the inference pipeline, API endpoints, and compiled frontend.
- **Multi-label chest X-ray classifier:** A model capable of predicting 21 supported labels, including pneumonia, pulmonary edema, cardiomegaly, pleural effusion, pneumothorax, consolidation, and atelectasis.
- **Conditional YOLO26 detection pipelines:** Two specialised detectors activated through classifier-based routing thresholds. One focuses on pneumonia and opacity-related findings, while the other covers pneumothorax, pleural effusion, cardiomegaly, and lung lesions.
- **Grad-CAM visualisation:** Visual overlays highlighting image regions influencing selected classifier predictions.
- **Retrieval-Augmented Generation (RAG):** Clinical guideline retrieval using PDF documents, BiomedBERT embeddings, FAISS, and a locally hosted MedGemma model through Ollama.
- **Grounded report generation:** Structured report sections covering findings, impression, recommendations, and supporting guideline references, alongside a visible safety disclaimer.

## How the analysis pipeline works

1. **Image ingestion:** The system accepts an uploaded X-ray image or DICOM study and converts the image into a normalised RGB representation for processing.
2. **Multi-label classification:** The CIHMLC classifier generates prediction scores across its 21 supported labels.
3. **Conditional model routing:** Predictions are evaluated against configured routing thresholds to determine whether specialised detection pipelines should be activated.
4. **Object detection:** YOLO26 models identify and localise relevant abnormalities, with detection outputs mapped to lung zones or anatomical locations where supported.
5. **Visual explainability:** Grad-CAM generates visual overlays corresponding to selected classifier targets.
6. **Guideline retrieval:** Relevant excerpts are retrieved from the locally indexed clinical guideline documents.
7. **Report generation:** MedGemma, operating through a LangChain-based RAG pipeline, generates a structured preliminary report grounded in retrieved references.

The inference routing thresholds are configured in `finalnewinf.py`. The pneumonia detection route activates at a classifier threshold of `0.05`, while the four-class thoracic detection route activates at `0.15`. The frontend additionally assigns a higher-priority state to findings when classifier confidence reaches `0.5`.

## Technology stack

### Backend and AI/ML

- Python
- FastAPI and Uvicorn
- TensorFlow / Keras
- PyTorch
- Ultralytics YOLO26
- OpenCV
- Pillow
- NumPy
- pydicom
- LangChain
- FAISS
- Hugging Face Transformers and BiomedBERT embeddings
- Ollama
- MedGemma 4B

### Frontend

- React 19
- Vite 8
- Tailwind CSS 4
- Lucide Icons
- oxlint

## Project structure

The project is organised around the following core components:

- `api.py` — Combined FastAPI entry point serving `/api/infer` and the built frontend.
- `finalnewinf.py` — Classifier inference, detector routing, Grad-CAM generation, and result assembly.
- `nivara_langchain.py` — Guideline retrieval and local MedGemma report-generation pipeline.
- `Nivara_Models/` — YOLO model weight files.
- `CIHMLC/src/assets/models/default_model.keras` — Local classifier weights.
- `guidelines_pdfs/` — Source clinical guideline PDFs.
- `nivara_faiss_index/` — Cached FAISS guideline vector index.
- `fullstack/nivara/` — React application source.

Some model paths in `finalnewinf.py` are currently configured for the original development environment. When deploying the project on another machine, the following environment variables or path configurations need to be updated accordingly:

- `CIHMLC_WEIGHTS`
- `YOLO26_PNEUMONIA_WEIGHTS`
- `YOLO26_THORACIC4_WEIGHTS`

These should point to the corresponding locally available model files.

## Running NIVARA locally

The project was developed primarily in a Linux environment using a Python virtual environment. The following instructions assume the repository is located at `/home/voyager/Downloads/Nivara`.

### 1. Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies required by the inference and web components, including TensorFlow, PyTorch, Ultralytics, FastAPI, Uvicorn, OpenCV, Pillow, NumPy, pydicom, LangChain, FAISS, Hugging Face Transformers, and the Ollama Python integration.

### 2. Start Ollama

The report-generation pipeline uses a locally hosted MedGemma model.

```bash
ollama pull medgemma:4b
ollama serve
```

The existing FAISS index is loaded during the initialisation of `nivara_langchain.py`. If the index is unavailable or invalid, the guideline PDFs should be placed in `guidelines_pdfs/` so the application can rebuild the index.

### 3. Build the frontend

```bash
cd fullstack/nivara
npm install
npm run build
cd ../..
```

### 4. Start the combined application

```bash
source .venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Once the server is running, the application can be accessed at:

`http://localhost:8000`

During development, the React frontend can also be started independently using Vite:

```bash
cd fullstack/nivara
npm run dev
```

The Vite development server is configured to communicate with the local FastAPI backend. Its development URL is typically:

`http://localhost:5173`

## API endpoints

### Health check

```http
GET /api/health
```

### Run inference

```http
POST /api/infer
Content-Type: multipart/form-data
```

The uploaded file is passed through the `file` field.

The response includes classifier scores, routing decisions, YOLO detection outputs, the Grad-CAM target, processing timings, retrieved guideline citations, and a structured report representation.

The repository also contains a smaller backend under `backend/app/` with `/health`, `/predict`, and `/compare` endpoints for DICOM validation and previous-versus-current study comparison. This backend is separate from the combined `api.py` entry point and can be launched independently from its application directory.

## Demonstration accounts

The React workspace includes demonstration credentials intended for interface testing:

- `dr.rao@nivara.health` / `nivara2026`
- `dr.sen@nivara.health` / `nivara2026`

These accounts are intended exclusively for demonstration purposes and do not constitute production authentication.

## Development and testing

Frontend validation can be performed using the following commands:

```bash
cd fullstack/nivara
npm run lint
npm run build
```

The Python components primarily consist of inference and experimentation code. Datasets, DICOM studies, trained weights, evaluation outputs, and archives are maintained outside the source repository wherever possible and excluded through `.gitignore`.

## Privacy and safety considerations

NIVARA's report-generation pipeline is designed to operate using a locally hosted language model through Ollama, rather than sending inference requests to a hosted LLM by default. The guideline retrieval index is also generated from locally stored PDF documents.

However, local inference alone does not guarantee patient-data security. Any deployment involving real medical studies would require appropriate access controls, audit logging, secure data storage, retention policies, and a validated privacy and security framework.

## Current limitations

- The pipeline depends on large locally stored model weights and an Ollama runtime.
- Several model paths remain tied to the original development workspace.
- The DICOM ingestion and preprocessing pipeline requires validation across different institutional export formats.
- YOLO detection is conditional on classifier thresholds, meaning low classifier scores may prevent downstream detections.
- MedGemma may be unavailable or slow on systems with limited memory.
- The frontend contains realistic demonstration studies, which are replaced by inference results when studies are uploaded.
- The current implementation does not include production-grade authentication or a database-backed patient management workflow.

## Project status

NIVARA is a working research prototype developed by our team to explore chest X-ray screening workflows, multi-model inference routing, visual explainability, local retrieval-augmented reporting, and clinician-facing software design.

As a hackathon-stage project, it represents an experimental integration of several AI and software engineering components into a single workstation. Further independent model evaluation, clinical validation, robustness testing, formal safety review, and regulatory assessment would be required before considering deployment in a clinical setting.
