import json
import math
import re
from collections import Counter
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_DIR = Path("guidelines_pdfs")
FAISS_INDEX_DIR = Path("nivara_faiss_index")

EMBED_MODEL_NAME = "NeuML/biomedbert-small-embeddings"
OLLAMA_MODEL = "medgemma:4b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Decision thresholds for clinical summary & YOLO-to-finding promotion
YOLO_CONFIRMED_THRESHOLD = 0.45
CIHMLC_CONFIRMED_THRESHOLD = 0.50
CIHMLC_BORDERLINE_THRESHOLD = 0.18

# Maps YOLO26m class names (from both pneumonia & 4-class thoracic detectors) to CIHMLC clinical labels
YOLO_TO_CLINICAL_LABELS: dict[str, list[str]] = {
    "pneumonia_opacity": ["Pneumonia", "Consolidation", "Lung Opacity"],
    "Pneumothorax": ["Pneumothorax"],
    "Pleural Effusion": ["Pleural Effusion"],
    "Cardiomegaly": ["Cardiomegaly", "Enlarged Cardiomediastinum"],
    "Nodule/Mass": ["Lung Lesion", "Lung Opacity"],
}

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


def _is_valid_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def load_or_build_guideline_store() -> FAISS:
    PDF_DIR.mkdir(exist_ok=True)

    if FAISS_INDEX_DIR.exists():
        print(f"Loading cached FAISS guideline index from {FAISS_INDEX_DIR}...")
        return FAISS.load_local(
            str(FAISS_INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    valid_pdfs = [p for p in sorted(PDF_DIR.glob("*.pdf")) if _is_valid_pdf(p)]
    if not valid_pdfs:
        raise FileNotFoundError(f"No valid PDF files found in '{PDF_DIR.resolve()}'.")

    print(f"Parsing and indexing {len(valid_pdfs)} PDF(s) from {PDF_DIR}...")
    raw_pages: list[Document] = []
    for pdf_path in valid_pdfs:
        raw_pages.extend(PyPDFLoader(str(pdf_path)).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=650,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_pages)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(FAISS_INDEX_DIR))
    return store


vector_store = load_or_build_guideline_store()

# keep_alive=0 releases medgemma:4b from 8GB system RAM immediately after generation
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.15,
    repeat_penalty=1.12,
    keep_alive=0,
)

REPORT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a radiology reporting assistant. Convert the patient chest X-ray JSON and retrieved clinical "
        "guideline excerpts into a concise, structured preliminary radiology report.\n"
        "STRICT RULES:\n"
        "1. Write complete clinical sentences—never output bracketed placeholders or raw JSON keys.\n"
        "2. Use ONLY the facts in PATIENT FINDINGS. Never invent unmeasured anatomy (costophrenic angles, vascular cephalization, air bronchograms).\n"
        "3. In FINDINGS:\n"
        "   - Put ONLY lung/pleural findings (including pneumonia, opacity, consolidation, atelectasis, pleural effusion, pneumothorax) under `Lungs & Pleura` (do NOT put cardiomegaly or mediastinum there). Use 'lung zone' (never 'lobe').\n"
        "   - Put ONLY heart/mediastinal findings (including cardiomegaly and enlarged cardiomediastinum) under `Cardiomediastinal Silhouette`.\n"
        "   - Put support devices, fractures, and any pulmonary nodule/mass or lung lesion findings under `Support Devices & Other`.\n"
        "   - State all `present` items definitively as present. State `borderline_not_excluded` items as 'cannot be excluded', and list `excluded_absent` items as absent.\n"
        "4. In IMPRESSION, if Cardiomegaly/Edema co-occur with bilateral mid-to-lower zone opacities, favor cardiogenic "
        "pulmonary edema / fluid overload as the primary impression while naming any borderline pulmonary differentials "
        "(such as consolidation or pneumonia) that cannot be excluded.\n"
        "5. In RECOMMENDATIONS, write 2-3 bullet points translating the RETRIEVED GUIDELINE EXCERPTS below into "
        "actionable diagnostic workup, biomarker tests, severity scoring, or follow-up imaging strictly for the confirmed `present` "
        "findings (or the primary `borderline_not_excluded` findings if no confirmed abnormalities are present; never state a "
        "borderline differential as an established diagnosis). End each bullet with the exact citation `[filename, p. X]`.\n"
        "6. Output ONLY these four sections in exact order:\n\n"
        "**EXAMINATION:** Chest Radiograph (PA/AP)\n\n"
        "**FINDINGS:**\n"
        "- **Lungs & Pleura:** ...\n"
        "- **Cardiomediastinal Silhouette:** ...\n"
        "- **Support Devices & Other:** ...\n\n"
        "**IMPRESSION:**\n"
        "1. ...\n"
        "2. ...\n\n"
        "**RECOMMENDATIONS:**\n"
        "- ...",
    ),
    (
        "user",
        "PATIENT FINDINGS (JSON):\n{clinical_json}\n\n"
        "RETRIEVED GUIDELINE EXCERPTS:\n{guideline_context}\n\n"
        "Generate the preliminary radiology report now:",
    ),
])


def _is_noise_chunk(text: str) -> bool:
    lower = text.lower()
    noise_markers = (
        "author disclosures", "correspondence:", "received:", "accepted:",
        "et al.", "doi:", "pmid:", "all rights reserved", "table of contents",
    )
    if sum(m in lower for m in noise_markers) >= 2:
        return True
    return len(re.findall(r"\(\d{4}\)", text)) >= 4


def _keyword_overlap_bonus(text: str, keywords: list[str]) -> float:
    tokens = Counter(re.findall(r"[a-z0-9\-]{3,}", text.lower()))
    if not tokens:
        return 0.0
    hits = sum(math.log1p(tokens[kw]) for kw in keywords if kw in tokens)
    return min(0.18, hits * 0.03)


def _retrieve_pdf_guidelines(active_present: list[str], active_borderline: list[str], top_k: int = 3) -> list[dict]:
    if not (active_present or active_borderline):
        return []

    findings_str = ", ".join(dict.fromkeys(active_present + active_borderline))
    query = (
        f"Recommended diagnostic evaluation, natriuretic peptide BNP, echocardiography, severity score, "
        f"thoracic ultrasound, chest CT or follow-up chest radiograph for {findings_str}."
    )
    clinical_keywords = re.findall(
        r"[a-z0-9\-]{3,}",
        f"{findings_str} recommend recommendation diagnostic bnp nt-probnp echocardiography ultrasound "
        f"thoracentesis nodule mass fleischner pneumothorax effusion cardiomegaly ct curb-65 psi follow-up radiograph".lower(),
    )

    docs_and_scores = vector_store.similarity_search_with_score(query, k=20)
    scored_candidates = []
    seen = set()

    for doc, l2_dist in docs_and_scores:
        cos_sim = float(1.0 - (l2_dist / 2.0))
        clean_excerpt = " ".join(doc.page_content.split())
        if len(clean_excerpt) < 120 or _is_noise_chunk(clean_excerpt):
            continue

        sig = clean_excerpt[:80]
        if sig in seen:
            continue
        seen.add(sig)

        hybrid_score = round(cos_sim + _keyword_overlap_bonus(clean_excerpt, clinical_keywords), 3)
        if hybrid_score < 0.22:
            continue

        pdf_name = Path(doc.metadata.get("source", "guideline.pdf")).name
        page_num = int(doc.metadata.get("page", 0)) + 1
        scored_candidates.append({
            "citation": f"{pdf_name}, p. {page_num}",
            "score": hybrid_score,
            "hybrid_score": hybrid_score,
            "excerpt": clean_excerpt,
        })

    scored_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
    top_hits = scored_candidates[:top_k]

    print(f"[RAG] Retrieved {len(top_hits)} guideline chunk(s):")
    for hit in top_hits:
        print(f"   -> [{hit['citation']}] (score={hit['hybrid_score']}): {hit['excerpt'][:110]}...")
    return top_hits


def _describe_box_anatomy(det: dict) -> str:
    xywh_norm = det["bbox_xywh_norm"]
    conf = float(det["confidence"])
    class_name = det.get("class_name", "pneumonia_opacity")
    strength = "low-confidence" if conf < 0.25 else "moderate-confidence" if conf < 0.50 else "high-confidence"

    if class_name == "Cardiomegaly":
        return f"cardiomediastinal silhouette [{strength} cardiomegaly localization, score={conf:.2f}]"

    cx, cy, nw, nh = xywh_norm
    side = "right lung" if cx < 0.45 else "left lung" if cx > 0.55 else "central lung region"
    y_top, y_bot = max(0.0, cy - nh / 2.0), min(1.0, cy + nh / 2.0)
    if y_bot < 0.42:
        zone = "upper lung zone"
    elif y_top > 0.58:
        zone = "lower lung zone"
    elif nh >= 0.45:
        zone = "mid-to-lower lung zones"
    else:
        zone = "mid lung zone"

    label_desc = {
        "pneumonia_opacity": "parenchymal opacity localization",
        "Pneumothorax": "pneumothorax localization",
        "Pleural Effusion": "pleural effusion localization",
        "Nodule/Mass": "pulmonary nodule/mass localization",
    }.get(class_name, f"{class_name} localization")

    return f"{side} ({zone}) [{strength} {label_desc}, score={conf:.2f}]"


def build_rag_clinical_summary(nivara_output: dict) -> tuple[dict, list[dict]]:
    scores = nivara_output["cihmlc_scores"]
    detections = nivara_output.get("yolo26_detections", [])

    lung_pleura_labels = [
        "Edema", "Lung Opacity", "Consolidation", "Pneumonia",
        "Atelectasis", "Pleural Effusion", "Pleural Other", "Pneumothorax",
    ]
    cardiac_labels = ["Cardiomegaly", "Enlarged Cardiomediastinum"]
    other_labels = ["Fracture", "Lung Lesion"]

    def split_scores(labels: list[str]) -> tuple[list[str], list[str], list[str]]:
        present = [k for k in labels if scores.get(k, 0.0) >= CIHMLC_CONFIRMED_THRESHOLD]
        borderline = [
            k for k in labels
            if CIHMLC_BORDERLINE_THRESHOLD <= scores.get(k, 0.0) < CIHMLC_CONFIRMED_THRESHOLD
        ]
        absent = [k for k in labels if scores.get(k, 0.0) < 0.05]
        return present, borderline, absent

    lp_present, lp_borderline, lp_absent = split_scores(lung_pleura_labels)
    cm_present, cm_borderline, _ = split_scores(cardiac_labels)
    oth_present, oth_borderline, _ = split_scores(other_labels)

    lp_boxes: list[str] = []
    cm_boxes: list[str] = []
    oth_boxes: list[str] = []
    yolo_query_terms: list[str] = []

    def _promote_label(target_list: list[str], label: str) -> None:
        if label not in target_list:
            target_list.append(label)
        for pool in (lp_borderline, cm_borderline, oth_borderline, lp_absent):
            if label in pool:
                pool.remove(label)

    for det in detections:
        cls_name = det.get("class_name", "pneumonia_opacity")
        conf = float(det.get("confidence", 0.0))
        desc = _describe_box_anatomy(det)

        if cls_name == "Cardiomegaly":
            cm_boxes.append(desc)
            yolo_query_terms.append("Cardiomegaly")
            if conf >= YOLO_CONFIRMED_THRESHOLD:
                _promote_label(cm_present, "Cardiomegaly")

        elif cls_name == "Nodule/Mass":
            oth_boxes.append(desc)
            yolo_query_terms.extend(["Lung Lesion", "pulmonary nodule"])
            if conf >= YOLO_CONFIRMED_THRESHOLD:
                _promote_label(oth_present, "Lung Lesion")

        else:
            lp_boxes.append(desc)
            if cls_name == "Pneumothorax":
                yolo_query_terms.append("Pneumothorax")
                if conf >= YOLO_CONFIRMED_THRESHOLD:
                    _promote_label(lp_present, "Pneumothorax")
            elif cls_name == "Pleural Effusion":
                yolo_query_terms.append("Pleural Effusion")
                if conf >= YOLO_CONFIRMED_THRESHOLD:
                    _promote_label(lp_present, "Pleural Effusion")
            else:
                yolo_query_terms.append("Pneumonia")
                if conf >= YOLO_CONFIRMED_THRESHOLD:
                    _promote_label(lp_present, "Pneumonia")

    retrieved_guidelines = _retrieve_pdf_guidelines(
        lp_present + cm_present + oth_present + yolo_query_terms,
        lp_borderline + cm_borderline + oth_borderline,
        top_k=3,
    )

    support_score = scores.get("Support Devices", 0.0)
    if support_score >= CIHMLC_CONFIRMED_THRESHOLD:
        devices_status = "Support devices present."
    elif support_score >= CIHMLC_BORDERLINE_THRESHOLD:
        devices_status = "Possible support devices (borderline signal)."
    else:
        devices_status = "No support devices detected."

    findings_json = {
        "study": "Chest Radiograph (PA/AP)",
        "lungs_and_pleura": {
            "present": lp_present or ["No high-probability parenchymal or pleural abnormality"],
            "yolo26_localized_regions": lp_boxes or ["None"],
            "borderline_not_excluded": lp_borderline or ["None"],
            "excluded_absent": lp_absent or ["None"],
        },
        "cardiomediastinal_silhouette": {
            "present": cm_present or ["Within normal limits"],
            "yolo26_localized_regions": cm_boxes or ["None"],
            "borderline_not_excluded": cm_borderline or ["None"],
        },
        "devices_and_other": {
            "support_devices": devices_status,
            "present": oth_present or ["None"],
            "yolo26_localized_nodule_mass": oth_boxes or ["None"],
            "borderline_not_excluded": oth_borderline or ["None"],
        },
    }
    return findings_json, retrieved_guidelines


def audit_report_grounding(
    report_text: str,
    cihmlc_scores: dict[str, float],
    retrieved_guidelines: list[dict] | None = None,
    yolo_detections: list[dict] | None = None,
) -> dict:
    monitored = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Pneumothorax", "Pleural Effusion", "Pneumonia",
    ]
    negation_cues = ("no ", "not ", "absent", "excluded", "without ", "negative for", "rule out", "cannot be excluded")
    verified, ungrounded = [], []

    yolo_verified_labels: set[str] = set()
    for det in yolo_detections or []:
        for mapped_label in YOLO_TO_CLINICAL_LABELS.get(det.get("class_name", ""), []):
            yolo_verified_labels.add(mapped_label)

    for label in monitored:
        for match in re.finditer(rf"\b{re.escape(label)}\b", report_text, re.IGNORECASE):
            window = report_text[max(0, match.start() - 45):match.end()].lower()
            if not any(cue in window for cue in negation_cues):
                score = cihmlc_scores.get(label, 0.0)
                if score >= 0.12 or label in yolo_verified_labels:
                    verified.append(label)
                else:
                    ungrounded.append({"label": label, "cihmlc_score": score})

    allowed_citations = {g["citation"] for g in (retrieved_guidelines or [])}
    cited_in_report = re.findall(r"\[([^\[\]]+?\.pdf,\s*p\.\s*\d+)\]", report_text, re.IGNORECASE)
    invalid_citations = [c for c in cited_in_report if c not in allowed_citations]

    total_checks = len(set(verified)) + len(ungrounded) + len(cited_in_report)
    passed_checks = len(set(verified)) + (len(cited_in_report) - len(invalid_citations))

    return {
        "grounding_score": 1.0 if total_checks == 0 else round(passed_checks / total_checks, 3),
        "verified_positive_mentions": sorted(set(verified)),
        "ungrounded_hallucinations": ungrounded,
        "verified_citations": [c for c in cited_in_report if c in allowed_citations],
        "ungrounded_citations": invalid_citations,
    }


def generate_langchain_report(nivara_output: dict) -> tuple[str, dict]:
    try:
        findings_json, retrieved_guidelines = build_rag_clinical_summary(nivara_output)
        nivara_output["retrieved_guidelines"] = retrieved_guidelines

        if retrieved_guidelines:
            guideline_context = "\n\n".join(
                f"[{g['citation']}]\n{g['excerpt']}" for g in retrieved_guidelines
            )
        else:
            guideline_context = "No matching guideline excerpts retrieved; write 'Routine clinical correlation as indicated.' under RECOMMENDATIONS."

        chain = REPORT_PROMPT | llm | StrOutputParser()
        report = chain.invoke({
            "clinical_json": json.dumps(findings_json, indent=2),
            "guideline_context": guideline_context,
        }).strip()

        disclaimer = (
            "**AI DISCLAIMER:** Preliminary AI-generated report for clinical decision support only; "
            "requires confirmation by a qualified radiologist."
        )
        if "AI DISCLAIMER" not in report.upper():
            report = f"{report}\n\n{disclaimer}"

        audit = audit_report_grounding(
            report,
            nivara_output["cihmlc_scores"],
            retrieved_guidelines,
            nivara_output.get("yolo26_detections", []),
        )
        return report, audit
    except Exception as exc:
        err = f"[LangChain/Ollama unavailable at {OLLAMA_BASE_URL} ({exc}). Start Ollama or verify '{OLLAMA_MODEL}'.]"
        return err, {"grounding_score": 0.0, "verified_positive_mentions": [], "ungrounded_hallucinations": []}
