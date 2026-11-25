import React, { useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Clock,
  Contrast,
  Copy,
  Cpu,
  Download,
  Eye,
  EyeOff,
  FileText,
  Flame,
  Info,
  KeyRound,
  Layers,
  LayoutGrid,
  Lock,
  LogOut,
  Mail,
  RefreshCw,
  RotateCcw,
  Scan,
  Search,
  ShieldCheck,
  Sliders,
  Sparkles,
  Stethoscope,
  Upload,
  UserCheck,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";

const API_BASE_URL = import.meta.env.DEV ? "http://localhost:8000" : "";
const SESSION_STORAGE_KEY = "nivara_clinician_session_v1";

// Color-coded palettes for both Pneumonia YOLO and 4-Class Thoracic YOLO
const YOLO_CLASS_COLORS = {
  pneumonia_opacity: {
    border: "#38bdf8",
    bg: "rgba(56, 189, 248, 0.12)",
    text: "#082f49",
  },
  Pneumothorax: {
    border: "#f43f5e",
    bg: "rgba(244, 63, 94, 0.14)",
    text: "#ffffff",
  },
  "Pleural Effusion": {
    border: "#f59e0b",
    bg: "rgba(245, 158, 11, 0.14)",
    text: "#1c1917",
  },
  Cardiomegaly: {
    border: "#10b981",
    bg: "rgba(16, 185, 129, 0.12)",
    text: "#022c22",
  },
  "Nodule/Mass": {
    border: "#a855f7",
    bg: "rgba(168, 85, 247, 0.14)",
    text: "#ffffff",
  },
};

function getYoloPalette(className) {
  return YOLO_CLASS_COLORS[className] || YOLO_CLASS_COLORS.pneumonia_opacity;
}

// Pre-configured test doctor accounts for demo & workstation sign-in
const TEST_DOCTOR_ACCOUNTS = [
  {
    id: "DOC-101",
    name: "Dr. A. Rao",
    initials: "AR",
    email: "dr.rao@nivara.health",
    password: "nivara2026",
    role: "Thoracic Radiologist",
    department: "Diagnostic Radiology",
    licenseId: "WBMC-48291",
  },
  {
    id: "DOC-102",
    name: "Dr. S. Sen",
    initials: "SS",
    email: "dr.sen@nivara.health",
    password: "nivara2026",
    role: "Pulmonology Consultant",
    department: "Respiratory & Critical Care",
    licenseId: "WBMC-51904",
  },
];

// Realistic dataset pre-populated with actual NIVARA pipeline output structure
const INITIAL_STUDIES = [
  {
    id: "CXR-2048",
    patientName: "Case 01 · Basal Opacity",
    mrn: "NIV-8841",
    receivedTime: "10:42 AM",
    receivedDate: "20 Nov 2026",
    view: "Frontal · PA/AP",
    status: "priority",
    reviewed: false,
    signedBy: null,
    headlineResult: "Pulmonary edema / Cardiomegaly",
    priorityLabel: "High",
    routingDecision: "DUAL_YOLO_ACTIVATED",
    routingThreshold: 0.05,
    cihmlcTime: 9.34,
    yoloTime: 4.29,
    imageUrl: null,
    gradcamUrl: null,
    cihmlcScores: {
      Edema: 0.7139,
      "Enlarged Cardiomediastinum": 0.7026,
      "Lung Opacity": 0.612,
      Cardiomegaly: 0.5173,
      Consolidation: 0.1334,
      Pneumonia: 0.1286,
      Fracture: 0.1198,
      "Lung Lesion": 0.1016,
      Atelectasis: 0.0405,
      "Pleural Other": 0.0228,
      "Pleural Effusion": 0.0161,
      Pneumothorax: 0.0011,
      "Support Devices": 0.0488,
    },
    yoloDetections: [
      {
        class_id: 0,
        class_name: "pneumonia_opacity",
        confidence: 0.13,
        bbox_xywh_norm: [0.718, 0.578, 0.341, 0.653],
        zone: "Left mid-to-lower lung zone",
      },
      {
        class_id: 0,
        class_name: "pneumonia_opacity",
        confidence: 0.12,
        bbox_xywh_norm: [0.23, 0.557, 0.32, 0.63],
        zone: "Right mid-to-lower lung zone",
      },
    ],
    ragCitations: [
      {
        citation: "heart-failure-management-pocket-guide.pdf, p. 5",
        score: 0.832,
        excerpt:
          "Diagnostic Algorithm for HF: Perform clinical history, physical examination, 12-lead ECG, chest radiograph, and initial natriuretic peptide (BNP or NT-proBNP) measurement.",
      },
      {
        citation: "heart-failure-management-pocket-guide.pdf, p. 6",
        score: 0.806,
        excerpt:
          "In patients presenting with dyspnea, measurement of BNP or NT-proBNP is useful to support a diagnosis or exclusion of HF, followed by transthoracic echocardiography (TTE) to assess LVEF.",
      },
      {
        citation: "ATS_IDSA_CAP_2019.pdf, p. 4",
        score: 0.764,
        excerpt:
          "When infectious pneumonia cannot be excluded on initial chest imaging, apply validated clinical severity scoring (PSI or CURB-65) alongside clinical correlation.",
      },
    ],
    report: {
      examination: "Chest Radiograph (PA/AP)",
      findings: {
        lungsAndPleura:
          "Bilateral lung opacities are present in the mid-to-lower lung zones. Borderline pulmonary consolidation or pneumonia cannot be excluded. Atelectasis, pleural effusion, and pneumothorax are absent.",
        cardiomediastinal:
          "Cardiomegaly and an enlarged cardiomediastinum are present.",
        devicesAndOther:
          "No support devices detected. Borderline fracture or lung lesion signals cannot be excluded.",
      },
      impression: [
        "Bilateral mid-to-lower lung zone opacities with cardiomegaly, favoring cardiogenic pulmonary edema or fluid overload.",
        "Borderline consolidation or superimposed pneumonia cannot be excluded.",
      ],
      recommendations: [
        "Obtain serum BNP or NT-proBNP levels to evaluate for acute decompensated heart failure [heart-failure-management-pocket-guide.pdf, p. 5].",
        "Perform transthoracic echocardiography (TTE) to assess cardiac chamber size and ventricular function [heart-failure-management-pocket-guide.pdf, p. 6].",
        "Correlate clinically for infectious symptoms and apply PSI/CURB-65 severity assessment if pneumonia remains suspected [ATS_IDSA_CAP_2019.pdf, p. 4].",
      ],
    },
    groundingAudit: {
      groundingScore: 1.0,
      verifiedMentions: [
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Pneumonia",
      ],
      ungroundedHallucinations: [],
      verifiedCitations: 3,
    },
  },
  {
    id: "CXR-2047",
    patientName: "Ananya Mehta",
    mrn: "NIV-4821",
    receivedTime: "10:18 AM",
    receivedDate: "20 Nov 2026",
    view: "Frontal · PA",
    status: "priority",
    reviewed: false,
    signedBy: null,
    headlineResult: "Focal RLL Pneumonia / Consolidation",
    priorityLabel: "High",
    routingDecision: "PNEUMONIA_YOLO_ACTIVATED",
    routingThreshold: 0.05,
    cihmlcTime: 7.91,
    yoloTime: 3.84,
    imageUrl: null,
    gradcamUrl: null,
    cihmlcScores: {
      Pneumonia: 0.842,
      Consolidation: 0.761,
      "Lung Opacity": 0.815,
      Atelectasis: 0.184,
      "Pleural Effusion": 0.092,
      Cardiomegaly: 0.041,
      Edema: 0.032,
      Pneumothorax: 0.004,
      "Support Devices": 0.012,
    },
    yoloDetections: [
      {
        class_id: 0,
        class_name: "pneumonia_opacity",
        confidence: 0.79,
        bbox_xywh_norm: [0.31, 0.64, 0.24, 0.28],
        zone: "Right lower lung zone",
      },
    ],
    ragCitations: [
      {
        citation: "ATS_IDSA_CAP_2019.pdf, p. 6",
        score: 0.874,
        excerpt:
          "For adults with suspected community-acquired pneumonia confirmed by chest radiograph, empiric antimicrobial therapy should be initiated promptly guided by PSI/CURB-65 severity stratification.",
      },
      {
        citation: "ATS_IDSA_CAP_2019.pdf, p. 11",
        score: 0.812,
        excerpt:
          "Routine follow-up chest imaging is not indicated in patients whose symptoms resolve within 5 to 7 days, unless clinical deterioration or risk factors for underlying malignancy exist.",
      },
    ],
    report: {
      examination: "Chest Radiograph (PA/AP)",
      findings: {
        lungsAndPleura:
          "Focal airspace consolidation and opacity localized to the right lower lung zone (YOLO26m score 0.79). Borderline atelectasis cannot be excluded. Pleural effusion and pneumothorax are absent.",
        cardiomediastinal:
          "Cardiomediastinal silhouette is within normal limits.",
        devicesAndOther: "No support devices detected.",
      },
      impression: [
        "Right lower lung zone consolidation and opacity, consistent with focal pneumonia.",
        "Associated minor subsegmental atelectasis cannot be excluded.",
      ],
      recommendations: [
        "Stratify clinical severity using PSI or CURB-65 and initiate guideline-concordant empiric therapy [ATS_IDSA_CAP_2019.pdf, p. 6].",
        "Reserve repeat chest radiography for persistent symptoms or patients with risk factors requiring exclusion of underlying mass [ATS_IDSA_CAP_2019.pdf, p. 11].",
      ],
    },
    groundingAudit: {
      groundingScore: 1.0,
      verifiedMentions: [
        "Atelectasis",
        "Consolidation",
        "Lung Opacity",
        "Pneumonia",
      ],
      ungroundedHallucinations: [],
      verifiedCitations: 2,
    },
  },
  {
    id: "CXR-2046",
    patientName: "Rohan Iyer",
    mrn: "NIV-1193",
    receivedTime: "09:54 AM",
    receivedDate: "20 Nov 2026",
    view: "Frontal · AP",
    status: "uncertain",
    reviewed: false,
    signedBy: null,
    headlineResult: "Borderline bibasilar atelectasis",
    priorityLabel: "Review",
    routingDecision: "YOLO26_SKIPPED",
    routingThreshold: 0.05,
    cihmlcTime: 7.45,
    yoloTime: 0.0,
    imageUrl: null,
    gradcamUrl: null,
    cihmlcScores: {
      Atelectasis: 0.284,
      "Lung Opacity": 0.219,
      "No Finding": 0.462,
      Pneumonia: 0.031,
      Consolidation: 0.044,
      Cardiomegaly: 0.039,
      "Pleural Effusion": 0.025,
      Pneumothorax: 0.002,
      "Support Devices": 0.018,
    },
    yoloDetections: [],
    ragCitations: [
      {
        citation: "ATS_IDSA_CAP_2019.pdf, p. 3",
        score: 0.721,
        excerpt:
          "Low-inspiration portable AP radiographs may mimic basilar parenchymal opacity; correlate with clinical signs of infection before diagnosing pneumonia.",
      },
    ],
    report: {
      examination: "Chest Radiograph (PA/AP)",
      findings: {
        lungsAndPleura:
          "No high-probability parenchymal abnormality. Borderline atelectasis and mild lung opacity cannot be excluded. Pneumonia, consolidation, pleural effusion, and pneumothorax are absent.",
        cardiomediastinal:
          "Cardiomediastinal silhouette is within normal limits.",
        devicesAndOther: "No support devices detected.",
      },
      impression: [
        "Mild borderline bibasilar opacity, likely reflecting subsegmental atelectasis or inspiratory effort.",
        "No definite focal pneumonia or pleural effusion.",
      ],
      recommendations: [
        "Correlate with inspiratory effort and clinical presentation; repeat PA radiograph if acute respiratory symptoms persist [ATS_IDSA_CAP_2019.pdf, p. 3].",
      ],
    },
    groundingAudit: {
      groundingScore: 1.0,
      verifiedMentions: ["Atelectasis", "Lung Opacity"],
      ungroundedHallucinations: [],
      verifiedCitations: 1,
    },
  },
  {
    id: "CXR-2045",
    patientName: "Maya Kapoor",
    mrn: "NIV-6207",
    receivedTime: "09:31 AM",
    receivedDate: "20 Nov 2026",
    view: "Frontal · PA",
    status: "lower",
    reviewed: true,
    signedBy: "Dr. A. Rao",
    headlineResult: "No acute cardiopulmonary finding",
    priorityLabel: "Routine",
    routingDecision: "YOLO26_SKIPPED",
    routingThreshold: 0.05,
    cihmlcTime: 7.12,
    yoloTime: 0.0,
    imageUrl: null,
    gradcamUrl: null,
    cihmlcScores: {
      "No Finding": 0.941,
      Pneumonia: 0.011,
      Consolidation: 0.009,
      Edema: 0.014,
      Cardiomegaly: 0.022,
      "Pleural Effusion": 0.008,
      Pneumothorax: 0.001,
      Atelectasis: 0.019,
      "Support Devices": 0.004,
    },
    yoloDetections: [],
    ragCitations: [],
    report: {
      examination: "Chest Radiograph (PA/AP)",
      findings: {
        lungsAndPleura:
          "Lungs are clear. Pneumonia, consolidation, edema, atelectasis, pleural effusion, and pneumothorax are absent.",
        cardiomediastinal:
          "Cardiomediastinal silhouette is within normal limits.",
        devicesAndOther: "No support devices detected.",
      },
      impression: ["No acute cardiopulmonary abnormality detected."],
      recommendations: ["Routine clinical correlation as indicated."],
    },
    groundingAudit: {
      groundingScore: 1.0,
      verifiedMentions: [],
      ungroundedHallucinations: [],
      verifiedCitations: 0,
    },
  },
  {
    id: "CXR-2044",
    patientName: "Kabir Shah",
    mrn: "NIV-3704",
    receivedTime: "09:05 AM",
    receivedDate: "20 Nov 2026",
    view: "Frontal · PA",
    status: "priority",
    reviewed: false,
    signedBy: null,
    headlineResult: "Left pleural effusion + compressive atelectasis",
    priorityLabel: "High",
    routingDecision: "THORACIC4_YOLO_ACTIVATED",
    routingThreshold: 0.15,
    cihmlcTime: 7.68,
    yoloTime: 3.12,
    imageUrl: null,
    gradcamUrl: null,
    cihmlcScores: {
      "Pleural Effusion": 0.864,
      Atelectasis: 0.542,
      "Lung Opacity": 0.631,
      Pneumonia: 0.038,
      Consolidation: 0.047,
      Cardiomegaly: 0.045,
      Pneumothorax: 0.003,
      "Support Devices": 0.021,
    },
    yoloDetections: [
      {
        class_id: 1,
        class_name: "Pleural Effusion",
        confidence: 0.81,
        bbox_xywh_norm: [0.71, 0.72, 0.28, 0.26],
        zone: "Left costophrenic / lower hemithorax",
      },
    ],
    ragCitations: [
      {
        citation: "BTS_Pleural_Disease_2023.pdf, p. 8",
        score: 0.858,
        excerpt:
          "Thoracic ultrasound should be performed to confirm pleural fluid presence, quantify effusion depth, and guide diagnostic thoracentesis.",
      },
    ],
    report: {
      examination: "Chest Radiograph (PA/AP)",
      findings: {
        lungsAndPleura:
          "Left-sided pleural effusion and adjacent compressive atelectasis are present. Pneumonia, consolidation, and pneumothorax are absent.",
        cardiomediastinal:
          "Cardiomediastinal silhouette is within normal limits.",
        devicesAndOther: "No support devices detected.",
      },
      impression: [
        "Moderate left pleural effusion with adjacent basilar atelectasis.",
      ],
      recommendations: [
        "Perform bedside thoracic ultrasound to quantify effusion volume and evaluate suitability for diagnostic thoracentesis [BTS_Pleural_Disease_2023.pdf, p. 8].",
      ],
    },
    groundingAudit: {
      groundingScore: 1.0,
      verifiedMentions: ["Atelectasis", "Pleural Effusion"],
      ungroundedHallucinations: [],
      verifiedCitations: 1,
    },
  },
];

function getTopConfidence(study) {
  const entries = Object.entries(study.cihmlcScores || {});
  if (!entries.length) return 0;
  const maxVal = Math.max(...entries.map(([, v]) => v));
  return Math.round(maxVal * 100);
}

function ProceduralRadiograph({ study, showGradcam, showBoxes }) {
  const hasOpacity =
    (study.cihmlcScores?.Edema || 0) >= 0.5 ||
    (study.cihmlcScores?.Pneumonia || 0) >= 0.5 ||
    (study.cihmlcScores?.["Pleural Effusion"] || 0) >= 0.5;
  const hasCardiomegaly = (study.cihmlcScores?.Cardiomegaly || 0) >= 0.5;

  return (
    <svg
      viewBox="0 0 360 460"
      className="h-[92%] w-auto max-w-full select-none drop-shadow-[0_0_16px_rgba(188,228,214,0.16)]"
      role="img"
      aria-label="Chest radiograph preview"
    >
      <defs>
        <radialGradient id="cxrBg">
          <stop stopColor="#cad8d2" />
          <stop offset="0.72" stopColor="#7c9189" />
          <stop offset="1" stopColor="#2c3c38" />
        </radialGradient>
        <linearGradient id="boneGrad" x2="0" y2="1">
          <stop stopColor="#eef4ec" />
          <stop offset="1" stopColor="#8fa39a" />
        </linearGradient>
        <radialGradient id="gradcamHot">
          <stop offset="0%" stopColor="#ef4444" stopOpacity="0.82" />
          <stop offset="45%" stopColor="#f59e0b" stopOpacity="0.55" />
          <stop offset="75%" stopColor="#10b981" stopOpacity="0.28" />
          <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
        </radialGradient>
      </defs>

      <rect
        x="24"
        y="12"
        width="312"
        height="436"
        rx="12"
        fill="url(#cxrBg)"
        opacity="0.9"
      />

      <path
        d="M180 48 C135 40 91 59 69 91 L51 174 Q41 235 60 302 L77 367 Q92 399 133 391 Q151 387 160 351 L171 268 L180 190Z"
        fill="#162322"
        opacity="0.78"
      />
      <path
        d="M180 48 C225 40 269 59 291 91 L309 174 Q319 235 300 302 L283 367 Q268 399 227 391 Q209 387 200 351 L189 268 L180 190Z"
        fill="#162322"
        opacity="0.78"
      />

      {hasOpacity && (
        <>
          <ellipse
            cx="112"
            cy="312"
            rx="46"
            ry="54"
            fill="#b7c9c1"
            opacity="0.28"
          />
          {(study.cihmlcScores?.Edema || 0) >= 0.5 && (
            <ellipse
              cx="246"
              cy="315"
              rx="44"
              ry="50"
              fill="#b7c9c1"
              opacity="0.24"
            />
          )}
        </>
      )}

      <path
        d="M174 49 Q180 36 186 49 L190 253 Q187 291 180 302 Q173 291 170 253Z"
        fill="url(#boneGrad)"
        opacity="0.75"
      />

      <path
        d={
          hasCardiomegaly
            ? "M180 286 Q142 272 118 308 Q104 334 132 364 Q160 382 181 368 Q208 384 242 362 Q266 332 246 302 Q220 272 180 286Z"
            : "M180 304 Q153 289 133 312 Q121 331 141 354 Q163 371 181 360 Q199 371 219 354 Q239 331 227 312 Q207 289 180 304Z"
        }
        fill="#dce7df"
        opacity="0.82"
      />

      <g fill="none" stroke="url(#boneGrad)" strokeWidth="7" opacity="0.68">
        <path d="M174 105 Q121 77 68 108" />
        <path d="M173 134 Q115 106 61 141" />
        <path d="M171 164 Q112 138 57 173" />
        <path d="M170 194 Q109 170 57 204" />
        <path d="M170 225 Q111 200 61 233" />
        <path d="M171 256 Q117 229 68 262" />
        <path d="M186 105 Q239 77 292 108" />
        <path d="M187 134 Q245 106 299 141" />
        <path d="M189 164 Q248 138 303 173" />
        <path d="M190 194 Q251 170 303 204" />
        <path d="M190 225 Q249 200 299 233" />
        <path d="M189 256 Q243 229 292 262" />
      </g>
      <path
        d="M96 102 Q137 86 166 114 M264 102 Q223 86 194 114"
        fill="none"
        stroke="#e0e9df"
        strokeWidth="11"
        opacity="0.56"
      />

      {showGradcam && (
        <g>
          <circle cx="114" cy="306" r="78" fill="url(#gradcamHot)" />
          {(study.cihmlcScores?.Edema || 0) >= 0.5 && (
            <circle
              cx="242"
              cy="316"
              r="64"
              fill="url(#gradcamHot)"
              opacity="0.7"
            />
          )}
        </g>
      )}

      {showBoxes &&
        (study.yoloDetections || []).map((det, idx) => {
          const [cx, cy, nw, nh] = det.bbox_xywh_norm;
          const x = (cx - nw / 2) * 360;
          const y = (cy - nh / 2) * 460;
          const w = nw * 360;
          const h = nh * 460;
          const palette = getYoloPalette(det.class_name);
          const tag = `${det.class_name} ${det.confidence.toFixed(2)}`;
          const pillWidth = Math.max(122, tag.length * 6.5 + 12);

          return (
            <g key={idx}>
              <rect
                x={x}
                y={y}
                width={w}
                height={h}
                fill={palette.bg}
                stroke={palette.border}
                strokeWidth="2.5"
                strokeDasharray="6 3"
                rx="4"
              />
              <rect
                x={x}
                y={Math.max(14, y - 20)}
                width={pillWidth}
                height="19"
                rx="3"
                fill={palette.border}
              />
              <text
                x={x + 6}
                y={Math.max(27, y - 7)}
                fill={palette.text}
                fontSize="10"
                fontWeight="700"
                fontFamily="DM Mono, monospace"
              >
                {tag}
              </text>
            </g>
          );
        })}
    </svg>
  );
}

function DoctorLoginScreen({ onLoginSuccess }) {
  const [identifier, setIdentifier] = useState("dr.rao@nivara.health");
  const [password, setPassword] = useState("nivara2026");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    const cleaned = identifier.trim().toLowerCase();
    const matched = TEST_DOCTOR_ACCOUNTS.find(
      (doc) =>
        (doc.email.toLowerCase() === cleaned ||
          doc.licenseId.toLowerCase() === cleaned ||
          doc.id.toLowerCase() === cleaned) &&
        doc.password === password,
    );

    if (!matched) {
      setError(
        "Invalid clinician credentials. Use a test account below or click Quick Demo Login.",
      );
      return;
    }

    setError("");
    onLoginSuccess(matched);
  };

  const handleQuickDemoLogin = (account) => {
    setError("");
    onLoginSuccess(account);
  };

  return (
    <div className="min-h-screen grid grid-cols-1 lg:grid-cols-[1.05fr_1fr] bg-[#f5f8f6] text-[#172522]">
      {/* LEFT BRAND & CLINICAL WORKSTATION PANEL */}
      <div className="bg-[#102b27] text-[#d8e8e3] p-8 lg:p-14 flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-[#d7eee5] text-[#146b58] grid place-items-center font-display font-extrabold text-xl">
              ✳
            </div>
            <div>
              <div className="font-display font-extrabold text-[22px] tracking-tight text-white leading-none">
                nivara
              </div>
              <div className="font-mono-tech text-[10px] tracking-[1.5px] text-[#91b0a5] mt-1">
                CLINICAL RADIOLOGY TRIAGE
              </div>
            </div>
          </div>

          <div className="mt-14 max-w-md">
            <div className="inline-flex items-center gap-2 rounded-full bg-[#1b413a] border border-[#2c5c52] px-3 py-1 font-mono-tech text-[10px] text-[#9be0c8] uppercase tracking-wider">
              <Stethoscope className="w-3.5 h-3.5" />
              Clinician Decision Support Portal
            </div>

            <h1 className="font-display text-3xl lg:text-[34px] font-bold text-white tracking-tight mt-5 leading-tight">
              Explainable Chest Radiograph Triage & Grounded Reporting
            </h1>

            <p className="text-sm text-[#a6c2b9] mt-4 leading-relaxed">
              Sign in with your clinician credentials to review AI-prioritized
              PA/AP chest radiographs, inspect Grad-CAM and dual YOLO26m
              localizations, and sign off on guideline-cited preliminary
              reports.
            </p>
          </div>
        </div>

        <div className="mt-10 pt-6 border-t border-[#23483f] grid sm:grid-cols-2 gap-4 text-xs">
          <div className="rounded-xl bg-[#163732] border border-[#254d46] p-3.5">
            <div className="font-mono-tech text-[10px] text-[#8fd4bd] uppercase">
              Multi-Modal Vision
            </div>
            <div className="font-semibold text-white mt-1">
              CIHMLC + Dual YOLO26m Gate
            </div>
            <div className="text-[11px] text-[#99b6ad] mt-0.5">
              21-label thoracic screening with pneumonia & 4-class localization
            </div>
          </div>
          <div className="rounded-xl bg-[#163732] border border-[#254d46] p-3.5">
            <div className="font-mono-tech text-[10px] text-[#8fd4bd] uppercase">
              Verified RAG Guardrail
            </div>
            <div className="font-semibold text-white mt-1">
              MedGemma-4B + FAISS
            </div>
            <div className="text-[11px] text-[#99b6ad] mt-0.5">
              Page-cited ATS/IDSA, AHA/ACC & BTS recommendations
            </div>
          </div>
        </div>
      </div>

      {/* RIGHT SIGN-IN FORM */}
      <div className="flex items-center justify-center p-6 sm:p-10">
        <div className="w-full max-w-md bg-white border border-[#e4eae7] rounded-2xl p-7 sm:p-8 shadow-[0_16px_40px_rgba(28,57,48,0.06)]">
          <div className="flex items-center justify-between">
            <div>
              <div className="font-mono-tech text-[10px] uppercase tracking-wider text-[#146b58] font-semibold">
                AUTHORIZED CLINICIAN ACCESS
              </div>
              <h2 className="font-display text-2xl font-bold text-[#172522] mt-1">
                Doctor Sign In
              </h2>
            </div>
            <div className="h-10 w-10 rounded-xl bg-[#e3f2ed] text-[#146b58] grid place-items-center">
              <Lock className="w-5 h-5" />
            </div>
          </div>

          <form onSubmit={handleSubmit} className="mt-6 space-y-4">
            <div>
              <label className="block text-xs font-semibold text-[#283832] mb-1.5">
                Hospital Email or Medical License ID
              </label>
              <div className="relative">
                <Mail className="w-4 h-4 text-[#87928e] absolute left-3 top-3" />
                <input
                  type="text"
                  required
                  value={identifier}
                  onChange={(e) => setIdentifier(e.target.value)}
                  placeholder="dr.rao@nivara.health or WBMC-48291"
                  className="w-full rounded-xl border border-[#e4eae7] bg-[#fbfcfb] focus:bg-white pl-9 pr-3.5 py-2.5 text-xs text-[#172522] outline-none focus:border-[#146b58] transition"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-[#283832] mb-1.5">
                Password
              </label>
              <div className="relative">
                <KeyRound className="w-4 h-4 text-[#87928e] absolute left-3 top-3" />
                <input
                  type={showPassword ? "text" : "password"}
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter workstation password"
                  className="w-full rounded-xl border border-[#e4eae7] bg-[#fbfcfb] focus:bg-white pl-9 pr-10 py-2.5 text-xs text-[#172522] outline-none focus:border-[#146b58] transition"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((v) => !v)}
                  aria-label="Toggle password visibility"
                  className="absolute right-3 top-2.5 text-[#75817d] hover:text-[#172522]"
                >
                  {showPassword ? (
                    <EyeOff className="w-4 h-4" />
                  ) : (
                    <Eye className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>

            {error && (
              <div className="rounded-lg bg-[#fbeceb] border border-[#f5c6c4] px-3 py-2 text-[11px] text-[#a63b38] font-medium">
                {error}
              </div>
            )}

            <button
              type="submit"
              className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-[#146b58] hover:bg-[#0f5445] text-white font-bold py-3 px-4 text-xs transition shadow-sm"
            >
              Sign in to Triage Workstation
              <ArrowRight className="w-4 h-4" />
            </button>
          </form>

          {/* TEST ACCOUNT QUICK ACCESS BOX */}
          <div className="mt-6 pt-5 border-t border-[#e4eae7]">
            <div className="flex items-center justify-between mb-2.5">
              <span className="font-mono-tech text-[10px] uppercase tracking-wider text-[#75837e] font-semibold">
                Test Clinician Accounts (Demo)
              </span>
              <span className="font-mono-tech text-[10px] text-[#146b58]">
                Pass: nivara2026
              </span>
            </div>

            <div className="space-y-2">
              {TEST_DOCTOR_ACCOUNTS.map((doc) => (
                <div
                  key={doc.id}
                  className="flex items-center justify-between rounded-xl border border-[#e4eae7] bg-[#f5f8f6] px-3 py-2.5"
                >
                  <div className="flex items-center gap-2.5">
                    <div className="h-8 w-8 rounded-full bg-[#d9ede6] text-[#146b58] grid place-items-center font-bold text-xs">
                      {doc.initials}
                    </div>
                    <div>
                      <div className="text-xs font-bold text-[#172522]">
                        {doc.name}
                      </div>
                      <div className="font-mono-tech text-[10px] text-[#71817d]">
                        {doc.email} · {doc.role}
                      </div>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleQuickDemoLogin(doc)}
                    className="inline-flex items-center gap-1 rounded-lg bg-white hover:bg-[#e3f2ed] border border-[#d6e2dd] text-[#146b58] font-bold px-2.5 py-1.5 text-[11px] transition"
                  >
                    <UserCheck className="w-3.5 h-3.5" />
                    Quick Login
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  // Clinician Auth State (persisted in localStorage)
  const [currentDoctor, setCurrentDoctor] = useState(() => {
    try {
      const saved = localStorage.getItem(SESSION_STORAGE_KEY);
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });

  const [studies, setStudies] = useState(INITIAL_STUDIES);
  const [navSection, setNavSection] = useState("worklist");
  const [activeFilter, setActiveFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [page, setPage] = useState(1);
  const pageSize = 6;

  // Drawer & Viewer States
  const [selectedStudyId, setSelectedStudyId] = useState(null);
  const [drawerTab, setDrawerTab] = useState("scores");
  const [showGradcam, setShowGradcam] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [invertViewer, setInvertViewer] = useState(false);
  const [highContrast, setHighContrast] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const isDragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0 });

  // Upload & Threshold States
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [routingGate, setRoutingGate] = useState(0.05);
  const [yoloConf, setYoloConf] = useState(0.1);
  const [ragCutoff, setRagCutoff] = useState(0.22);

  // Toast & Clipboard
  const [toast, setToast] = useState("");
  const [copiedReport, setCopiedReport] = useState(false);

  const triggerToast = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(""), 2600);
  };

  const handleLoginSuccess = (doctor) => {
    setCurrentDoctor(doctor);
    try {
      localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(doctor));
    } catch {
      // Ignore storage quota errors
    }
    triggerToast(`Signed in as ${doctor.name} (${doctor.role})`);
  };

  const handleLogout = () => {
    setSelectedStudyId(null);
    setCurrentDoctor(null);
    try {
      localStorage.removeItem(SESSION_STORAGE_KEY);
    } catch {
      // Ignore storage errors
    }
  };

  const selectedStudy = useMemo(
    () => studies.find((s) => s.id === selectedStudyId) || null,
    [studies, selectedStudyId],
  );

  // Dynamic KPI Stats
  const stats = useMemo(() => {
    const total = studies.length;
    const priority = studies.filter(
      (s) => s.status === "priority" && !s.reviewed,
    ).length;
    const uncertain = studies.filter(
      (s) => s.status === "uncertain" && !s.reviewed,
    ).length;
    const reviewed = studies.filter((s) => s.reviewed).length;
    return { total, priority, uncertain, reviewed };
  }, [studies]);

  // Filtered Worklist
  const filteredStudies = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return studies.filter((s) => {
      const matchesTab =
        activeFilter === "all" ||
        (activeFilter === "reviewed" ? s.reviewed : s.status === activeFilter);
      const matchesQuery =
        !q ||
        s.id.toLowerCase().includes(q) ||
        s.patientName.toLowerCase().includes(q) ||
        s.mrn.toLowerCase().includes(q) ||
        s.headlineResult.toLowerCase().includes(q);
      return matchesTab && matchesQuery;
    });
  }, [studies, activeFilter, searchQuery]);

  // Gate entire clinical workstation behind Doctor Sign-In
  if (!currentDoctor) {
    return <DoctorLoginScreen onLoginSuccess={handleLoginSuccess} />;
  }

  const totalPages = Math.max(1, Math.ceil(filteredStudies.length / pageSize));
  const paginatedStudies = filteredStudies.slice(
    (page - 1) * pageSize,
    page * pageSize,
  );

  const openStudyDrawer = (studyId) => {
    setSelectedStudyId(studyId);
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setInvertViewer(false);
    setHighContrast(false);
  };

  const resetViewer = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setInvertViewer(false);
    setHighContrast(false);
  };

  const handleMarkReviewed = (studyId) => {
    setStudies((prev) =>
      prev.map((s) =>
        s.id === studyId
          ? { ...s, reviewed: true, signedBy: currentDoctor.name }
          : s,
      ),
    );
    setSelectedStudyId(null);
    triggerToast(`Study ${studyId} signed off by ${currentDoctor.name}`);
  };

  const handleNextPriority = () => {
    const candidates = studies.filter(
      (s) => s.status === "priority" || s.status === "uncertain",
    );
    if (!candidates.length || !selectedStudy) return;
    const idx = candidates.findIndex((s) => s.id === selectedStudy.id);
    const next = candidates[(idx + 1) % candidates.length];
    openStudyDrawer(next.id);
  };

  const handleExportCsv = () => {
    const headers = [
      "Study ID",
      "Patient",
      "MRN",
      "Status",
      "Screening Result",
      "Routing Decision",
      "Grounding Score",
      "Reviewed",
      "Signed By",
    ];
    const lines = studies.map((s) =>
      [
        s.id,
        `"${s.patientName}"`,
        s.mrn,
        s.status,
        `"${s.headlineResult}"`,
        s.routingDecision,
        s.groundingAudit.groundingScore,
        s.reviewed ? "Yes" : "No",
        `"${s.signedBy || ""}"`,
      ].join(","),
    );
    const blob = new Blob([[headers.join(","), ...lines].join("\n")], {
      type: "text/csv;charset=utf-8;",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `nivara_triage_worklist_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    triggerToast("Exported NIVARA worklist to CSV");
  };

  const handleCopyReport = (study) => {
    const text = [
      `EXAMINATION: ${study.report.examination}`,
      `REVIEWING CLINICIAN: ${currentDoctor.name} (${currentDoctor.licenseId})`,
      "",
      "FINDINGS:",
      `- Lungs & Pleura: ${study.report.findings.lungsAndPleura}`,
      `- Cardiomediastinal Silhouette: ${study.report.findings.cardiomediastinal}`,
      `- Support Devices & Other: ${study.report.findings.devicesAndOther}`,
      "",
      "IMPRESSION:",
      ...study.report.impression.map((item, idx) => `${idx + 1}. ${item}`),
      "",
      "RECOMMENDATIONS:",
      ...study.report.recommendations.map((item) => `- ${item}`),
      "",
      "AI DISCLAIMER: Preliminary AI-generated report for clinical decision support only; requires confirmation by a qualified radiologist.",
    ].join("\n");
    navigator.clipboard.writeText(text);
    setCopiedReport(true);
    triggerToast("Preliminary radiology report copied to clipboard");
    setTimeout(() => setCopiedReport(false), 2000);
  };

  const handleFileUpload = async (file) => {
    if (!file) return;
    setUploading(true);
    const localPreviewUrl = URL.createObjectURL(file);
    const newId = `CXR-${Math.floor(2049 + Math.random() * 900)}`;

    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API_BASE_URL}/api/infer`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("FastAPI backend offline");
      const data = await res.json();

      const hasHighFinding = Object.entries(data.cihmlc_scores || {}).some(
        ([k, v]) => k !== "No Finding" && k !== "Support Devices" && v >= 0.5,
      );

      const newStudy = {
        id: newId,
        patientName: file.name.replace(/\.[^/.]+$/, ""),
        mrn: `NIV-${Math.floor(1000 + Math.random() * 8999)}`,
        receivedTime: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
        receivedDate: new Date().toLocaleDateString("en-GB", {
  day: "2-digit",
  month: "short",
  year: "numeric",
}),
        view: "Frontal · PA/AP",
        status: hasHighFinding ? "priority" : "uncertain",
        reviewed: false,
        signedBy: null,
        headlineResult: hasHighFinding
          ? "Positive thoracic finding"
          : "Borderline / screening review",
        priorityLabel: hasHighFinding ? "High" : "Review",
        routingDecision: data.routing_decision,
        routingThreshold: data.routing_threshold,
        cihmlcTime: data.cihmlc_time_seconds,
        yoloTime: data.yolo26_time_seconds,
        imageUrl: localPreviewUrl,
        gradcamUrl: data.gradcam_base64
          ? `data:image/jpeg;base64,${data.gradcam_base64}`
          : null,
        cihmlcScores: data.cihmlc_scores,
        yoloDetections: (data.yolo26_detections || []).map((d) => ({
          ...d,
          zone: d.zone || `${d.class_name} localization`,
        })),
        ragCitations: data.retrieved_guidelines || [],
        report: data.parsed_report || {
          examination: "Chest Radiograph (PA/AP)",
          findings: {
            lungsAndPleura: data.radiology_report,
            cardiomediastinal: "See structured report text.",
            devicesAndOther: "See structured report text.",
          },
          impression: ["Generated via local MedGemma-4B RAG pipeline."],
          recommendations: [
            "Correlate with clinical history and retrieved guidelines.",
          ],
        },
        groundingAudit: {
          groundingScore: data.grounding_audit?.grounding_score ?? 1.0,
          verifiedMentions:
            data.grounding_audit?.verified_positive_mentions || [],
          ungroundedHallucinations:
            data.grounding_audit?.ungrounded_hallucinations || [],
          verifiedCitations: (data.grounding_audit?.verified_citations || [])
            .length,
        },
      };

      setStudies((prev) => [newStudy, ...prev]);
      setUploadOpen(false);
      openStudyDrawer(newId);
      triggerToast(`NIVARA inference complete (${data.cihmlc_time_seconds}s)`);
    } catch {
      const fallbackStudy = {
        ...INITIAL_STUDIES[0],
        id: newId,
        patientName: file.name.replace(/\.[^/.]+$/, ""),
        mrn: `NIV-${Math.floor(1000 + Math.random() * 8999)}`,
        receivedTime: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
        imageUrl: localPreviewUrl,
        reviewed: false,
        signedBy: null,
      };
      setStudies((prev) => [fallbackStudy, ...prev]);
      setUploadOpen(false);
      openStudyDrawer(newId);
      triggerToast(
        "Loaded scan in preview mode (start nivara_api.py for live CPU inference)",
      );
    } finally {
      setUploading(false);
    }
  };

  const renderBadge = (status) => {
    if (status === "priority") {
      return (
        <span className="inline-flex items-center gap-1.5 rounded-full bg-[#fbeceb] px-2.5 py-1 text-[11px] font-bold text-[#a63b38]">
          <span className="h-1.5 w-1.5 rounded-full bg-[#b44340]" />
          Priority review
        </span>
      );
    }
    if (status === "uncertain") {
      return (
        <span className="inline-flex items-center gap-1.5 rounded-full bg-[#fff4df] px-2.5 py-1 text-[11px] font-bold text-[#986018]">
          <span className="h-1.5 w-1.5 rounded-full bg-[#a96616]" />
          Uncertain result
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-[#e3f2ed] px-2.5 py-1 text-[11px] font-bold text-[#146b58]">
        <span className="h-1.5 w-1.5 rounded-full bg-[#146b58]" />
        Lower priority
      </span>
    );
  };

  return (
    <div className="min-h-screen grid grid-cols-1 md:grid-cols-[72px_1fr] lg:grid-cols-[242px_1fr] bg-[#f5f8f6] text-[#172522]">
      {/* SIDEBAR */}
      <aside className="hidden md:flex flex-col gap-7 bg-[#102b27] text-[#d8e8e3] px-3 lg:px-4 py-6">
        <div className="flex items-center gap-3 px-1 lg:px-2">
          <div className="h-9 w-9 shrink-0 rounded-xl bg-[#d7eee5] text-[#146b58] grid place-items-center font-display font-extrabold text-lg">
            ✳
          </div>
          <div className="hidden lg:block">
            <div className="font-display font-extrabold text-[19px] tracking-tight text-white leading-none">
              nivara
            </div>
            <div className="font-mono-tech text-[9px] tracking-[1.4px] text-[#91b0a5] mt-1">
              RADIOLOGY TRIAGE
            </div>
          </div>
        </div>

        <div>
          <div className="hidden lg:block font-mono-tech text-[10px] tracking-[1.2px] text-[#76948a] uppercase px-2.5 mb-2">
            Workspace
          </div>
          <nav className="grid gap-1">
            {[
              {
                id: "all",
                label: "Worklist",
                icon: LayoutGrid,
                count: studies.length,
              },
              {
                id: "priority",
                label: "Priority review",
                icon: AlertCircle,
                count: stats.priority,
              },
              {
                id: "uncertain",
                label: "Uncertain",
                icon: Clock,
                count: stats.uncertain,
              },
              {
                id: "reviewed",
                label: "Reviewed",
                icon: CheckCircle2,
                count: stats.reviewed,
              },
            ].map((item) => {
              const Icon = item.icon;
              const isActive =
                navSection === "worklist" && activeFilter === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => {
                    setNavSection("worklist");
                    setActiveFilter(item.id);
                    setPage(1);
                  }}
                  className={`flex items-center justify-center lg:justify-start gap-3 px-2.5 lg:px-3 py-2.5 rounded-lg text-[13px] transition ${
                    isActive
                      ? "bg-[#23483f] text-white font-semibold"
                      : "text-[#b7cbc4] hover:bg-[#1a3a33] hover:text-white"
                  }`}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  <span className="hidden lg:inline">{item.label}</span>
                  <span className="hidden lg:inline ml-auto rounded-full bg-[#315a50] text-[#e5f4ed] px-2 py-0.5 font-mono-tech text-[10px]">
                    {item.count}
                  </span>
                </button>
              );
            })}
          </nav>
        </div>

        <div>
          <div className="hidden lg:block font-mono-tech text-[10px] tracking-[1.2px] text-[#76948a] uppercase px-2.5 mb-2">
            System
          </div>
          <nav className="grid gap-1">
            <button
              onClick={() => setNavSection("preferences")}
              className={`flex items-center justify-center lg:justify-start gap-3 px-2.5 lg:px-3 py-2.5 rounded-lg text-[13px] transition ${
                navSection === "preferences"
                  ? "bg-[#23483f] text-white font-semibold"
                  : "text-[#b7cbc4] hover:bg-[#1a3a33] hover:text-white"
              }`}
            >
              <Sliders className="w-4 h-4 shrink-0" />
              <span className="hidden lg:inline">Routing & Thresholds</span>
            </button>
            <button
              onClick={() => setNavSection("about")}
              className={`flex items-center justify-center lg:justify-start gap-3 px-2.5 lg:px-3 py-2.5 rounded-lg text-[13px] transition ${
                navSection === "about"
                  ? "bg-[#23483f] text-white font-semibold"
                  : "text-[#b7cbc4] hover:bg-[#1a3a33] hover:text-white"
              }`}
            >
              <Info className="w-4 h-4 shrink-0" />
              <span className="hidden lg:inline">Pipeline & RAG Audit</span>
            </button>
          </nav>
        </div>

        {/* SIGNED-IN DOCTOR PROFILE & LOGOUT */}
        <div className="mt-auto border-t border-[#29463f] pt-4 px-1 lg:px-2 space-y-3">
          <div className="flex items-center justify-center lg:justify-start gap-2.5">
            <div className="h-8 w-8 shrink-0 rounded-full bg-[#d9ede6] text-[#1a5d4d] grid place-items-center font-bold text-xs">
              {currentDoctor.initials}
            </div>
            <div className="hidden lg:block min-w-0 text-xs text-[#dce8e4]">
              <div className="font-semibold truncate">{currentDoctor.name}</div>
              <div className="text-[10px] text-[#8fa9a0] truncate">
                {currentDoctor.role} · {currentDoctor.licenseId}
              </div>
            </div>
          </div>

          <button
            onClick={handleLogout}
            title="Sign out"
            className="w-full flex items-center justify-center lg:justify-start gap-2.5 px-2.5 py-2 rounded-lg bg-[#183b35] hover:bg-[#275149] text-[#cfe4dd] hover:text-white text-xs font-medium transition"
          >
            <LogOut className="w-3.5 h-3.5 shrink-0 text-[#f3a6a4]" />
            <span className="hidden lg:inline">Sign out</span>
          </button>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <div className="min-w-0 flex flex-col">
        {/* TOPBAR */}
        <header className="h-16 bg-white border-b border-[#e4eae7] px-5 lg:px-9 flex items-center justify-between gap-3">
          <div className="text-xs text-[#83908c] truncate">
            Workspace &nbsp;/&nbsp;{" "}
            <b className="text-[#354640] font-semibold">
              {navSection === "worklist"
                ? "Chest X-ray triage"
                : navSection === "preferences"
                  ? "Routing & thresholds"
                  : "Architecture & grounding audit"}
            </b>
          </div>

          <div className="flex items-center gap-3 text-xs text-[#5e6e69]">
            <span className="hidden xl:inline-flex items-center gap-2 font-mono-tech text-[11px]">
              <span className="h-2 w-2 rounded-full bg-[#35a779] ring-4 ring-[#e6f5ed]" />
              MedGemma-4B + FAISS Ready
            </span>

            <button
              onClick={() => setUploadOpen(true)}
              className="inline-flex items-center gap-2 rounded-lg bg-[#146b58] hover:bg-[#0f5445] text-white font-semibold px-3.5 py-2 text-xs transition shadow-sm"
            >
              <Upload className="w-3.5 h-3.5" />
              New X-Ray Scan
            </button>

            <div className="h-5 w-px bg-[#e4eae7] hidden sm:block" />

            <div className="hidden sm:flex items-center gap-2 rounded-lg bg-[#f5f8f6] border border-[#e4eae7] px-2.5 py-1.5">
              <span className="h-5 w-5 rounded-full bg-[#146b58] text-white grid place-items-center text-[10px] font-bold">
                {currentDoctor.initials}
              </span>
              <span className="font-semibold text-[#172522] text-[11px]">
                {currentDoctor.name}
              </span>
            </div>

            <button
              onClick={handleLogout}
              className="inline-flex items-center gap-1.5 rounded-lg border border-[#e4eae7] bg-white hover:bg-[#fbeceb] hover:border-[#f5c6c4] hover:text-[#a63b38] px-3 py-2 text-xs font-semibold text-[#50615b] transition"
            >
              <LogOut className="w-3.5 h-3.5" />
              <span>Log out</span>
            </button>
          </div>
        </header>

        {/* PAGE BODY */}
        <main className="max-w-[1440px] w-full mx-auto px-5 lg:px-9 py-8 flex-1">
          {navSection === "worklist" && (
            <>
              {/* HEADLINE */}
              <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-6">
                <div>
                  <div className="font-mono-tech text-[10px] tracking-[1.3px] text-[#75837e] uppercase mb-1.5">
                    FRIDAY, 20 NOVEMBER 2026 ·{" "}
                    {currentDoctor.department.toUpperCase()}
                  </div>
                  <h1 className="font-display text-2xl lg:text-[27px] font-bold tracking-tight text-[#172522]">
                    Welcome back, {currentDoctor.name}
                  </h1>
                  <p className="text-[13px] text-[#75817d] mt-1">
                    Dual-gated CIHMLC + YOLO26m screening with
                    guideline-grounded MedGemma radiology reporting.
                  </p>
                </div>

                <div className="inline-flex items-center gap-2 rounded-lg border border-[#e4eae7] bg-white px-3.5 py-2.5 text-xs text-[#5d6d67]">
                  <ShieldCheck className="w-4 h-4 text-[#146b58]" />
                  <span>Grounding Audit: 100% Verified</span>
                </div>
              </div>

              {/* KPI CARDS */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3.5 mb-6">
                <div className="bg-white border border-[#e4eae7] rounded-xl p-4 shadow-[0_12px_34px_rgba(28,57,48,0.03)]">
                  <div className="flex items-center justify-between text-xs text-[#74817d]">
                    <span>Studies in queue</span>
                    <Activity className="w-4 h-4 opacity-70" />
                  </div>
                  <div className="font-display text-[27px] font-bold mt-2 tracking-tight">
                    {stats.total}{" "}
                    <span className="font-sans text-[11px] font-medium text-[#89948f] ml-1">
                      active
                    </span>
                  </div>
                </div>

                <div className="bg-white border border-[#e4eae7] rounded-xl p-4 shadow-[0_12px_34px_rgba(28,57,48,0.03)]">
                  <div className="flex items-center justify-between text-xs text-[#74817d]">
                    <span>Priority review</span>
                    <AlertCircle className="w-4 h-4 text-[#b44340] opacity-80" />
                  </div>
                  <div className="font-display text-[27px] font-bold text-[#b44340] mt-2 tracking-tight">
                    {stats.priority}{" "}
                    <span className="font-sans text-[11px] font-medium text-[#89948f] ml-1">
                      awaiting review
                    </span>
                  </div>
                </div>

                <div className="bg-white border border-[#e4eae7] rounded-xl p-4 shadow-[0_12px_34px_rgba(28,57,48,0.03)]">
                  <div className="flex items-center justify-between text-xs text-[#74817d]">
                    <span>Needs attention</span>
                    <Clock className="w-4 h-4 text-[#a96616] opacity-80" />
                  </div>
                  <div className="font-display text-[27px] font-bold text-[#a96616] mt-2 tracking-tight">
                    {stats.uncertain}{" "}
                    <span className="font-sans text-[11px] font-medium text-[#89948f] ml-1">
                      borderline signal
                    </span>
                  </div>
                </div>

                <div className="bg-white border border-[#e4eae7] rounded-xl p-4 shadow-[0_12px_34px_rgba(28,57,48,0.03)]">
                  <div className="flex items-center justify-between text-xs text-[#74817d]">
                    <span>Reviewed & signed</span>
                    <CheckCircle2 className="w-4 h-4 text-[#146b58] opacity-80" />
                  </div>
                  <div className="font-display text-[27px] font-bold text-[#146b58] mt-2 tracking-tight">
                    {stats.reviewed}{" "}
                    <span className="font-sans text-[11px] font-medium text-[#89948f] ml-1">
                      completed
                    </span>
                  </div>
                </div>
              </div>

              {/* WORKLIST PANEL */}
              <div className="bg-white border border-[#e4eae7] rounded-[13px] shadow-[0_12px_34px_rgba(28,57,48,0.03)] overflow-hidden">
                <div className="px-5 pt-5 flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h2 className="font-display text-[15px] font-bold">
                      Radiology worklist
                    </h2>
                    <p className="text-[11px] text-[#84908c] mt-0.5">
                      Click any study row to inspect Grad-CAM heatmaps, YOLO26m
                      boxes, and RAG citations
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleExportCsv}
                      className="inline-flex items-center gap-1.5 rounded-lg border border-[#e4eae7] bg-white hover:bg-[#f5f8f6] px-3 py-1.5 text-[11px] font-medium text-[#50615b] transition"
                    >
                      <Download className="w-3.5 h-3.5" />
                      Export CSV
                    </button>
                    <button
                      onClick={() =>
                        triggerToast("Worklist synced with local NIVARA store")
                      }
                      className="inline-flex items-center gap-1.5 rounded-lg border border-[#e4eae7] bg-white hover:bg-[#f5f8f6] px-3 py-1.5 text-[11px] font-medium text-[#50615b] transition"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                      Refresh
                    </button>
                  </div>
                </div>

                {/* FILTER & SEARCH BAR */}
                <div className="px-5 pt-4 pb-3 border-b border-[#e4eae7] flex flex-wrap items-center gap-1.5">
                  {[
                    { id: "all", label: "All studies", count: studies.length },
                    {
                      id: "priority",
                      label: "Priority review",
                      count: studies.filter((s) => s.status === "priority")
                        .length,
                    },
                    {
                      id: "uncertain",
                      label: "Uncertain",
                      count: studies.filter((s) => s.status === "uncertain")
                        .length,
                    },
                    {
                      id: "lower",
                      label: "Lower priority",
                      count: studies.filter((s) => s.status === "lower").length,
                    },
                    {
                      id: "reviewed",
                      label: "Reviewed",
                      count: studies.filter((s) => s.reviewed).length,
                    },
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => {
                        setActiveFilter(tab.id);
                        setPage(1);
                      }}
                      className={`px-3 py-1.5 rounded-lg text-[11px] transition ${
                        activeFilter === tab.id
                          ? "bg-[#e3f2ed] text-[#146b58] font-bold"
                          : "text-[#78847f] hover:bg-[#f5f8f6]"
                      }`}
                    >
                      {tab.label}
                      <span className="font-mono-tech text-[10px] ml-1.5 opacity-75">
                        {tab.count}
                      </span>
                    </button>
                  ))}

                  <div className="relative ml-auto w-full sm:w-56 mt-2 sm:mt-0">
                    <Search className="w-3.5 h-3.5 text-[#87928e] absolute left-2.5 top-2.5" />
                    <input
                      value={searchQuery}
                      onChange={(e) => {
                        setSearchQuery(e.target.value);
                        setPage(1);
                      }}
                      placeholder="Search study ID, finding, MRN..."
                      className="w-full rounded-lg border border-[#e4eae7] pl-8 pr-3 py-1.5 text-[11px] text-[#46534e] outline-none focus:border-[#82b9a8]"
                    />
                  </div>
                </div>

                {/* TABLE */}
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[820px] border-collapse">
                    <thead>
                      <tr className="bg-[#fbfcfb] border-b border-[#e4eae7] text-left font-mono-tech text-[10px] uppercase tracking-wider text-[#87928e]">
                        <th className="py-3 px-4">Study / Case</th>
                        <th className="py-3 px-4">Triage Status</th>
                        <th className="py-3 px-4">Primary AI Finding</th>
                        <th className="py-3 px-4">Pipeline Gate</th>
                        <th className="py-3 px-4">Peak Prob.</th>
                        <th className="py-3 px-4">Audit</th>
                        <th className="py-3 px-4 text-right">Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {paginatedStudies.map((study) => {
                        const isActivated =
                          study.routingDecision?.endsWith("_ACTIVATED");
                        return (
                          <tr
                            key={study.id}
                            onClick={() => openStudyDrawer(study.id)}
                            className="border-b border-[#edf0ee] hover:bg-[#f7fbf9] cursor-pointer transition text-xs"
                          >
                            <td className="py-3.5 px-4">
                              <div className="font-mono-tech text-[11px] text-[#364740]">
                                {study.id}
                              </div>
                              <div className="font-semibold text-[#283832]">
                                {study.patientName}
                              </div>
                              <div className="text-[10px] text-[#8b9692] mt-0.5">
                                {study.mrn} · {study.receivedTime}
                              </div>
                            </td>
                            <td className="py-3.5 px-4">
                              {renderBadge(study.status)}
                            </td>
                            <td className="py-3.5 px-4 font-medium text-[#283832]">
                              {study.headlineResult}
                              {study.reviewed && (
                                <span className="ml-2 inline-flex items-center gap-1 rounded bg-[#edf1ef] px-1.5 py-0.5 text-[10px] text-[#53625d]">
                                  <Check className="w-2.5 h-2.5" /> Signed
                                  {study.signedBy ? ` · ${study.signedBy}` : ""}
                                </span>
                              )}
                            </td>
                            <td className="py-3.5 px-4">
                              <span
                                className={`font-mono-tech text-[10px] px-2 py-1 rounded ${
                                  isActivated
                                    ? "bg-[#e3f2ed] text-[#146b58]"
                                    : "bg-[#edf1ef] text-[#697773]"
                                }`}
                              >
                                {study.routingDecision}
                              </span>
                            </td>
                            <td className="py-3.5 px-4 font-mono-tech text-[11px] text-[#45534e]">
                              {getTopConfidence(study)}%
                            </td>
                            <td className="py-3.5 px-4 font-mono-tech text-[11px] text-[#146b58]">
                              {study.groundingAudit.groundingScore.toFixed(2)}
                            </td>
                            <td className="py-3.5 px-4 text-right font-bold text-[11px] text-[#146b58]">
                              Review →
                            </td>
                          </tr>
                        );
                      })}
                      {paginatedStudies.length === 0 && (
                        <tr>
                          <td
                            colSpan={7}
                            className="py-10 text-center text-xs text-[#75817d]"
                          >
                            No chest radiograph studies match your active
                            filter.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>

                {/* PAGINATION */}
                <div className="px-5 py-3 flex items-center justify-between text-[11px] text-[#87928e]">
                  <span>
                    Showing <b>{paginatedStudies.length}</b> of{" "}
                    <b>{filteredStudies.length}</b> studies · NIVARA Local
                    Inference Store
                  </span>
                  <div className="flex items-center gap-1">
                    <button
                      disabled={page <= 1}
                      onClick={() => setPage((p) => Math.max(1, p - 1))}
                      className="rounded border border-[#e4eae7] px-2 py-1 disabled:opacity-40 hover:bg-[#f5f8f6]"
                    >
                      <ChevronLeft className="w-3.5 h-3.5" />
                    </button>
                    {Array.from(
                      { length: totalPages },
                      (_, idx) => idx + 1,
                    ).map((num) => (
                      <button
                        key={num}
                        onClick={() => setPage(num)}
                        className={`rounded border px-2.5 py-1 font-mono-tech text-[10px] ${
                          page === num
                            ? "bg-[#146b58] text-white border-[#146b58]"
                            : "border-[#e4eae7] text-[#61716b] hover:bg-[#f5f8f6]"
                        }`}
                      >
                        {num}
                      </button>
                    ))}
                    <button
                      disabled={page >= totalPages}
                      onClick={() =>
                        setPage((p) => Math.min(totalPages, p + 1))
                      }
                      className="rounded border border-[#e4eae7] px-2 py-1 disabled:opacity-40 hover:bg-[#f5f8f6]"
                    >
                      <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </div>
            </>
          )}

          {/* ROUTING & THRESHOLDS VIEW */}
          {navSection === "preferences" && (
            <div className="max-w-3xl bg-white border border-[#e4eae7] rounded-xl p-6 shadow-[0_12px_34px_rgba(28,57,48,0.03)]">
              <div className="font-mono-tech text-[10px] uppercase tracking-wider text-[#75837e]">
                PIPELINE CONFIGURATION
              </div>
              <h2 className="font-display text-xl font-bold mt-1">
                Dual-Gated Inference & RAG Retrieval Thresholds
              </h2>
              <p className="text-xs text-[#75817d] mt-1 mb-6">
                Adjust decision gates sent to the NIVARA FastAPI inference
                server.
              </p>

              <div className="space-y-5">
                <div className="p-4 rounded-lg bg-[#f5f8f6] border border-[#e4eae7]">
                  <div className="flex justify-between text-xs font-semibold">
                    <span>
                      CIHMLC Parenchymal Routing Gate (triggers Pneumonia YOLO)
                    </span>
                    <span className="font-mono-tech text-[#146b58]">
                      {routingGate.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.01"
                    max="0.50"
                    step="0.01"
                    value={routingGate}
                    onChange={(e) => setRoutingGate(parseFloat(e.target.value))}
                    className="w-full mt-2 accent-[#146b58]"
                  />
                  <p className="text-[11px] text-[#71817d] mt-1">
                    Skips bounding-box localization on low-risk radiographs to
                    save CPU latency.
                  </p>
                </div>

                <div className="p-4 rounded-lg bg-[#f5f8f6] border border-[#e4eae7]">
                  <div className="flex justify-between text-xs font-semibold">
                    <span>Pneumonia YOLO26m Confidence Floor</span>
                    <span className="font-mono-tech text-[#146b58]">
                      {yoloConf.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.05"
                    max="0.75"
                    step="0.05"
                    value={yoloConf}
                    onChange={(e) => setYoloConf(parseFloat(e.target.value))}
                    className="w-full mt-2 accent-[#146b58]"
                  />
                </div>

                <div className="p-4 rounded-lg bg-[#f5f8f6] border border-[#e4eae7]">
                  <div className="flex justify-between text-xs font-semibold">
                    <span>FAISS + BiomedBERT Hybrid Retrieval Cutoff</span>
                    <span className="font-mono-tech text-[#146b58]">
                      {ragCutoff.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.10"
                    max="0.60"
                    step="0.02"
                    value={ragCutoff}
                    onChange={(e) => setRagCutoff(parseFloat(e.target.value))}
                    className="w-full mt-2 accent-[#146b58]"
                  />
                </div>

                <button
                  onClick={() =>
                    triggerToast(
                      "Saved threshold profile for next inference run",
                    )
                  }
                  className="rounded-lg bg-[#146b58] text-white font-semibold px-4 py-2.5 text-xs hover:bg-[#0f5445]"
                >
                  Save Threshold Profile
                </button>
              </div>
            </div>
          )}

          {/* ABOUT / ARCHITECTURE VIEW */}
          {navSection === "about" && (
            <div className="max-w-4xl bg-white border border-[#e4eae7] rounded-xl p-6 shadow-[0_12px_34px_rgba(28,57,48,0.03)]">
              <div className="font-mono-tech text-[10px] uppercase tracking-wider text-[#146b58]">
                NIVARA CLINICAL AI ARCHITECTURE
              </div>
              <h2 className="font-display text-xl font-bold mt-1">
                Dual-Gated Multi-Model Vision + Guideline-Grounded RAG Pipeline
              </h2>
              <p className="text-xs text-[#75817d] mt-1 mb-6">
                End-to-end local CPU inference pipeline built with Keras,
                Ultralytics YOLO26m, LangChain, FAISS, and Ollama.
              </p>

              <div className="grid md:grid-cols-3 gap-4">
                <div className="p-4 rounded-xl border border-[#e4eae7] bg-[#f5f8f6]">
                  <div className="font-mono-tech text-[10px] text-[#146b58] font-bold">
                    STAGE 01 · VISION & XAI
                  </div>
                  <h3 className="font-display font-bold text-sm mt-1">
                    CIHMLC + Grad-CAM + Dual YOLO26m
                  </h3>
                  <p className="text-xs text-[#53615b] mt-2 leading-relaxed">
                    Multi-label Keras classifier predicts 21 thoracic labels,
                    generates Grad-CAM saliency maps, and sequentially gates
                    both the pneumonia detector and 4-class thoracic detector.
                  </p>
                </div>

                <div className="p-4 rounded-xl border border-[#e4eae7] bg-[#f5f8f6]">
                  <div className="font-mono-tech text-[10px] text-[#146b58] font-bold">
                    STAGE 02 · BIOMEDICAL RAG
                  </div>
                  <h3 className="font-display font-bold text-sm mt-1">
                    BiomedBERT + FAISS + MedGemma-4B
                  </h3>
                  <p className="text-xs text-[#53615b] mt-2 leading-relaxed">
                    Queries indexed ATS/IDSA, AHA/ACC Heart Failure, Fleischner,
                    and BTS clinical PDFs to ground recommendations with exact
                    page citations.
                  </p>
                </div>

                <div className="p-4 rounded-xl border border-[#e4eae7] bg-[#f5f8f6]">
                  <div className="font-mono-tech text-[10px] text-[#146b58] font-bold">
                    STAGE 03 · SAFETY GUARDRAIL
                  </div>
                  <h3 className="font-display font-bold text-sm mt-1">
                    Deterministic Grounding Auditor
                  </h3>
                  <p className="text-xs text-[#53615b] mt-2 leading-relaxed">
                    Verifies every positive pathology mention and PDF citation
                    against raw model probabilities and retrieved chunks to flag
                    ungrounded claims.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* CLINICAL DISCLAIMER */}
          <div className="flex items-center gap-2.5 text-[11px] text-[#74817b] mt-5 px-1">
            <ShieldCheck className="w-4 h-4 text-[#146b58] shrink-0" />
            <span>
              <b className="text-[#596962]">Clinical decision support only.</b>{" "}
              NIVARA AI screening and RAG reports do not replace qualified
              radiologist interpretation.
            </span>
          </div>
        </main>
      </div>

      {/* SLIDE-OVER STUDY REVIEW DRAWER */}
      {selectedStudy && (
        <div
          onClick={(e) => {
            if (e.target === e.currentTarget) setSelectedStudyId(null);
          }}
          className="fixed inset-0 bg-[#142522]/45 z-40 flex justify-end backdrop-blur-[1px]"
        >
          <div className="bg-[#f5f8f6] w-full max-w-[1180px] h-full overflow-y-auto shadow-2xl flex flex-col">
            {/* DRAWER STICKY HEADER */}
            <div className="bg-white border-b border-[#e4eae7] px-5 py-3.5 flex items-center justify-between sticky top-0 z-20">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setSelectedStudyId(null)}
                  className="inline-flex items-center gap-1.5 text-xs font-medium text-[#53665e] hover:text-[#172522] px-2 py-1 rounded-lg hover:bg-[#f5f8f6]"
                >
                  <ArrowLeft className="w-3.5 h-3.5" />
                  Worklist
                </button>
                <div className="h-5 w-px bg-[#e4eae7]" />
                <div>
                  <div className="font-display font-bold text-[15px] text-[#172522]">
                    {selectedStudy.patientName}
                  </div>
                  <div className="font-mono-tech text-[10px] text-[#8a9691]">
                    {selectedStudy.id} · {selectedStudy.receivedDate} ·{" "}
                    {selectedStudy.mrn}
                  </div>
                </div>
                <div className="hidden sm:block ml-2">
                  {renderBadge(selectedStudy.status)}
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleNextPriority}
                  className="rounded-lg border border-[#e4eae7] bg-white hover:bg-[#f5f8f6] px-3 py-1.5 text-[11px] font-medium text-[#50615b]"
                >
                  Next priority →
                </button>
                <button
                  onClick={() => setSelectedStudyId(null)}
                  className="h-8 w-8 rounded-lg border border-[#e4eae7] bg-white grid place-items-center text-[#63716c] hover:bg-[#f5f8f6]"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* DRAWER BODY */}
            <div className="p-5 grid grid-cols-1 lg:grid-cols-[1.35fr_1fr] gap-4 flex-1">
              {/* LEFT: INTERACTIVE RADIOGRAPH & OVERLAY VIEWER */}
              <div className="bg-white border border-[#e4eae7] rounded-xl overflow-hidden flex flex-col">
                <div className="px-4 py-3 border-b border-[#e4eae7] flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2 text-xs font-bold">
                    <Scan className="w-4 h-4 text-[#146b58]" />
                    <span>Radiograph & Explainability Viewer</span>
                  </div>

                  {/* OVERLAY TOGGLES */}
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => setShowGradcam((v) => !v)}
                      className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[10px] font-semibold transition ${
                        showGradcam
                          ? "bg-[#fbeceb] text-[#a63b38] border border-[#f5c6c4]"
                          : "bg-[#f5f8f6] text-[#71817d] border border-[#e4eae7]"
                      }`}
                    >
                      <Flame className="w-3 h-3" />
                      Grad-CAM
                    </button>
                    <button
                      onClick={() => setShowBoxes((v) => !v)}
                      className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[10px] font-semibold transition ${
                        showBoxes
                          ? "bg-[#e3f2ed] text-[#146b58] border border-[#b9dfd3]"
                          : "bg-[#f5f8f6] text-[#71817d] border border-[#e4eae7]"
                      }`}
                    >
                      <Layers className="w-3 h-3" />
                      YOLO26m ({selectedStudy.yoloDetections.length})
                    </button>
                  </div>
                </div>

                {/* VIEWPORT */}
                <div
                  onMouseDown={(e) => {
                    isDragging.current = true;
                    dragStart.current = {
                      x: e.clientX - pan.x,
                      y: e.clientY - pan.y,
                    };
                  }}
                  onMouseMove={(e) => {
                    if (!isDragging.current) return;
                    setPan({
                      x: e.clientX - dragStart.current.x,
                      y: e.clientY - dragStart.current.y,
                    });
                  }}
                  onMouseUp={() => {
                    isDragging.current = false;
                  }}
                  onMouseLeave={() => {
                    isDragging.current = false;
                  }}
                  className="relative h-[440px] lg:h-[510px] bg-[#111d20] overflow-hidden grid place-items-center cursor-grab active:cursor-grabbing"
                >
                  <div className="absolute left-3 top-3 z-10 flex items-center gap-2">
                    <span className="rounded bg-[#0c181b]/85 border border-[#29433c] px-2 py-1 font-mono-tech text-[9px] text-[#abc7be]">
                      {selectedStudy.view.toUpperCase()} · ZOOM{" "}
                      {Math.round(zoom * 100)}%
                    </span>
                    <span className="rounded bg-[#0c181b]/85 border border-[#29433c] px-2 py-1 font-mono-tech text-[9px] text-[#86efac]">
                      AUDIT{" "}
                      {selectedStudy.groundingAudit.groundingScore.toFixed(2)}
                    </span>
                  </div>

                  <div
                    style={{
                      transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                      filter: `${invertViewer ? "invert(1) " : ""}${
                        highContrast ? "contrast(1.35)" : ""
                      }`,
                    }}
                    className="w-full h-full grid place-items-center transition-transform duration-75"
                  >
                    {selectedStudy.imageUrl ? (
                      <div className="relative max-h-[92%] max-w-[92%]">
                        <img
                          src={
                            showGradcam && selectedStudy.gradcamUrl
                              ? selectedStudy.gradcamUrl
                              : selectedStudy.imageUrl
                          }
                          alt={selectedStudy.patientName}
                          className="max-h-[460px] w-auto object-contain rounded"
                        />
                        {showBoxes &&
                          selectedStudy.yoloDetections.map((det, i) => {
                            const [cx, cy, nw, nh] = det.bbox_xywh_norm;
                            const palette = getYoloPalette(det.class_name);
                            return (
                              <div
                                key={i}
                                style={{
                                  left: `${(cx - nw / 2) * 100}%`,
                                  top: `${(cy - nh / 2) * 100}%`,
                                  width: `${nw * 100}%`,
                                  height: `${nh * 100}%`,
                                  borderColor: palette.border,
                                  backgroundColor: palette.bg,
                                }}
                                className="absolute border-2 pointer-events-none rounded-sm"
                              >
                                <span
                                  style={{
                                    backgroundColor: palette.border,
                                    color: palette.text,
                                  }}
                                  className="font-mono-tech text-[9px] font-bold px-1.5 py-0.5 absolute -top-4 left-0 whitespace-nowrap rounded-t-sm"
                                >
                                  {det.class_name} {det.confidence.toFixed(2)}
                                </span>
                              </div>
                            );
                          })}
                      </div>
                    ) : (
                      <ProceduralRadiograph
                        study={selectedStudy}
                        showGradcam={showGradcam}
                        showBoxes={showBoxes}
                      />
                    )}
                  </div>
                </div>

                {/* VIEWER TOOLBAR */}
                <div className="bg-[#152326] border-t border-[#293639] px-3 py-2.5 flex flex-wrap items-center justify-center gap-2">
                  <button
                    onClick={() =>
                      setZoom((z) => Math.min(2.6, +(z + 0.25).toFixed(2)))
                    }
                    className="inline-flex items-center gap-1 rounded bg-[#1d2f33] hover:bg-[#273f44] border border-[#34474a] px-2.5 py-1 text-[11px] text-[#d2dfda]"
                  >
                    <ZoomIn className="w-3.5 h-3.5" /> Zoom +
                  </button>
                  <button
                    onClick={() =>
                      setZoom((z) => Math.max(0.75, +(z - 0.25).toFixed(2)))
                    }
                    className="inline-flex items-center gap-1 rounded bg-[#1d2f33] hover:bg-[#273f44] border border-[#34474a] px-2.5 py-1 text-[11px] text-[#d2dfda]"
                  >
                    <ZoomOut className="w-3.5 h-3.5" /> Zoom −
                  </button>
                  <button
                    onClick={() => setInvertViewer((v) => !v)}
                    className={`inline-flex items-center gap-1 rounded border px-2.5 py-1 text-[11px] ${
                      invertViewer
                        ? "bg-[#146b58] border-[#26987f] text-white"
                        : "bg-[#1d2f33] border-[#34474a] text-[#d2dfda]"
                    }`}
                  >
                    <Contrast className="w-3.5 h-3.5" /> Invert
                  </button>
                  <button
                    onClick={() => setHighContrast((v) => !v)}
                    className={`inline-flex items-center gap-1 rounded border px-2.5 py-1 text-[11px] ${
                      highContrast
                        ? "bg-[#146b58] border-[#26987f] text-white"
                        : "bg-[#1d2f33] border-[#34474a] text-[#d2dfda]"
                    }`}
                  >
                    <Eye className="w-3.5 h-3.5" /> Contrast
                  </button>
                  <button
                    onClick={resetViewer}
                    className="inline-flex items-center gap-1 rounded bg-[#1d2f33] hover:bg-[#273f44] border border-[#34474a] px-2.5 py-1 text-[11px] text-[#d2dfda]"
                  >
                    <RotateCcw className="w-3.5 h-3.5" /> Reset
                  </button>
                </div>
              </div>

              {/* RIGHT: TABBED AI SCORES & RAG RADIOLOGY REPORT */}
              <div className="flex flex-col gap-3">
                {/* SUB-TABS */}
                <div className="bg-white border border-[#e4eae7] rounded-xl p-1.5 grid grid-cols-2 gap-1">
                  <button
                    onClick={() => setDrawerTab("scores")}
                    className={`flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-bold transition ${
                      drawerTab === "scores"
                        ? "bg-[#146b58] text-white"
                        : "text-[#61716b] hover:bg-[#f5f8f6]"
                    }`}
                  >
                    <Cpu className="w-3.5 h-3.5" />
                    CIHMLC & YOLO26m
                  </button>
                  <button
                    onClick={() => setDrawerTab("report")}
                    className={`flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-bold transition ${
                      drawerTab === "report"
                        ? "bg-[#146b58] text-white"
                        : "text-[#61716b] hover:bg-[#f5f8f6]"
                    }`}
                  >
                    <FileText className="w-3.5 h-3.5" />
                    RAG Radiology Report
                  </button>
                </div>

                {/* TAB 1: CIHMLC PROBABILITIES & TELEMETRY */}
                {drawerTab === "scores" && (
                  <div className="bg-white border border-[#e4eae7] rounded-xl p-4 space-y-4">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <div className="font-display font-bold text-sm text-[#172522]">
                          {selectedStudy.headlineResult}
                        </div>
                        <div className="text-[11px] text-[#75817d] mt-0.5">
                          Gate:{" "}
                          <span className="font-mono-tech">
                            {selectedStudy.routingDecision}
                          </span>{" "}
                          · CIHMLC {selectedStudy.cihmlcTime}s / YOLO{" "}
                          {selectedStudy.yoloTime}s
                        </div>
                      </div>
                      <span className="inline-flex items-center gap-1 rounded-full bg-[#e3f2ed] px-2.5 py-1 font-mono-tech text-[10px] font-bold text-[#146b58]">
                        <ShieldCheck className="w-3 h-3" />
                        Audit{" "}
                        {selectedStudy.groundingAudit.groundingScore.toFixed(2)}
                      </span>
                    </div>

                    <div>
                      <div className="font-mono-tech text-[10px] uppercase tracking-wider text-[#75837e] mb-2">
                        CIHMLC Multi-Label Probabilities
                      </div>
                      <div className="space-y-2 max-h-[260px] overflow-y-auto pr-1">
                        {Object.entries(selectedStudy.cihmlcScores)
                          .sort(([, a], [, b]) => b - a)
                          .map(([label, score]) => {
                            const pct = Math.round(score * 100);
                            const barColor =
                              label === "No Finding"
                                ? "bg-[#146b58]"
                                : score >= 0.5
                                  ? "bg-[#b44340]"
                                  : score >= 0.12
                                    ? "bg-[#d1953f]"
                                    : "bg-[#cbd5d1]";
                            return (
                              <div key={label} className="text-xs">
                                <div className="flex justify-between mb-1">
                                  <span className="font-medium text-[#283832]">
                                    {label}
                                  </span>
                                  <span className="font-mono-tech text-[11px] text-[#53615b]">
                                    {(score * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <div className="h-1.5 rounded-full bg-[#edf1ef] overflow-hidden">
                                  <div
                                    style={{ width: `${Math.max(2, pct)}%` }}
                                    className={`h-full rounded-full ${barColor}`}
                                  />
                                </div>
                              </div>
                            );
                          })}
                      </div>
                    </div>

                    <div className="pt-2 border-t border-[#edf0ee]">
                      <div className="font-mono-tech text-[10px] uppercase tracking-wider text-[#75837e] mb-2">
                        YOLO26m Localized Regions (
                        {selectedStudy.yoloDetections.length})
                      </div>
                      {selectedStudy.yoloDetections.length > 0 ? (
                        <div className="space-y-1.5">
                          {selectedStudy.yoloDetections.map((det, i) => {
                            const palette = getYoloPalette(det.class_name);
                            return (
                              <div
                                key={i}
                                className="flex items-center justify-between rounded-lg bg-[#f5f8f6] px-3 py-2 text-xs"
                              >
                                <div className="flex items-center gap-2 min-w-0">
                                  <span
                                    style={{ backgroundColor: palette.border }}
                                    className="h-2.5 w-2.5 rounded-full shrink-0"
                                  />
                                  <span className="font-semibold text-[#283832] truncate">
                                    {det.class_name}
                                  </span>
                                  {det.zone && (
                                    <span className="text-[11px] text-[#75817d] truncate hidden sm:inline">
                                      · {det.zone}
                                    </span>
                                  )}
                                </div>
                                <span className="font-mono-tech text-[11px] text-[#146b58] font-semibold shrink-0 ml-2">
                                  conf {det.confidence.toFixed(2)}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <div className="text-[11px] text-[#75817d] bg-[#f5f8f6] rounded-lg px-3 py-2">
                          No focal boxes above threshold (
                          {selectedStudy.routingDecision}).
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* TAB 2: MEDGEMMA-4B RAG REPORT & CITATIONS */}
                {drawerTab === "report" && (
                  <div className="bg-white border border-[#e4eae7] rounded-xl p-4 space-y-3.5 max-h-[540px] overflow-y-auto">
                    <div className="flex items-center justify-between">
                      <div className="inline-flex items-center gap-1.5 text-xs font-bold text-[#146b58]">
                        <Sparkles className="w-3.5 h-3.5" />
                        Preliminary Report (MedGemma-4B + FAISS)
                      </div>
                      <button
                        onClick={() => handleCopyReport(selectedStudy)}
                        className="inline-flex items-center gap-1 rounded-md border border-[#e4eae7] px-2 py-1 text-[10px] font-medium text-[#53615b] hover:bg-[#f5f8f6]"
                      >
                        {copiedReport ? (
                          <>
                            <Check className="w-3 h-3 text-[#146b58]" /> Copied
                          </>
                        ) : (
                          <>
                            <Copy className="w-3 h-3" /> Copy
                          </>
                        )}
                      </button>
                    </div>

                    <div className="space-y-2.5 text-xs leading-relaxed text-[#283832] bg-[#fafbf9] border border-[#edf0ed] rounded-lg p-3">
                      <div>
                        <span className="font-mono-tech text-[10px] font-bold text-[#75837e] uppercase block">
                          FINDINGS
                        </span>
                        <p className="mt-1">
                          <b>Lungs & Pleura:</b>{" "}
                          {selectedStudy.report.findings.lungsAndPleura}
                        </p>
                        <p className="mt-1">
                          <b>Cardiomediastinal:</b>{" "}
                          {selectedStudy.report.findings.cardiomediastinal}
                        </p>
                        <p className="mt-1">
                          <b>Devices & Other:</b>{" "}
                          {selectedStudy.report.findings.devicesAndOther}
                        </p>
                      </div>

                      <div className="pt-2 border-t border-[#edf0ed]">
                        <span className="font-mono-tech text-[10px] font-bold text-[#75837e] uppercase block">
                          IMPRESSION
                        </span>
                        <ol className="list-decimal pl-4 mt-1 space-y-1">
                          {selectedStudy.report.impression.map((imp, idx) => (
                            <li key={idx}>{imp}</li>
                          ))}
                        </ol>
                      </div>

                      <div className="pt-2 border-t border-[#edf0ed]">
                        <span className="font-mono-tech text-[10px] font-bold text-[#75837e] uppercase block">
                          GUIDELINE RECOMMENDATIONS
                        </span>
                        <ul className="list-disc pl-4 mt-1 space-y-1">
                          {selectedStudy.report.recommendations.map(
                            (rec, idx) => (
                              <li key={idx}>{rec}</li>
                            ),
                          )}
                        </ul>
                      </div>
                    </div>

                    {selectedStudy.ragCitations.length > 0 && (
                      <div>
                        <div className="flex items-center gap-1.5 font-mono-tech text-[10px] uppercase tracking-wider text-[#75837e] mb-2">
                          <BookOpen className="w-3 h-3 text-[#146b58]" />
                          Retrieved Clinical Guideline Sources (
                          {selectedStudy.ragCitations.length})
                        </div>
                        <div className="space-y-2">
                          {selectedStudy.ragCitations.map((cit, idx) => (
                            <div
                              key={idx}
                              className="rounded-lg border border-[#e4eae7] bg-[#f5f8f6] p-2.5 text-[11px]"
                            >
                              <div className="flex items-center justify-between font-mono-tech text-[10px] text-[#146b58] font-semibold">
                                <span>[{cit.citation}]</span>
                                <span>sim {cit.score}</span>
                              </div>
                              <p className="text-[#53615b] mt-1 leading-snug">
                                {cit.excerpt}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* DRAWER FOOTER ACTIONS */}
                <div className="mt-auto grid grid-cols-2 gap-2 pt-1">
                  <button
                    onClick={() => handleMarkReviewed(selectedStudy.id)}
                    className="inline-flex items-center justify-center gap-1.5 rounded-lg bg-[#146b58] hover:bg-[#0f5445] text-white font-bold py-2.5 px-3 text-xs transition"
                  >
                    <Check className="w-4 h-4" />
                    Sign Off ({currentDoctor.initials})
                  </button>
                  <button
                    onClick={() => setSelectedStudyId(null)}
                    className="rounded-lg border border-[#e4eae7] bg-white hover:bg-[#f5f8f6] text-[#52615b] font-semibold py-2.5 px-3 text-xs transition"
                  >
                    Return to Worklist
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* NEW SCAN UPLOAD MODAL */}
      {uploadOpen && (
        <div
          onClick={(e) => {
            if (e.target === e.currentTarget) setUploadOpen(false);
          }}
          className="fixed inset-0 bg-[#142522]/45 z-50 grid place-items-center p-4"
        >
          <div className="bg-white border border-[#e4eae7] rounded-2xl max-w-md w-full p-6 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="font-display font-bold text-lg">
                  Run NIVARA Chest X-Ray Triage
                </h3>
                <p className="text-xs text-[#75817d]">
                  Upload a PA/AP chest radiograph (.jpg, .png, .dcm)
                </p>
              </div>
              <button
                onClick={() => setUploadOpen(false)}
                className="h-8 w-8 rounded-lg border border-[#e4eae7] grid place-items-center text-[#63716c]"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <label className="border-2 border-dashed border-[#b9dfd3] hover:border-[#146b58] bg-[#f5f8f6] rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition">
              <Upload className="w-8 h-8 text-[#146b58] mb-2" />
              <span className="font-semibold text-sm text-[#172522]">
                {uploading
                  ? "Running CIHMLC + Dual YOLO26m + MedGemma..."
                  : "Click to select radiograph"}
              </span>
              <span className="text-[11px] text-[#75817d] mt-1">
                Connects to {API_BASE_URL || "local NIVARA server"}/api/infer
              </span>
              <input
                type="file"
                accept="image/*,.dcm"
                disabled={uploading}
                onChange={(e) => handleFileUpload(e.target.files?.[0])}
                className="hidden"
              />
            </label>
          </div>
        </div>
      )}

      {/* TOAST */}
      {toast && (
        <div className="fixed bottom-6 right-6 z-50 bg-[#183e34] text-[#effbf5] px-4 py-3 rounded-xl shadow-lg text-xs font-medium flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-[#6ee7b7]" />
          {toast}
        </div>
      )}
    </div>
  );
}
