import { useState, useRef, useEffect } from "react";

// Backend contract: POST multipart "file" -> JSON returned by run_nivara_inference(),
// plus "gradcam_url" (a URL the browser can load for the Grad-CAM image).
const API_URL = "/api/analyze";

const MAIN = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices", "No Finding"];

const DEMO = {
  demo: true,
  filename: "demo_study.jpeg",
  image_size: [1000, 1200],
  status: "success",
  cihmlc_pneumonia_score: 0.31,
  routing_threshold: 0.05,
  routing_decision: "YOLO26_ACTIVATED",
  yolo26_detection_count: 2,
  yolo26_detections: [
    { class_name: "pneumonia_opacity", confidence: 0.58, bbox_xywh_norm: [0.3, 0.62, 0.22, 0.3] },
    { class_name: "pneumonia_opacity", confidence: 0.41, bbox_xywh_norm: [0.68, 0.66, 0.2, 0.28] },
  ],
  cihmlc_time_seconds: 0.42,
  yolo26_time_seconds: 0.31,
  cihmlc_scores: { Atelectasis: 0.18, Cardiomegaly: 0.71, Consolidation: 0.27, Edema: 0.62, "Enlarged Cardiomediastinum": 0.14, Fracture: 0.01, "Lung Lesion": 0.02, "Lung Opacity": 0.58, "No Finding": 0.02, "Pleural Effusion": 0.33, "Pleural Other": 0.02, Pneumonia: 0.31, Pneumothorax: 0.01, "Support Devices": 0.04, Uncertain: 0.09, Abnormal: 0.93, Cardiac: 0.48, "Fluid Accumulation": 0.36, "Missing Lung Tissue": 0.01, Opacity: 0.55, Other: 0.06 },
  radiology_report:
    "**EXAMINATION:** Chest Radiograph (PA/AP)\n\n**FINDINGS:**\n- **Lungs & Pleura:** Bilateral opacities are seen in the mid-to-lower lung zones. A small pleural effusion and consolidation cannot be excluded. Pneumothorax is absent.\n- **Cardiomediastinal Silhouette:** The cardiac silhouette is enlarged. Enlarged cardiomediastinum cannot be excluded.\n- **Support Devices & Other:** No support devices or fractures are identified.\n\n**IMPRESSION:**\n1. Cardiomegaly with bilateral mid-to-lower zone opacities, favoring cardiogenic pulmonary edema or fluid overload.\n2. Pneumonia cannot be excluded as a contributing process.\n\n**RECOMMENDATIONS:**\n- Measure BNP or NT-proBNP and obtain an echocardiogram to assess cardiac function. [esc_hf_2021.pdf, p. 9]\n- Consider a follow-up chest radiograph after diuresis to reassess the opacities. [ats_cap_2019.pdf, p. 14]\n\n**AI DISCLAIMER:** Preliminary AI-generated report for clinical decision support only; requires confirmation by a qualified radiologist.",
  grounding_audit: { grounding_score: 1.0, verified_positive_mentions: ["Cardiomegaly", "Edema"], ungrounded_hallucinations: [], verified_citations: ["esc_hf_2021.pdf, p. 9", "ats_cap_2019.pdf, p. 14"], ungrounded_citations: [] },
};

const bucket = (s) => (s >= 0.5 ? "present" : s >= 0.12 ? "borderline" : s >= 0.05 ? "low" : "absent");
const BAR = { present: "bg-rose-500", borderline: "bg-amber-400", low: "bg-slate-400", absent: "bg-slate-300" };
const BUCKET_TEXT = { present: "Present", borderline: "Borderline", low: "Unlikely", absent: "Absent" };

function describeBox([cx, cy, , nh]) {
  const side = cx < 0.45 ? "Right lung" : cx > 0.55 ? "Left lung" : "Central";
  const zone = cy < 0.42 ? "upper zone" : cy > 0.58 ? (nh >= 0.45 ? "mid-to-lower zones" : "lower zone") : "mid zone";
  return `${side}, ${zone}`;
}

async function analyze(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(API_URL, { method: "POST", body: fd });
  if (!res.ok) throw new Error(`The server answered ${res.status}. Check that the Nivara API is running at ${API_URL}.`);
  return res.json();
}

function DemoXray({ heat }) {
  return (
    <svg viewBox="0 0 400 480" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
      <defs>
        <radialGradient id="lg"><stop offset="0" stopColor="#64748b" stopOpacity=".6" /><stop offset="1" stopColor="#64748b" stopOpacity=".12" /></radialGradient>
        <radialGradient id="hot"><stop offset="0" stopColor="#ef4444" stopOpacity=".85" /><stop offset=".5" stopColor="#f59e0b" stopOpacity=".45" /><stop offset="1" stopColor="#f59e0b" stopOpacity="0" /></radialGradient>
      </defs>
      <rect width="400" height="480" fill="#0b1220" />
      <ellipse cx="135" cy="250" rx="75" ry="150" fill="url(#lg)" />
      <ellipse cx="270" cy="250" rx="70" ry="145" fill="url(#lg)" />
      <path d="M200 40V440" stroke="#94a3b8" strokeOpacity=".3" strokeWidth="14" />
      <ellipse cx="228" cy="300" rx="55" ry="70" fill="#94a3b8" fillOpacity=".4" />
      {[0, 1, 2, 3, 4, 5].map((i) => (
        <g key={i} fill="none" stroke="#cbd5e1" strokeOpacity=".22" strokeWidth="3">
          <path d={`M195 ${95 + i * 48} Q130 ${70 + i * 48} 62 ${110 + i * 48}`} />
          <path d={`M205 ${95 + i * 48} Q270 ${70 + i * 48} 338 ${110 + i * 48}`} />
        </g>
      ))}
      {heat && (<><ellipse cx="120" cy="320" rx="75" ry="95" fill="url(#hot)" /><ellipse cx="285" cy="335" rx="65" ry="85" fill="url(#hot)" /></>)}
    </svg>
  );
}

function Film({ data, src, mode }) {
  const [w, h] = data.image_size;
  const skipped = data.routing_decision === "YOLO26_SKIPPED";
  let body;
  if (mode === "cam") {
    body = data.gradcam_url ? <img src={data.gradcam_url} alt="Grad-CAM heatmap for pneumonia" className="absolute inset-0 h-full w-full object-fill" /> : data.demo ? <DemoXray heat /> : <Note>The API did not return a Grad-CAM image. Add gradcam_url to the response.</Note>;
  } else {
    body = (
      <>
        {src ? <img src={src} alt="Chest radiograph" className="absolute inset-0 h-full w-full object-fill" /> : data.demo ? <DemoXray /> : <Note>Preview is not available for this file type.</Note>}
        {!skipped && data.yolo26_detections.map((d, i) => {
          const [cx, cy, bw, bh] = d.bbox_xywh_norm;
          return (
            <div key={i} className="absolute border-2 border-cyan-300" style={{ left: `${(cx - bw / 2) * 100}%`, top: `${(cy - bh / 2) * 100}%`, width: `${bw * 100}%`, height: `${bh * 100}%` }}>
              <span className="absolute -top-6 left-[-2px] bg-cyan-300 px-1.5 py-0.5 text-xs font-semibold tabular-nums text-slate-900">Region {i + 1} {d.confidence.toFixed(2)}</span>
            </div>
          );
        })}
      </>
    );
  }
  return (
    <div className="relative mx-auto w-full overflow-hidden rounded-md bg-black" style={{ aspectRatio: `${w} / ${h}`, maxWidth: `calc(min(70vh, 640px) * ${w / h})` }}>
      {body}
      {mode !== "cam" && skipped && <div className="absolute inset-x-0 bottom-0 bg-slate-900/85 px-4 py-3 text-sm text-slate-200">No regions were located. The screening score was under the {data.routing_threshold} gate.</div>}
    </div>
  );
}

const Note = ({ children }) => <div className="absolute inset-0 flex items-center justify-center p-6 text-center text-sm text-slate-400">{children}</div>;
const Card = ({ title, aside, children, className = "" }) => (
  <section className={`rounded-lg bg-white p-5 shadow-sm ring-1 ring-slate-200 ${className}`}>
    <div className="mb-3 flex items-baseline justify-between gap-3"><h2 className="text-base font-semibold">{title}</h2>{aside}</div>
    {children}
  </section>
);

function Pipeline({ d }) {
  const skipped = d.routing_decision === "YOLO26_SKIPPED";
  const steps = [
    ["Screened for chest findings", `Pneumonia score ${d.cihmlc_pneumonia_score.toFixed(2)}, ${d.cihmlc_time_seconds}s`, true],
    [skipped ? "Localizer skipped" : "Localizer started", `${skipped ? "Score is under" : "Score passed"} the ${d.routing_threshold} gate`, !skipped],
    ["Opacities located", skipped ? "Nothing to locate" : `${d.yolo26_detection_count} region${d.yolo26_detection_count === 1 ? "" : "s"}, ${d.yolo26_time_seconds}s`, !skipped],
    ["Report drafted from guidelines", `Grounding score ${d.grounding_audit.grounding_score.toFixed(2)}`, true],
  ];
  return (
    <ol className="space-y-3">
      {steps.map(([t, s, on], i) => (
        <li key={i} className="flex gap-3">
          <span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${on ? "bg-cyan-600 text-white" : "bg-slate-200 text-slate-500"}`}>{i + 1}</span>
          <div><p className={`text-sm font-medium ${on ? "text-slate-900" : "text-slate-500"}`}>{t}</p><p className="text-sm tabular-nums text-slate-500">{s}</p></div>
        </li>
      ))}
    </ol>
  );
}

function Findings({ scores }) {
  const [all, setAll] = useState(false);
  const keys = (all ? Object.keys(scores) : MAIN.filter((k) => k in scores)).sort((a, b) => scores[b] - scores[a]);
  return (
    <>
      <ul className="space-y-2.5">
        {keys.map((k) => {
          const s = scores[k], b = bucket(s);
          return (
            <li key={k}>
              <div className="flex justify-between text-sm"><span className={b === "absent" ? "text-slate-500" : "font-medium"}>{k}</span><span className="tabular-nums text-slate-500">{s.toFixed(2)} {BUCKET_TEXT[b]}</span></div>
              <div className="relative mt-1 h-2 rounded-full bg-slate-100">
                <div className={`h-2 rounded-full ${BAR[b]}`} style={{ width: `${Math.max(1, s * 100)}%` }} />
                <span className="absolute -top-0.5 h-3 w-px bg-slate-400" style={{ left: "12%" }} />
                <span className="absolute -top-0.5 h-3 w-px bg-slate-400" style={{ left: "50%" }} />
              </div>
            </li>
          );
        })}
      </ul>
      <p className="mt-3 text-xs text-slate-500">Ticks mark 0.12 (borderline) and 0.50 (present), the cut-offs the report uses.</p>
      <button onClick={() => setAll(!all)} className="mt-2 text-sm font-medium text-cyan-700 underline-offset-2 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500">{all ? "Show main findings only" : `Show all ${Object.keys(scores).length} model outputs`}</button>
    </>
  );
}

function Inline({ text, audit }) {
  return text.split(/(\*\*[^*]+\*\*|\[[^\[\]]+?\.pdf,\s*p\.\s*\d+\])/gi).map((p, i) => {
    if (/^\*\*/.test(p)) return <strong key={i}>{p.slice(2, -2)}</strong>;
    if (/^\[.+\.pdf/i.test(p)) {
      const c = p.slice(1, -1), ok = audit.verified_citations.includes(c);
      return <span key={i} className={`mx-0.5 whitespace-nowrap rounded px-1.5 py-0.5 font-sans text-xs ring-1 ${ok ? "bg-emerald-50 text-emerald-800 ring-emerald-200" : "bg-rose-50 text-rose-800 ring-rose-200"}`} title={ok ? "Found in retrieved guideline excerpts" : "Not in retrieved excerpts"}>{c}</span>;
    }
    return p;
  });
}

function Report({ text, audit }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try { await navigator.clipboard.writeText(text.replace(/\*\*/g, "")); setCopied(true); setTimeout(() => setCopied(false), 1500); } catch { /* clipboard blocked */ }
  };
  return (
    <Card title="Preliminary report" aside={<button onClick={copy} className="rounded-md px-2.5 py-1 text-sm font-medium text-cyan-700 ring-1 ring-cyan-200 hover:bg-cyan-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500">{copied ? "Copied" : "Copy report"}</button>}>
      <article className="rounded-md bg-stone-50 px-6 py-5 font-serif text-[15px] leading-7 text-slate-800 ring-1 ring-stone-200">
        {text.split("\n").filter((l) => l.trim()).map((l, i) => {
          const h = l.match(/^\*\*([A-Z][A-Z &]+):\*\*\s*(.*)$/);
          if (h && h[1] === "AI DISCLAIMER") return <p key={i} className="mt-5 border-t border-stone-300 pt-3 font-sans text-xs leading-5 text-slate-500">{h[2]}</p>;
          if (h) return <div key={i} className="mt-4 first:mt-0"><h3 className="font-sans text-sm font-semibold text-slate-900">{h[1][0] + h[1].slice(1).toLowerCase()}</h3>{h[2] && <p><Inline text={h[2]} audit={audit} /></p>}</div>;
          const b = l.match(/^\s*[-•]\s+(.*)$/);
          return <p key={i} className={b ? "relative pl-4 before:absolute before:left-0 before:content-['–']" : ""}><Inline text={b ? b[1] : l} audit={audit} /></p>;
        })}
      </article>
    </Card>
  );
}

function Ring({ value }) {
  const r = 26, c = 2 * Math.PI * r, col = value >= 0.8 ? "#059669" : value >= 0.5 ? "#d97706" : "#e11d48";
  return (
    <svg width="64" height="64" viewBox="0 0 64 64" role="img" aria-label={`Grounding score ${Math.round(value * 100)} out of 100`}>
      <circle cx="32" cy="32" r={r} fill="none" stroke="#e2e8f0" strokeWidth="6" />
      <circle cx="32" cy="32" r={r} fill="none" stroke={col} strokeWidth="6" strokeLinecap="round" strokeDasharray={`${c * value} ${c}`} transform="rotate(-90 32 32)" />
      <text x="32" y="37" textAnchor="middle" className="fill-slate-900 text-base font-semibold">{Math.round(value * 100)}</text>
    </svg>
  );
}

function Audit({ a }) {
  const chip = (t, cls, k) => <span key={k} className={`rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ${cls}`}>{t}</span>;
  const bad = a.ungrounded_hallucinations.length + (a.ungrounded_citations || []).length;
  return (
    <Card title="Report check">
      <div className="flex items-center gap-4">
        <Ring value={a.grounding_score} />
        <p className="text-sm text-slate-600">{bad === 0 ? "Every finding and citation in the report matches the model outputs and retrieved guidelines." : `${bad} item${bad === 1 ? "" : "s"} in the report could not be matched. Review them before use.`}</p>
      </div>
      <div className="mt-4 flex flex-wrap gap-1.5">
        {a.verified_positive_mentions.map((m) => chip(`${m} matches`, "bg-emerald-50 text-emerald-800 ring-emerald-200", m))}
        {a.ungrounded_hallucinations.map((h) => chip(`${h.label} not supported (${h.cihmlc_score.toFixed(2)})`, "bg-rose-50 text-rose-800 ring-rose-200", h.label))}
        {(a.ungrounded_citations || []).map((c) => chip(`${c} not retrieved`, "bg-rose-50 text-rose-800 ring-rose-200", c))}
      </div>
    </Card>
  );
}

function Detections({ d }) {
  if (d.routing_decision === "YOLO26_SKIPPED") return <p className="text-sm text-slate-500">The localizer did not run for this study.</p>;
  if (!d.yolo26_detections.length) return <p className="text-sm text-slate-500">The localizer ran and found no regions above the confidence cut-off.</p>;
  return (
    <ul className="divide-y divide-slate-100">
      {d.yolo26_detections.map((x, i) => (
        <li key={i} className="flex items-center justify-between py-2 text-sm">
          <span><span className="mr-2 inline-block h-3 w-3 rounded-sm border-2 border-cyan-400 align-middle" />Region {i + 1}: {describeBox(x.bbox_xywh_norm)}</span>
          <span className="tabular-nums text-slate-500">{x.confidence.toFixed(2)}</span>
        </li>
      ))}
    </ul>
  );
}

function Upload({ onFile, onDemo, error }) {
  const ref = useRef(null);
  const [over, setOver] = useState(false);
  return (
    <div onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={(e) => { e.preventDefault(); setOver(false); e.dataTransfer.files[0] && onFile(e.dataTransfer.files[0]); }}
      className={`mx-auto mt-10 max-w-2xl rounded-xl border-2 border-dashed p-12 text-center ${over ? "border-cyan-500 bg-cyan-50" : "border-slate-300 bg-white"}`}>
      <h1 className="text-2xl font-semibold">Read a chest radiograph</h1>
      <p className="mx-auto mt-2 max-w-md text-slate-600">Drop a JPEG, PNG or DICOM file. Nivara screens for findings, locates opacities, and drafts a report you can edit.</p>
      <input ref={ref} type="file" accept="image/*,.dcm" className="hidden" onChange={(e) => e.target.files[0] && onFile(e.target.files[0])} />
      <div className="mt-6 flex justify-center gap-3">
        <button onClick={() => ref.current.click()} className="rounded-md bg-cyan-700 px-4 py-2 font-medium text-white hover:bg-cyan-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500 focus-visible:ring-offset-2">Choose a file</button>
        <button onClick={onDemo} className="rounded-md px-4 py-2 font-medium text-slate-700 ring-1 ring-slate-300 hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500">Open a demo study</button>
      </div>
      {error && <p role="alert" className="mx-auto mt-5 max-w-md rounded-md bg-rose-50 px-4 py-3 text-sm text-rose-800 ring-1 ring-rose-200">{error}</p>}
    </div>
  );
}

const STAGES = ["Screening for findings", "Checking the gate", "Locating opacities", "Drafting the report"];

export default function NivaraApp() {
  const [data, setData] = useState(null);
  const [src, setSrc] = useState(null);
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState(0);
  const [error, setError] = useState("");
  const [mode, setMode] = useState("det");

  useEffect(() => {
    if (!loading) return;
    const t = setInterval(() => setStage((s) => Math.min(s + 1, STAGES.length - 1)), 2500);
    return () => clearInterval(t);
  }, [loading]);

  const reset = () => { setData(null); setSrc(null); setError(""); setMode("det"); };
  const onFile = async (file) => {
    reset(); setStage(0); setLoading(true);
    if (file.type.startsWith("image/")) setSrc(URL.createObjectURL(file));
    try { setData(await analyze(file)); } catch (e) { setError(e.message || "The study could not be analyzed."); setSrc(null); } finally { setLoading(false); }
  };

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <header className="flex items-center justify-between bg-white px-6 py-3 ring-1 ring-slate-200">
        <div className="flex items-baseline gap-3"><span className="text-lg font-bold tracking-tight">Nivara</span><span className="hidden text-sm text-slate-500 sm:inline">Chest radiograph triage</span></div>
        {(data || loading) && <button onClick={reset} className="rounded-md px-3 py-1.5 text-sm font-medium text-cyan-700 ring-1 ring-cyan-200 hover:bg-cyan-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500">New study</button>}
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6">
        {!data && !loading && <Upload onFile={onFile} onDemo={() => { reset(); setData(DEMO); }} error={error} />}

        {loading && (
          <div className="mx-auto mt-16 max-w-md rounded-xl bg-white p-8 shadow-sm ring-1 ring-slate-200" role="status">
            <h2 className="text-lg font-semibold">Reading the study</h2>
            <ul className="mt-4 space-y-2">
              {STAGES.map((s, i) => (
                <li key={s} className={`flex items-center gap-3 text-sm ${i < stage ? "text-slate-500" : i === stage ? "font-medium text-slate-900" : "text-slate-400"}`}>
                  <span className={`h-2.5 w-2.5 rounded-full ${i < stage ? "bg-cyan-600" : i === stage ? "bg-cyan-500 motion-safe:animate-pulse" : "bg-slate-300"}`} />{s}
                </li>
              ))}
            </ul>
          </div>
        )}

        {data && (
          <div className="grid gap-5 lg:grid-cols-5">
            <section className="rounded-lg bg-slate-900 p-4 lg:col-span-3">
              <div className="mb-3 flex items-center justify-between">
                <p className="truncate text-sm text-slate-300">{data.filename.split("/").pop()}{data.demo && " (demo data)"}</p>
                <div className="flex rounded-md bg-slate-800 p-0.5 text-sm" role="tablist">
                  {[["det", "Located regions"], ["cam", "Grad-CAM"]].map(([k, l]) => (
                    <button key={k} role="tab" aria-selected={mode === k} onClick={() => setMode(k)} className={`rounded px-3 py-1 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 ${mode === k ? "bg-cyan-300 font-semibold text-slate-900" : "text-slate-300 hover:text-white"}`}>{l}</button>
                  ))}
                </div>
              </div>
              <Film data={data} src={src} mode={mode} />
              <p className="mt-3 text-xs text-slate-400">{mode === "cam" ? "Warmer areas show where the screening model looked when scoring pneumonia." : "Boxes show where the localizer found opacities. Confidence is shown beside each region."}</p>
            </section>

            <div className="space-y-5 lg:col-span-2">
              <Card title="How this study was read"><Pipeline d={data} /></Card>
              <Card title="Findings"><Findings scores={data.cihmlc_scores} /></Card>
            </div>

            <div className="lg:col-span-3"><Report text={data.radiology_report} audit={data.grounding_audit} /></div>
            <div className="space-y-5 lg:col-span-2">
              <Audit a={data.grounding_audit} />
              <Card title="Located regions"><Detections d={data} /></Card>
            </div>
          </div>
        )}
      </main>

      <footer className="px-6 pb-8 text-center text-xs text-slate-500">AI-generated and preliminary. A qualified radiologist must confirm every finding before clinical use.</footer>
    </div>
  );
}
