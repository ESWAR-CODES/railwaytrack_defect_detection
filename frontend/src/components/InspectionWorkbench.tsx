import React, { useState, useRef, ChangeEvent, DragEvent } from "react";
import {
  UploadCloud,
  Sliders,
  CheckCircle,
  AlertTriangle,
  Download,
  Maximize2,
  X,
  FileJson,
  FileSpreadsheet,
  RefreshCw,
  Eye,
  Layers,
} from "lucide-react";
import { inspectImage } from "@/services/api";
import { InspectionResponse, SeverityLevel } from "@/types/api";

export const InspectionWorkbench: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [includeCrops, setIncludeCrops] = useState<boolean>(true);
  const [drawHud, setDrawHud] = useState<boolean>(true);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InspectionResponse | null>(null);

  // View mode: "annotated" | "original" | "split"
  const [viewMode, setViewMode] = useState<"annotated" | "original" | "split">("annotated");
  const [lightboxOpen, setLightboxOpen] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const processFile = (file: File) => {
    const validExtensions = [".jpg", ".jpeg", ".png", ".webp"];
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!validExtensions.includes(ext)) {
      setError(`Unsupported file format. Please upload a .jpg, .jpeg, .png, or .webp image.`);
      return;
    }
    setError(null);
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setResult(null); // Clear previous result
  };

  const handleRunInspection = async () => {
    if (!selectedFile) {
      setError("Please select or drop an aerial railway image to inspect.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await inspectImage(selectedFile, confThreshold, includeCrops, drawHud);
      setResult(data);
      setViewMode("annotated");
    } catch (err: any) {
      setError(err.message || "Failed to execute defect inspection on the backend.");
    } finally {
      setLoading(false);
    }
  };

  // Quick sample generator for immediate testing without local drone images
  const handleLoadSyntheticSample = () => {
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 640;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Draw realistic synthetic railway segment
    ctx.fillStyle = "#80776b"; // Ballast gravel
    ctx.fillRect(0, 0, 640, 640);

    // Sleepers (ties)
    for (let y = 40; y < 640; y += 65) {
      ctx.fillStyle = "#3e2723";
      ctx.fillRect(80, y, 480, 28);
      // Fastener plates
      ctx.fillStyle = "#78909c";
      ctx.fillRect(170, y - 4, 20, 36);
      ctx.fillRect(450, y - 4, 20, 36);
    }

    // Steel rails
    ctx.fillStyle = "#b0bec5";
    ctx.fillRect(175, 0, 12, 640);
    ctx.fillRect(455, 0, 12, 640);

    // Simulated defect: surface crack & corrosion
    ctx.fillStyle = "#b71c1c";
    ctx.beginPath();
    ctx.moveTo(176, 280);
    ctx.lineTo(185, 310);
    ctx.lineTo(178, 330);
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#212121";
    ctx.stroke();

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], "synthetic_rail_track_sample.jpg", { type: "image/jpeg" });
        processFile(file);
      }
    }, "image/jpeg");
  };

  const downloadAnnotatedImage = () => {
    if (!result?.annotated_image_base64) return;
    const link = document.createElement("a");
    link.href = `data:image/jpeg;base64,${result.annotated_image_base64}`;
    link.download = `annotated_${result.filename || "inspection"}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportJsonReport = () => {
    if (!result) return;
    const jsonStr = JSON.stringify(result, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `railway_inspection_${result.inspection_id}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportCsvReport = () => {
    if (!result || !result.detections) return;
    const headers = [
      "Defect Index",
      "Class ID",
      "Class Name",
      "Confidence (%)",
      "Severity",
      "Area Ratio",
      "Diagonal Ratio",
      "BBox (x1, y1, x2, y2)",
      "Recommended Action",
    ];
    const rows = result.detections.map((d, i) => [
      i + 1,
      d.class_id,
      d.class_name,
      (d.confidence * 100).toFixed(1),
      d.severity,
      d.area_ratio,
      d.diagonal_ratio,
      `"[${d.bbox_xyxy.join(", ")}]"`,
      `"${d.recommended_action.replace(/"/g, '""')}"`,
    ]);

    const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `defect_triage_${result.inspection_id}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getSeverityBadgeClass = (severity: SeverityLevel) => {
    switch (severity) {
      case "HIGH":
        return "bg-red-100 text-red-800 border-red-300 font-bold";
      case "MEDIUM":
        return "bg-orange-100 text-orange-800 border-orange-300 font-bold";
      case "LOW":
        return "bg-amber-100 text-amber-800 border-amber-300 font-bold";
      default:
        return "bg-slate-100 text-slate-800 border-slate-300 font-bold";
    }
  };

  const getOverallRiskClass = (level: string) => {
    switch (level) {
      case "HIGH":
        return "bg-red-950 text-red-300 border-red-700";
      case "MEDIUM":
        return "bg-orange-950 text-orange-300 border-orange-700";
      case "LOW":
        return "bg-amber-950 text-amber-300 border-amber-700";
      default:
        return "bg-emerald-950 text-emerald-300 border-emerald-700";
    }
  };

  return (
    <section id="workbench" className="space-y-8">
      {/* Section Header */}
      <div className="border-b border-border-muted pb-4 flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h2 className="font-editorial text-2xl sm:text-3xl font-bold text-navy">
            Drone Image Inspection Workbench
          </h2>
          <p className="text-xs text-text-muted mt-1">
            Upload aerial drone imagery to run real-time YOLOv11 defect inference and automated severity classification.
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleLoadSyntheticSample}
            className="text-xs font-telemetry px-3 py-1.5 rounded-btn bg-surface hover:bg-slate-200 text-navy border border-border-muted transition-colors flex items-center space-x-1"
          >
            <RefreshCw className="w-3.5 h-3.5 text-gold" />
            <span>Load Track Test Pattern</span>
          </button>
        </div>
      </div>

      {/* Control & Configuration Strip */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Col: Upload Dropzone & Controls (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          {/* File Dropzone */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-card p-6 text-center cursor-pointer transition-all bg-surface-card ${
              isDragging
                ? "border-gold bg-amber-50/50"
                : selectedFile
                ? "border-navy/40 bg-surface/30"
                : "border-border-muted hover:border-gold/60"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp"
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="flex flex-col items-center justify-center space-y-2">
              <div className="w-12 h-12 rounded-full bg-navy/5 flex items-center justify-center text-navy">
                <UploadCloud className="w-6 h-6 text-navy" />
              </div>
              <div>
                <p className="text-xs font-medium text-navy">
                  {selectedFile ? (
                    <span className="font-telemetry font-bold text-navy">{selectedFile.name}</span>
                  ) : (
                    <>
                      <span className="font-semibold text-gold">Click to upload</span> or drag and drop aerial image
                    </>
                  )}
                </p>
                <p className="text-[11px] text-text-muted mt-0.5 font-telemetry">
                  Supports .JPG, .JPEG, .PNG, .WEBP (up to 25MB)
                </p>
              </div>
              {selectedFile && (
                <div className="text-[10px] font-telemetry text-slate-500 bg-surface px-2 py-0.5 rounded">
                  {(selectedFile.size / 1024).toFixed(1)} KB
                </div>
              )}
            </div>
          </div>

          {/* Inference Parameters Box */}
          <div className="bg-surface-card rounded-card p-5 border border-border-muted shadow-card space-y-4">
            <div className="flex items-center justify-between border-b border-border-muted pb-3">
              <span className="text-xs font-telemetry uppercase tracking-wider font-bold text-navy flex items-center space-x-1.5">
                <Sliders className="w-3.5 h-3.5 text-gold" />
                <span>Inference Parameters</span>
              </span>
              <span className="text-[11px] font-telemetry text-gold font-bold bg-navy px-2 py-0.5 rounded-btn">
                Conf: {confThreshold.toFixed(2)}
              </span>
            </div>

            {/* Confidence Slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs font-telemetry text-text-muted">
                <span>Confidence Cutoff</span>
                <span className="text-navy font-semibold">{confThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.95"
                step="0.01"
                value={confThreshold}
                onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                className="w-full accent-gold h-1.5 bg-surface rounded-lg cursor-pointer"
              />
              <div className="flex justify-between text-[10px] font-telemetry text-slate-400">
                <span>0.05 (High Recall)</span>
                <span>0.25 (Default)</span>
                <span>0.95 (High Precision)</span>
              </div>
            </div>

            {/* Toggles */}
            <div className="pt-2 grid grid-cols-2 gap-3 text-xs font-sans">
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeCrops}
                  onChange={(e) => setIncludeCrops(e.target.checked)}
                  className="rounded border-slate-300 text-gold focus:ring-gold accent-gold"
                />
                <span className="text-slate-700 font-medium">Extract Defect Crops</span>
              </label>
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={drawHud}
                  onChange={(e) => setDrawHud(e.target.checked)}
                  className="rounded border-slate-300 text-gold focus:ring-gold accent-gold"
                />
                <span className="text-slate-700 font-medium">Render Telemetry HUD</span>
              </label>
            </div>

            {/* Execute Analysis Button */}
            <button
              onClick={handleRunInspection}
              disabled={loading || !selectedFile}
              className={`w-full py-3 rounded-btn font-semibold text-xs uppercase tracking-wider transition-all flex items-center justify-center space-x-2 shadow-sm ${
                loading || !selectedFile
                  ? "bg-slate-300 text-slate-500 cursor-not-allowed"
                  : "bg-gold hover:bg-gold-hover text-navy shadow-glow hover:translate-y-[-1px]"
              }`}
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-navy" />
                  <span>Processing YOLOv11 & Severity Engine...</span>
                </>
              ) : (
                <>
                  <Eye className="w-4 h-4" />
                  <span>Run Defect Analysis</span>
                </>
              )}
            </button>
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-red-50 border border-red-300 text-red-800 text-xs p-3.5 rounded-btn flex items-start space-x-2">
              <AlertTriangle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
              <div className="space-y-0.5">
                <p className="font-bold">Inspection Error</p>
                <p>{error}</p>
              </div>
            </div>
          )}
        </div>

        {/* Right Col: Interactive Visual Viewer (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-surface-card rounded-card border border-border-muted shadow-card overflow-hidden">
            {/* Viewer Header */}
            <div className="px-4 py-3 bg-navy text-white flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className="font-telemetry text-xs font-bold text-gold">VISION CANVAS</span>
                {result && (
                  <span className="text-[11px] font-telemetry text-slate-300">
                    • {result.image_width} × {result.image_height}px
                  </span>
                )}
              </div>

              {/* View Mode Switcher */}
              <div className="flex items-center space-x-1 bg-navy-surface p-1 rounded-btn text-xs font-telemetry">
                <button
                  onClick={() => setViewMode("annotated")}
                  disabled={!result?.annotated_image_base64}
                  className={`px-2 py-0.5 rounded-btn transition-colors ${
                    viewMode === "annotated"
                      ? "bg-gold text-navy font-bold"
                      : "text-slate-300 hover:text-white"
                  }`}
                >
                  YOLO Overlay
                </button>
                <button
                  onClick={() => setViewMode("original")}
                  disabled={!previewUrl}
                  className={`px-2 py-0.5 rounded-btn transition-colors ${
                    viewMode === "original"
                      ? "bg-gold text-navy font-bold"
                      : "text-slate-300 hover:text-white"
                  }`}
                >
                  Original
                </button>
                <button
                  onClick={() => setViewMode("split")}
                  disabled={!result?.annotated_image_base64 || !previewUrl}
                  className={`px-2 py-0.5 rounded-btn transition-colors ${
                    viewMode === "split"
                      ? "bg-gold text-navy font-bold"
                      : "text-slate-300 hover:text-white"
                  }`}
                >
                  Side-by-Side
                </button>
              </div>
            </div>

            {/* Image Canvas Container */}
            <div className="relative min-h-[380px] bg-slate-900 flex items-center justify-center p-2">
              {!previewUrl && (
                <div className="text-center text-slate-500 space-y-2 p-8">
                  <Layers className="w-10 h-10 mx-auto text-slate-600" />
                  <p className="text-xs font-telemetry">No drone image loaded</p>
                  <p className="text-[11px] text-slate-600">
                    Upload an aerial track image or click "Load Track Test Pattern" to begin
                  </p>
                </div>
              )}

              {previewUrl && !result && (
                <div className="relative max-h-[500px] overflow-hidden flex items-center justify-center">
                  <img
                    src={previewUrl}
                    alt="Original railway preview"
                    className="max-h-[480px] w-auto object-contain rounded"
                  />
                  <div className="absolute bottom-3 left-3 bg-navy/80 text-white text-[11px] font-telemetry px-2.5 py-1 rounded-btn backdrop-blur-sm border border-navy-light">
                    Pending Analysis • Click "Run Defect Analysis"
                  </div>
                </div>
              )}

              {result && (
                <div className="w-full flex items-center justify-center">
                  {viewMode === "annotated" && result.annotated_image_base64 && (
                    <div className="relative max-h-[500px]">
                      <img
                        src={`data:image/jpeg;base64,${result.annotated_image_base64}`}
                        alt="Annotated defect overlay"
                        className="max-h-[480px] w-auto object-contain rounded"
                      />
                    </div>
                  )}

                  {viewMode === "original" && previewUrl && (
                    <div className="relative max-h-[500px]">
                      <img
                        src={previewUrl}
                        alt="Original railway view"
                        className="max-h-[480px] w-auto object-contain rounded"
                      />
                    </div>
                  )}

                  {viewMode === "split" && (
                    <div className="grid grid-cols-2 gap-2 w-full max-h-[500px]">
                      <div className="text-center space-y-1">
                        <span className="text-[10px] font-telemetry text-slate-400">ORIGINAL</span>
                        <img
                          src={previewUrl || ""}
                          alt="Original"
                          className="max-h-[440px] w-full object-contain rounded border border-slate-700"
                        />
                      </div>
                      <div className="text-center space-y-1">
                        <span className="text-[10px] font-telemetry text-gold">YOLO ANNOTATED</span>
                        <img
                          src={`data:image/jpeg;base64,${result.annotated_image_base64}`}
                          alt="Annotated"
                          className="max-h-[440px] w-full object-contain rounded border border-slate-700"
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Viewer Overlay Controls */}
              {result && (
                <div className="absolute top-4 right-4 flex items-center space-x-1.5">
                  <button
                    onClick={() => setLightboxOpen(true)}
                    className="p-1.5 bg-navy/80 hover:bg-navy text-white rounded-btn text-xs border border-navy-light shadow backdrop-blur-sm"
                    title="Fullscreen Lightbox"
                  >
                    <Maximize2 className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={downloadAnnotatedImage}
                    className="p-1.5 bg-gold hover:bg-gold-hover text-navy font-bold rounded-btn text-xs shadow"
                    title="Download Annotated Image"
                  >
                    <Download className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}
            </div>

            {/* Viewer Footer Bar */}
            {result && (
              <div className="bg-surface p-3 border-t border-border-muted flex flex-wrap items-center justify-between text-xs font-telemetry text-slate-600 gap-2">
                <div>
                  <span className="font-semibold text-navy">Inspection ID:</span> {result.inspection_id.slice(0, 8)}...
                </div>
                <div className="flex items-center space-x-3">
                  <button
                    onClick={exportJsonReport}
                    className="flex items-center space-x-1 text-navy hover:text-gold font-medium"
                  >
                    <FileJson className="w-3.5 h-3.5" />
                    <span>JSON Report</span>
                  </button>
                  <button
                    onClick={exportCsvReport}
                    className="flex items-center space-x-1 text-navy hover:text-gold font-medium"
                  >
                    <FileSpreadsheet className="w-3.5 h-3.5" />
                    <span>CSV Triage</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Telemetry Summary Strip (Displayed upon successful inspection) */}
      {result && (
        <div className="bg-surface-card rounded-card border border-border-muted shadow-card p-6 space-y-5">
          <div className="flex items-center justify-between border-b border-border-muted pb-3">
            <h3 className="font-editorial text-lg font-bold text-navy">
              Real-Time Defect Telemetry & Severity Assessment
            </h3>
            <div className={`px-3 py-1 rounded-btn text-xs font-telemetry font-bold border ${getOverallRiskClass(result.summary.overall_risk_level)}`}>
              RISK: {result.summary.overall_risk_level}
            </div>
          </div>

          {/* Telemetry KPI Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {/* KPI 1: Track Risk Index */}
            <div className="bg-surface p-4 rounded-card border border-border-muted space-y-1">
              <div className="text-[11px] font-telemetry text-text-muted font-bold uppercase">
                Track Risk Index (TRI)
              </div>
              <div className="flex items-baseline space-x-1">
                <span className="font-telemetry text-2xl font-black text-navy">
                  {result.summary.track_risk_index.toFixed(1)}
                </span>
                <span className="font-telemetry text-xs text-slate-500">/ 100</span>
              </div>
              {/* Progress gauge bar */}
              <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden mt-2">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${Math.min(100, result.summary.track_risk_index)}%`,
                    backgroundColor: result.summary.overall_color_hex || "#10B981",
                  }}
                ></div>
              </div>
            </div>

            {/* KPI 2: Total Defects Detected */}
            <div className="bg-surface p-4 rounded-card border border-border-muted space-y-1">
              <div className="text-[11px] font-telemetry text-text-muted font-bold uppercase">
                Total Detected Defects
              </div>
              <div className="font-telemetry text-2xl font-black text-navy">
                {result.summary.total_defects}
              </div>
              <div className="text-[11px] text-text-muted font-telemetry">
                {result.detections.length === 0 ? "Track Clearance Nominal" : "Defect Bounding Boxes Flagged"}
              </div>
            </div>

            {/* KPI 3: Severity Distribution */}
            <div className="bg-surface p-4 rounded-card border border-border-muted space-y-1">
              <div className="text-[11px] font-telemetry text-text-muted font-bold uppercase">
                Severity Breakdown
              </div>
              <div className="flex items-center space-x-2 pt-1 font-telemetry text-xs font-bold">
                <span className="px-1.5 py-0.5 rounded bg-red-100 text-red-800" title="High Severity">
                  H: {result.summary.counts_by_severity?.HIGH || 0}
                </span>
                <span className="px-1.5 py-0.5 rounded bg-orange-100 text-orange-800" title="Medium Severity">
                  M: {result.summary.counts_by_severity?.MEDIUM || 0}
                </span>
                <span className="px-1.5 py-0.5 rounded bg-amber-100 text-amber-800" title="Low Severity">
                  L: {result.summary.counts_by_severity?.LOW || 0}
                </span>
              </div>
              <div className="text-[10px] text-slate-400 font-telemetry">
                Weighted Severity Formula
              </div>
            </div>

            {/* KPI 4: Inference Speed & Resolution */}
            <div className="bg-surface p-4 rounded-card border border-border-muted space-y-1">
              <div className="text-[11px] font-telemetry text-text-muted font-bold uppercase">
                Inspection Metadata
              </div>
              <div className="font-telemetry text-xs text-navy font-semibold">
                {result.filename}
              </div>
              <div className="text-[11px] font-telemetry text-slate-500">
                {result.image_width} × {result.image_height}px • Conf: {result.confidence_threshold}
              </div>
            </div>
          </div>

          {/* Actionable Maintenance Recommendation Banner */}
          <div className="bg-navy-surface text-white p-4 rounded-btn border border-navy-light flex items-start space-x-3">
            <CheckCircle className="w-5 h-5 text-gold shrink-0 mt-0.5" />
            <div className="space-y-0.5 text-xs">
              <span className="font-telemetry font-bold text-gold uppercase tracking-wider">
                Recommended Maintenance Advisory:
              </span>
              <p className="text-slate-200 leading-relaxed font-sans">
                {result.summary.recommended_action || "Track clearance nominal. No immediate speed restriction required."}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Defect Triage Table & Crops Gallery */}
      {result && result.detections.length > 0 && (
        <div className="bg-surface-card rounded-card border border-border-muted shadow-card p-6 space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border-muted pb-3">
            <div>
              <h3 className="font-editorial text-lg font-bold text-navy">
                Defect Triage Registry ({result.detections.length} Detected)
              </h3>
              <p className="text-xs text-text-muted mt-0.5">
                Itemized defect list ranked by location, geometric ratios, and severity weighting.
              </p>
            </div>
            <div className="text-xs font-telemetry text-slate-500">
              Crop Patches Extracted: {result.crops_base64?.length || 0}
            </div>
          </div>

          {/* Table Container */}
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-surface text-text-muted uppercase font-telemetry text-[10px] tracking-wider border-b border-border-muted">
                  <th className="py-2.5 px-3">#</th>
                  <th className="py-2.5 px-3">Patch</th>
                  <th className="py-2.5 px-3">Class</th>
                  <th className="py-2.5 px-3">Confidence</th>
                  <th className="py-2.5 px-3">Severity</th>
                  <th className="py-2.5 px-3">Geometric Ratios</th>
                  <th className="py-2.5 px-3">Bounding Box [xyxy]</th>
                  <th className="py-2.5 px-3">Maintenance Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border-muted">
                {result.detections.map((defect, index) => {
                  const cropBase64 = result.crops_base64?.[index];
                  return (
                    <tr key={index} className="hover:bg-surface/50 transition-colors">
                      <td className="py-3 px-3 font-telemetry font-bold text-slate-400">
                        {index + 1}
                      </td>
                      <td className="py-3 px-3">
                        {cropBase64 ? (
                          <img
                            src={`data:image/jpeg;base64,${cropBase64}`}
                            alt={defect.class_name}
                            className="w-12 h-12 object-cover rounded border border-border-muted hover:scale-150 transition-transform origin-left shadow-sm bg-black"
                            title="Hover to magnify defect crop"
                          />
                        ) : (
                          <div className="w-12 h-12 rounded bg-surface border border-border-muted flex items-center justify-center text-[10px] text-slate-400">
                            N/A
                          </div>
                        )}
                      </td>
                      <td className="py-3 px-3">
                        <span className="font-telemetry font-bold text-navy block">
                          {defect.class_name}
                        </span>
                        <span className="text-[10px] text-slate-400 font-telemetry">
                          ID: {defect.class_id}
                        </span>
                      </td>
                      <td className="py-3 px-3 font-telemetry font-semibold text-navy">
                        {(defect.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-3">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-telemetry uppercase border ${getSeverityBadgeClass(defect.severity)}`}>
                          {defect.severity}
                        </span>
                      </td>
                      <td className="py-3 px-3 font-telemetry text-[11px] text-slate-600">
                        <div>Area: {defect.area_ratio.toFixed(4)}</div>
                        <div>Diag: {defect.diagonal_ratio.toFixed(4)}</div>
                      </td>
                      <td className="py-3 px-3 font-telemetry text-[11px] text-slate-500 whitespace-nowrap">
                        [{defect.bbox_xyxy.map((c) => Math.round(c)).join(", ")}]
                      </td>
                      <td className="py-3 px-3 text-xs text-slate-700 max-w-xs leading-relaxed">
                        {defect.recommended_action}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Fullscreen Lightbox Modal */}
      {lightboxOpen && result?.annotated_image_base64 && (
        <div className="fixed inset-0 z-50 bg-black/90 flex flex-col items-center justify-center p-4">
          <div className="w-full max-w-6xl flex justify-between items-center text-white mb-3">
            <span className="font-telemetry text-xs font-bold text-gold">
              INSPECTION LIGHTBOX • {result.filename}
            </span>
            <button
              onClick={() => setLightboxOpen(false)}
              className="p-1.5 rounded-btn bg-slate-800 hover:bg-slate-700 text-white"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          <div className="max-h-[85vh] max-w-full overflow-auto flex items-center justify-center">
            <img
              src={`data:image/jpeg;base64,${result.annotated_image_base64}`}
              alt="Full size annotated inspection"
              className="max-h-[80vh] w-auto object-contain rounded"
            />
          </div>
        </div>
      )}
    </section>
  );
};
