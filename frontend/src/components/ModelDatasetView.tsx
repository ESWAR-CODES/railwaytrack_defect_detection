import React, { useEffect, useState } from "react";
import { Cpu, Database, BarChart3, RefreshCw, AlertCircle, HardDrive, CheckCircle2 } from "lucide-react";
import { getMetrics } from "@/services/api";
import { HealthResponse, MetricsResponse } from "@/types/api";

interface ModelDatasetViewProps {
  health: HealthResponse | null;
}

export const ModelDatasetView: React.FC<ModelDatasetViewProps> = ({ health }) => {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetricsData = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getMetrics();
      setMetrics(data);
    } catch (err: any) {
      setError(err.message || "Failed to retrieve evaluation metrics from server.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetricsData();
  }, []);

  const datasetStats = {
    totalRaw: 4975,
    datasetA: 2651,
    datasetB: 2324,
    droppedNormalLabels: 144, // fastener_present
    augmentedSamples: 696,
    totalProcessed: 5671,
    splits: {
      train: { count: 4178, pct: "73.7%" },
      val: { count: 995, pct: "17.5%" },
      test: { count: 498, pct: "8.8%" },
    },
    augmentations: [
      "Safe Rotation (±15°)",
      "Aerial Motion Blur",
      "Atmospheric Fog & Rain Weather Effects",
      "Dynamic Solar Illumination (Brightness/Contrast)",
    ],
  };

  return (
    <section id="telemetry" className="space-y-8">
      {/* Section Header */}
      <div className="border-b border-border-muted pb-4 flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h2 className="font-editorial text-2xl sm:text-3xl font-bold text-navy">
            Model Evaluation & Dataset Telemetry
          </h2>
          <p className="text-xs text-text-muted mt-1">
            Real-time Ultralytics YOLOv11 checkpoint metrics, hardware execution parameters, and harmonized multi-source dataset splits.
          </p>
        </div>
        <button
          onClick={fetchMetricsData}
          className="text-xs font-telemetry px-3 py-1.5 rounded-btn bg-surface hover:bg-slate-200 text-navy border border-border-muted transition-colors flex items-center space-x-1"
        >
          <RefreshCw className={`w-3.5 h-3.5 text-gold ${loading ? "animate-spin" : ""}`} />
          <span>Refresh Metrics</span>
        </button>
      </div>

      {/* Model Status & Device Banner */}
      <div className="bg-navy text-white rounded-card p-6 shadow-card border border-navy-surface flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-gold"></span>
            <span className="font-telemetry text-xs uppercase tracking-wider text-gold font-bold">
              Active Neural Checkpoint
            </span>
          </div>
          <h3 className="font-editorial text-xl font-bold">
            {health?.model_name || "YOLOv11 Nano (yolo11n.pt)"}
          </h3>
          <p className="text-xs text-slate-300 font-telemetry">
            Weights: <span className="text-slate-100">{health?.weights_path || "Default Checkpoint"}</span>
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 font-telemetry text-xs">
          <div className="px-3.5 py-2 rounded-btn bg-navy-surface border border-navy-light text-center">
            <div className="text-[10px] text-slate-400 font-bold">DEVICE RUNTIME</div>
            <div className="font-bold text-gold uppercase">{metrics?.device || "CPU"}</div>
          </div>
          <div className="px-3.5 py-2 rounded-btn bg-navy-surface border border-navy-light text-center">
            <div className="text-[10px] text-slate-400 font-bold">TARGET CLASSES</div>
            <div className="font-bold text-emerald-400">5 Canonical</div>
          </div>
          <div className="px-3.5 py-2 rounded-btn bg-navy-surface border border-navy-light text-center">
            <div className="text-[10px] text-slate-400 font-bold">TRAINING STATUS</div>
            <div className="font-bold text-white uppercase">{metrics?.status || "PENDING"}</div>
          </div>
        </div>
      </div>

      {/* Metrics Cards Grid */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-editorial text-lg font-bold text-navy flex items-center space-x-2">
            <BarChart3 className="w-4 h-4 text-gold" />
            <span>Validation Metrics (Test Split Benchmark)</span>
          </h3>
          {metrics?.timestamp && (
            <span className="text-[11px] font-telemetry text-slate-500">
              Evaluated on {new Date(metrics.timestamp).toLocaleString()}
            </span>
          )}
        </div>

        {/* Informative banner if metrics unavailable or pending full training */}
        {metrics?.status === "unavailable" && (
          <div className="bg-amber-50 border border-amber-300 text-amber-900 text-xs p-4 rounded-card flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="font-bold">Evaluation Metrics Pending Full Training Run</p>
              <p>
                The backend is currently operational in fast-inference mode. Execute <code className="font-telemetry bg-amber-100 px-1 py-0.5 rounded text-amber-950">python backend/scripts/train_yolo.py --epochs 50</code> to generate complete benchmark evaluation metrics.
              </p>
            </div>
          </div>
        )}

        {/* Metric Value Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-surface-card p-5 rounded-card border border-border-muted shadow-card space-y-1">
            <div className="text-[10px] font-telemetry uppercase text-text-muted font-bold">
              mAP @ 0.50 IoU
            </div>
            <div className="font-telemetry text-2xl font-black text-navy">
              {metrics?.metrics?.mAP50 !== undefined
                ? (metrics.metrics.mAP50 * 100).toFixed(2) + "%"
                : "—"}
            </div>
            <div className="text-[10px] font-telemetry text-slate-400">Mean Average Precision</div>
          </div>

          <div className="bg-surface-card p-5 rounded-card border border-border-muted shadow-card space-y-1">
            <div className="text-[10px] font-telemetry uppercase text-text-muted font-bold">
              mAP @ 0.50:0.95
            </div>
            <div className="font-telemetry text-2xl font-black text-navy">
              {metrics?.metrics?.mAP50_95 !== undefined
                ? (metrics.metrics.mAP50_95 * 100).toFixed(2) + "%"
                : "—"}
            </div>
            <div className="text-[10px] font-telemetry text-slate-400">Strict Multi-IoU Benchmark</div>
          </div>

          <div className="bg-surface-card p-5 rounded-card border border-border-muted shadow-card space-y-1">
            <div className="text-[10px] font-telemetry uppercase text-text-muted font-bold">
              Precision
            </div>
            <div className="font-telemetry text-2xl font-black text-navy">
              {metrics?.metrics?.precision !== undefined
                ? (metrics.metrics.precision * 100).toFixed(2) + "%"
                : "—"}
            </div>
            <div className="text-[10px] font-telemetry text-slate-400">True Positive Fidelity</div>
          </div>

          <div className="bg-surface-card p-5 rounded-card border border-border-muted shadow-card space-y-1">
            <div className="text-[10px] font-telemetry uppercase text-text-muted font-bold">
              Recall
            </div>
            <div className="font-telemetry text-2xl font-black text-navy">
              {metrics?.metrics?.recall !== undefined
                ? (metrics.metrics.recall * 100).toFixed(2) + "%"
                : "—"}
            </div>
            <div className="text-[10px] font-telemetry text-slate-400">Defect Detection Coverage</div>
          </div>
        </div>

        {/* Per-Class Performance Breakdown */}
        {metrics?.metrics?.per_class_map && (
          <div className="bg-surface-card p-5 rounded-card border border-border-muted shadow-card space-y-3">
            <div className="text-xs font-telemetry uppercase font-bold text-navy">
              Per-Class mAP@50 Benchmark
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
              {Object.entries(metrics.metrics.per_class_map).map(([className, score]) => (
                <div key={className} className="bg-surface p-3 rounded-btn border border-border-muted space-y-1">
                  <div className="text-[11px] font-telemetry font-bold text-navy truncate" title={className}>
                    {className}
                  </div>
                  <div className="font-telemetry text-base font-bold text-navy">
                    {(Number(score) * 100).toFixed(2)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Dataset Harmonization & Drone Augmentation Summary */}
      <div className="space-y-4">
        <h3 className="font-editorial text-lg font-bold text-navy flex items-center space-x-2">
          <Database className="w-4 h-4 text-gold" />
          <span>Dataset Architecture & Augmentation Telemetry</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {/* Col 1: Dataset Merging */}
          <div className="bg-surface-card p-5 rounded-card border border-border-muted shadow-card space-y-3">
            <div className="flex items-center justify-between border-b border-border-muted pb-2">
              <span className="font-telemetry text-xs font-bold text-navy">Raw Ingestion</span>
              <span className="text-[10px] font-telemetry px-2 py-0.5 bg-surface text-slate-600 rounded">
                Deduplicated
              </span>
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between font-telemetry">
                <span className="text-text-muted">Dataset A (Roboflow source):</span>
                <span className="font-bold text-navy">{datasetStats.datasetA.toLocaleString()}</span>
              </div>
              <div className="flex justify-between font-telemetry">
                <span className="text-text-muted">Dataset B (Kaggle/Roboflow):</span>
                <span className="font-bold text-navy">{datasetStats.datasetB.toLocaleString()}</span>
              </div>
              <div className="flex justify-between font-telemetry pt-1 border-t border-border-muted">
                <span className="font-bold text-navy">Total Ingested Raw:</span>
                <span className="font-bold text-navy">{datasetStats.totalRaw.toLocaleString()}</span>
              </div>
              <div className="text-[11px] text-amber-700 bg-amber-50 p-2 rounded">
                Cleaned: 144 non-defect fastener_present labels dropped.
              </div>
            </div>
          </div>

          {/* Col 2: Deterministic Splits */}
          <div className="bg-surface-card p-5 rounded-card border border-border-muted shadow-card space-y-3">
            <div className="flex items-center justify-between border-b border-border-muted pb-2">
              <span className="font-telemetry text-xs font-bold text-navy">Split Distribution</span>
              <span className="text-[10px] font-telemetry px-2 py-0.5 bg-surface text-slate-600 rounded">
                Seed: 42
              </span>
            </div>
            <div className="space-y-2 text-xs font-telemetry">
              <div className="flex justify-between">
                <span className="text-text-muted">Train Split (70%):</span>
                <span className="font-bold text-navy">{datasetStats.splits.train.count} ({datasetStats.splits.train.pct})</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Val Split (20%):</span>
                <span className="font-bold text-navy">{datasetStats.splits.val.count} ({datasetStats.splits.val.pct})</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Test Split (10%):</span>
                <span className="font-bold text-navy">{datasetStats.splits.test.count} ({datasetStats.splits.test.pct})</span>
              </div>
              <div className="flex justify-between pt-1 border-t border-border-muted font-bold text-navy">
                <span>Total Processed:</span>
                <span>{datasetStats.totalProcessed.toLocaleString()} images</span>
              </div>
            </div>
          </div>

          {/* Col 3: Drone Augmentations */}
          <div className="bg-surface-card p-5 rounded-card border border-border-muted shadow-card space-y-3">
            <div className="flex items-center justify-between border-b border-border-muted pb-2">
              <span className="font-telemetry text-xs font-bold text-navy">Aerial Augmentation</span>
              <span className="text-[10px] font-telemetry px-2 py-0.5 bg-emerald-100 text-emerald-800 rounded font-bold">
                +696 samples
              </span>
            </div>
            <ul className="text-xs space-y-1.5 text-text-muted font-sans">
              {datasetStats.augmentations.map((aug, i) => (
                <li key={i} className="flex items-center space-x-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                  <span>{aug}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
};
