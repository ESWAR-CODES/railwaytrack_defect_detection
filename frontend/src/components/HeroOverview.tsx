import React from "react";
import { ArrowRight, AlertTriangle, Layers, Crosshair, Cpu, ShieldAlert, CheckCircle2 } from "lucide-react";

interface HeroOverviewProps {
  onLaunchWorkbench: () => void;
}

export const HeroOverview: React.FC<HeroOverviewProps> = ({ onLaunchWorkbench }) => {
  const defectClasses = [
    {
      id: 0,
      name: "rail_crack",
      displayTitle: "Rail Crack & Joint Fracture",
      severityRule: "Geometric Diagonal Ratio (Low <0.03, Medium ≤0.08, High >0.08)",
      riskWeight: "10 / 25 / 50 pts",
      badgeColor: "bg-red-950 text-red-300 border-red-700",
      description: "Structural fissures, squats, head checks, and transverse/longitudinal web breaks.",
      icon: AlertTriangle,
    },
    {
      id: 1,
      name: "missing_fastener",
      displayTitle: "Missing Fastener & Clip",
      severityRule: "Density Escalation (1 = Low, 2 = Medium, ≥3 = High Emergency)",
      riskWeight: "10 / 25 / 50 pts",
      badgeColor: "bg-orange-950 text-orange-300 border-orange-700",
      description: "Dislodged Pandrol clips, absent tie-plate bolts, or loose elastic rail fastenings.",
      icon: Crosshair,
    },
    {
      id: 2,
      name: "rail_corrosion",
      displayTitle: "Rail Corrosion & Surface Rust",
      severityRule: "Surface Area Coverage (Low <0.05, Medium ≤0.15, High >0.15)",
      riskWeight: "10 / 25 / 50 pts",
      badgeColor: "bg-amber-950 text-amber-300 border-amber-700",
      description: "Atmospheric oxidation, material pitting, and metal flaking degrading rail strength.",
      icon: Layers,
    },
    {
      id: 3,
      name: "obstacle",
      displayTitle: "Track Obstacle & Foreign Object",
      severityRule: "Mandatory Critical Severity (Direct High Escalation)",
      riskWeight: "50 pts (Critical)",
      badgeColor: "bg-red-950 text-red-300 border-red-700",
      description: "Right-of-way intrusions, fallen debris, boulders, or misplaced equipment on ballast.",
      icon: ShieldAlert,
    },
    {
      id: 4,
      name: "sleeper_misalignment",
      displayTitle: "Sleeper Angular Misalignment",
      severityRule: "Cluster Count Escalation (1 = Low, 2 = Medium, ≥3 = High Hazard)",
      riskWeight: "10 / 25 / 50 pts",
      badgeColor: "bg-orange-950 text-orange-300 border-orange-700",
      description: "Concrete/timber tie rotation, ballast subsidence, or sleeper spacing irregularity.",
      icon: Cpu,
    },
  ];

  return (
    <section className="space-y-16 pb-12">
      {/* Editorial Navy Hero Section */}
      <div className="bg-navy text-white rounded-card p-8 md:p-14 shadow-editorial relative overflow-hidden border border-navy-surface">
        {/* Subtle grid background texture */}
        <div className="absolute inset-0 opacity-5 pointer-events-none bg-[radial-gradient(#F5A623_1px,transparent_1px)] [background-size:16px_16px]"></div>

        <div className="relative z-10 max-w-4xl space-y-6">
          {/* Academic Badge */}
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-btn bg-navy-surface border border-navy-light text-xs font-telemetry text-gold">
            <span className="w-2 h-2 rounded-full bg-gold"></span>
            <span>MBU DEPT. OF CSE (AI & ML) • TEAM B6-B3 • SUPERVISOR: MS. K. SANDHYA</span>
          </div>

          {/* Editorial Display Heading */}
          <h1 className="font-editorial text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight leading-tight">
            AI-Based Multi-Defect Railway Monitoring & Severity Assessment Using Drone Vision
          </h1>

          {/* Research Abstract */}
          <p className="text-base sm:text-lg text-slate-300 font-sans leading-relaxed max-w-3xl">
            An advanced computer vision platform combining high-resolution aerial drone imagery with fine-tuned
            <strong className="text-white font-semibold"> Ultralytics YOLOv11</strong> detection and deterministic OpenCV geometric profiling.
            Automatically quantifies railway defect severity, calculates a unified
            <strong className="text-gold font-semibold"> Track Risk Index (0–100)</strong>, and prioritizes maintenance crew dispatches in real time.
          </p>

          {/* Call to Actions & Telemetry Strip */}
          <div className="pt-4 flex flex-wrap items-center gap-4">
            <button
              onClick={onLaunchWorkbench}
              className="inline-flex items-center space-x-2.5 px-6 py-3.5 rounded-btn bg-gold text-navy font-semibold text-sm hover:bg-gold-hover transition-all shadow-glow hover:translate-y-[-1px]"
            >
              <span>Launch Inspection Workbench</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <div className="flex items-center space-x-3 text-xs font-telemetry text-slate-400 bg-navy-surface/80 px-4 py-3 rounded-btn border border-navy-light">
              <span className="text-emerald-400 font-semibold">5 Canonical Classes</span>
              <span>•</span>
              <span className="text-gold font-semibold">4,975 + 696 Dataset</span>
              <span>•</span>
              <span className="text-slate-300">FastAPI REST v1</span>
            </div>
          </div>
        </div>
      </div>

      {/* Track Risk Index Methodology Banner */}
      <div className="bg-surface-card rounded-card p-6 border border-border-muted shadow-card flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div className="space-y-1 max-w-2xl">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-telemetry uppercase tracking-wider font-bold text-navy">
              Standardized Severity Assessment Formula
            </span>
            <span className="text-[10px] px-1.5 py-0.5 rounded-btn bg-amber-100 text-amber-800 font-telemetry font-bold">
              Deterministic
            </span>
          </div>
          <p className="text-xs text-text-muted leading-relaxed">
            Every defect detection is mathematically scored using geometric bounding dimensions and cluster density.
            Track Risk Index is defined as: <code className="font-telemetry bg-surface px-1.5 py-0.5 rounded text-navy font-semibold">TRI = min(100.0, Σ Weight(severity_i))</code> where LOW = 10 pts, MEDIUM = 25 pts, and HIGH = 50 pts.
          </p>
        </div>
        <div className="flex items-center gap-2 font-telemetry text-xs shrink-0">
          <div className="px-3 py-2 rounded-btn bg-amber-50 border border-amber-200 text-amber-900 text-center">
            <div className="text-[10px] text-amber-700 font-bold">LOW RISK</div>
            <div className="font-bold">10 pts</div>
          </div>
          <div className="px-3 py-2 rounded-btn bg-orange-50 border border-orange-200 text-orange-900 text-center">
            <div className="text-[10px] text-orange-700 font-bold">MEDIUM RISK</div>
            <div className="font-bold">25 pts</div>
          </div>
          <div className="px-3 py-2 rounded-btn bg-red-50 border border-red-200 text-red-900 text-center">
            <div className="text-[10px] text-red-700 font-bold">HIGH RISK</div>
            <div className="font-bold">50 pts</div>
          </div>
        </div>
      </div>

      {/* Canonical Defect Taxonomy Grid */}
      <div className="space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2 border-b border-border-muted pb-4">
          <div>
            <h2 className="font-editorial text-2xl font-bold text-navy">
              Canonical Railway Defect Taxonomy
            </h2>
            <p className="text-xs text-text-muted mt-1">
              Harmonized across multi-source drone datasets into 5 standardized track inspection classes.
            </p>
          </div>
          <div className="text-xs font-telemetry text-slate-500">
            Rule-Based Severity Mapping
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {defectClasses.map((item) => {
            const IconComponent = item.icon;
            return (
              <div
                key={item.id}
                className="bg-surface-card rounded-card p-5 border border-border-muted hover:border-gold/50 shadow-card hover:shadow-editorial transition-all space-y-3 flex flex-col justify-between"
              >
                <div className="space-y-2.5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div className="p-1.5 rounded-btn bg-navy/5 text-navy">
                        <IconComponent className="w-4 h-4" />
                      </div>
                      <span className="font-telemetry text-xs font-bold text-navy">
                        Class {item.id}
                      </span>
                    </div>
                    <span className={`text-[10px] font-telemetry font-bold px-2 py-0.5 rounded-btn border ${item.badgeColor}`}>
                      {item.riskWeight}
                    </span>
                  </div>

                  <h3 className="font-editorial text-base font-bold text-navy">
                    {item.displayTitle}
                  </h3>

                  <p className="text-xs text-text-muted leading-relaxed">
                    {item.description}
                  </p>
                </div>

                <div className="pt-3 border-t border-border-muted text-[11px] font-telemetry text-slate-600 bg-surface/50 p-2.5 rounded-btn space-y-1">
                  <div className="text-[10px] uppercase font-bold text-slate-400">Severity Assessment Rule:</div>
                  <div className="font-medium text-slate-700">{item.severityRule}</div>
                </div>
              </div>
            );
          })}

          {/* Track Maintenance Clearance Card */}
          <div className="bg-gradient-to-br from-emerald-950/20 to-emerald-900/10 rounded-card p-5 border border-emerald-300 shadow-card flex flex-col justify-between space-y-3">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div className="p-1.5 rounded-btn bg-emerald-100 text-emerald-800">
                    <CheckCircle2 className="w-4 h-4" />
                  </div>
                  <span className="font-telemetry text-xs font-bold text-emerald-900">
                    Nominal Health
                  </span>
                </div>
                <span className="text-[10px] font-telemetry font-bold px-2 py-0.5 rounded-btn bg-emerald-100 text-emerald-900 border border-emerald-300">
                  0 pts
                </span>
              </div>
              <h3 className="font-editorial text-base font-bold text-emerald-950">
                Track Clearance Nominal
              </h3>
              <p className="text-xs text-emerald-900/80 leading-relaxed">
                Clear railway right-of-way with secure Pandrol clips, intact rail head surfaces, and zero sleeper skew. Regular drone surveillance schedule maintained.
              </p>
            </div>
            <div className="pt-3 border-t border-emerald-200 text-[11px] font-telemetry text-emerald-900 bg-emerald-50/70 p-2.5 rounded-btn">
              <div className="text-[10px] uppercase font-bold text-emerald-700">Dispatch Status:</div>
              <div className="font-medium">Clear for standard operational speed</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
