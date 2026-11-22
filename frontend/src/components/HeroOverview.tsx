import React from "react";
import { ArrowRight, AlertTriangle, Layers, Crosshair, Cpu, ShieldAlert } from "lucide-react";

interface HeroOverviewProps {
  onLaunchWorkbench: () => void;
}

export const HeroOverview: React.FC<HeroOverviewProps> = ({ onLaunchWorkbench }) => {
  const defectClasses = [
    {
      id: 0,
      name: "rail_crack",
      displayTitle: "Rail Crack",
      severityRule: "Evaluated by diagonal dimension ratio (Low <0.03, Medium ≤0.08, High >0.08).",
      icon: AlertTriangle,
      badge: "Low / Med / High",
      accent: "text-red-700 bg-red-50 border-red-200",
    },
    {
      id: 1,
      name: "missing_fastener",
      displayTitle: "Missing Fastener",
      severityRule: "Escalated by occurrence count (1 = Low, 2 = Medium, ≥3 = High).",
      icon: Crosshair,
      badge: "Count Escalated",
      accent: "text-orange-700 bg-orange-50 border-orange-200",
    },
    {
      id: 2,
      name: "rail_corrosion",
      displayTitle: "Rail Corrosion",
      severityRule: "Evaluated by surface area coverage (Low <0.05, Medium ≤0.15, High >0.15).",
      icon: Layers,
      badge: "Area Ratio",
      accent: "text-amber-700 bg-amber-50 border-amber-200",
    },
    {
      id: 3,
      name: "obstacle",
      displayTitle: "Track Obstacle",
      severityRule: "Mandatory immediate High severity for foreign object intrusions.",
      icon: ShieldAlert,
      badge: "Mandatory High",
      accent: "text-red-700 bg-red-50 border-red-200",
    },
    {
      id: 4,
      name: "sleeper_misalignment",
      displayTitle: "Sleeper Misalignment",
      severityRule: "Evaluated by cluster count (1 = Low, 2 = Medium, ≥3 = High).",
      icon: Cpu,
      badge: "Count Escalated",
      accent: "text-orange-700 bg-orange-50 border-orange-200",
    },
  ];

  return (
    <div className="-mt-8 md:-mt-10 -mx-4 sm:-mx-6 lg:-mx-8">
      {/* Full-Bleed Edge-to-Edge Cinematic Navy Hero Band */}
      <section className="bg-navy text-white pt-20 pb-24 px-4 sm:px-6 lg:px-8 relative overflow-hidden border-b border-navy-surface">
        {/* Subtle geometric track lines background pattern */}
        <div
          className="absolute inset-0 opacity-10 pointer-events-none"
          style={{
            backgroundImage:
              "linear-gradient(90deg, #F5A623 1px, transparent 1px), linear-gradient(0deg, #1C2541 1px, transparent 1px)",
            backgroundSize: "48px 48px",
          }}
        ></div>

        <div className="relative z-10 max-w-5xl mx-auto space-y-6 text-center md:text-left">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-btn bg-navy-surface border border-navy-light text-xs font-telemetry text-gold">
            <span className="w-1.5 h-1.5 rounded-full bg-gold"></span>
            <span>AERIAL DRONE COMPUTER VISION & TELEMETRY</span>
          </div>

          <h1 className="font-editorial text-3xl sm:text-5xl lg:text-6xl font-bold tracking-tight leading-tight text-white max-w-4xl">
            Autonomous Multi-Defect Railway Vision & Severity Assessment
          </h1>

          <p className="text-base sm:text-xl text-slate-300 font-sans leading-relaxed max-w-3xl">
            High-resolution drone vision combined with fine-tuned YOLOv11 and deterministic geometric profiling.
            Instantly identifies track surface anomalies, missing fastenings, and obstacles while calculating a standardized 0–100 Track Risk Index for maintenance dispatch.
          </p>

          <div className="pt-4 flex flex-wrap items-center justify-center md:justify-start gap-4">
            <button
              onClick={onLaunchWorkbench}
              className="inline-flex items-center space-x-2.5 px-6 py-3.5 rounded-btn bg-gold text-navy font-semibold text-sm hover:bg-gold-hover transition-all shadow-glow hover:translate-y-[-1px]"
            >
              <span>Open Inspection Workbench</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <div className="flex items-center space-x-3 text-xs font-telemetry text-slate-400 bg-navy-surface/80 px-4 py-3 rounded-btn border border-navy-light">
              <span className="text-emerald-400 font-semibold">5 Defect Classes</span>
              <span>•</span>
              <span className="text-gold font-semibold">Real-Time TRI (0–100)</span>
              <span>•</span>
              <span className="text-slate-300">FastAPI + OpenCV</span>
            </div>
          </div>
        </div>
      </section>

      {/* Airy, Unboxed Editorial Section on Warm Ivory */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 space-y-12">
        {/* Severity & Track Risk Index Methodology Overview */}
        <div className="border-b border-border-muted pb-8 space-y-2">
          <h2 className="font-editorial text-2xl sm:text-3xl font-bold text-navy">
            Target Defect Taxonomy & Severity Rules
          </h2>
          <p className="text-sm text-text-muted max-w-3xl">
            Each detection is classified into one of 5 canonical defect categories and rated as <strong className="text-amber-700">Low (10 pts)</strong>, <strong className="text-orange-700">Medium (25 pts)</strong>, or <strong className="text-red-700">High (50 pts)</strong>, culminating in a composite Track Risk Index clamped to 0–100.
          </p>
        </div>

        {/* 5 Canonical Defect Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {defectClasses.map((item) => {
            const Icon = item.icon;
            return (
              <div
                key={item.id}
                className="bg-white rounded-card p-6 border border-border-muted hover:border-gold/50 shadow-card hover:shadow-editorial transition-all space-y-4"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2.5">
                    <div className="w-8 h-8 rounded-btn bg-navy/5 flex items-center justify-center text-navy">
                      <Icon className="w-4 h-4 text-navy" />
                    </div>
                    <span className="font-telemetry text-xs font-bold text-slate-500">
                      Class {item.id}
                    </span>
                  </div>
                  <span className={`text-[11px] font-telemetry font-semibold px-2 py-0.5 rounded border ${item.accent}`}>
                    {item.badge}
                  </span>
                </div>

                <div className="space-y-1.5">
                  <h3 className="font-editorial text-lg font-bold text-navy">
                    {item.displayTitle}
                  </h3>
                  <code className="text-xs font-telemetry text-gold block">
                    {item.name}
                  </code>
                </div>

                <p className="text-xs text-text-muted leading-relaxed">
                  {item.severityRule}
                </p>
              </div>
            );
          })}

          {/* Quickstart Callout Card */}
          <div className="bg-navy text-white rounded-card p-6 border border-navy-surface shadow-card flex flex-col justify-between space-y-4">
            <div className="space-y-2">
              <span className="text-[10px] font-telemetry font-bold text-gold uppercase tracking-wider">
                Live Demonstration
              </span>
              <h3 className="font-editorial text-lg font-bold">
                Pre-Loaded Track Samples
              </h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                Experience real-time defect localization and severity profiling using pre-loaded test split imagery or custom aerial drone photographs.
              </p>
            </div>
            <button
              onClick={onLaunchWorkbench}
              className="w-full py-2.5 rounded-btn bg-gold hover:bg-gold-hover text-navy font-semibold text-xs transition-colors flex items-center justify-center space-x-1.5"
            >
              <span>Launch Demo Workbench</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};
