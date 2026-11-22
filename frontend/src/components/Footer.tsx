import React from "react";

interface FooterProps {
  setActiveTab: (tab: string) => void;
}

export const Footer: React.FC<FooterProps> = ({ setActiveTab }) => {
  return (
    <footer className="bg-navy text-slate-400 border-t border-navy-surface py-8 mt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs">
        {/* Brand & Stack */}
        <div className="space-y-1 text-center sm:text-left">
          <div className="font-editorial text-sm font-bold text-white tracking-wide">
            Rail<span className="text-gold">Vision</span>
            <span className="font-sans font-normal text-slate-400 ml-2">
              — Aerial Multi-Defect Railway Monitoring System
            </span>
          </div>
          <p className="text-[11px] font-telemetry text-slate-500">
            Core Architecture: Ultralytics YOLOv11 • OpenCV Annotations • FastAPI Backend • React 18 TypeScript
          </p>
        </div>

        {/* Quick Links */}
        <div className="flex items-center space-x-6 text-xs font-telemetry">
          <button
            onClick={() => {
              setActiveTab("overview");
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
            className="hover:text-gold transition-colors"
          >
            Overview
          </button>
          <button
            onClick={() => {
              setActiveTab("workbench");
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
            className="hover:text-gold transition-colors"
          >
            Workbench
          </button>
          <button
            onClick={() => {
              setActiveTab("metrics");
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
            className="hover:text-gold transition-colors"
          >
            Model & Metrics
          </button>
        </div>
      </div>
    </footer>
  );
};
