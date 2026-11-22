import React from "react";
import { Activity, ShieldCheck, AlertCircle, WifiOff } from "lucide-react";
import { HealthResponse } from "@/types/api";

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  health: HealthResponse | null;
  healthLoading: boolean;
  healthError: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  health,
  healthLoading,
  healthError,
}) => {
  const navItems = [
    { id: "overview", label: "Overview" },
    { id: "workbench", label: "Workbench" },
    { id: "metrics", label: "Model & Metrics" },
  ];

  // Derive compact status pill
  let statusText = "Checking...";
  let statusColor = "bg-slate-800 text-slate-300 border-slate-700";
  let StatusIcon = Activity;

  if (healthError) {
    statusText = "Offline";
    statusColor = "bg-red-950/80 text-red-300 border-red-800";
    StatusIcon = WifiOff;
  } else if (!healthLoading && health) {
    if (health.model_loaded) {
      statusText = "API Ready";
      statusColor = "bg-emerald-950/80 text-emerald-300 border-emerald-800";
      StatusIcon = ShieldCheck;
    } else {
      statusText = "No Weights";
      statusColor = "bg-amber-950/80 text-amber-300 border-amber-800";
      StatusIcon = AlertCircle;
    }
  }

  return (
    <header className="sticky top-0 z-50 bg-navy text-white border-b border-navy-surface shadow-editorial">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          {/* Brand Mark */}
          <div
            onClick={() => setActiveTab("overview")}
            className="flex items-center space-x-2.5 cursor-pointer select-none"
          >
            <div className="w-7 h-7 rounded-btn bg-gold flex items-center justify-center text-navy font-bold shadow-glow">
              <span className="font-telemetry text-xs tracking-wider">RV</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="font-editorial text-lg tracking-wide font-bold text-white">
                Rail<span className="text-gold">Vision</span>
              </span>
              <span className="text-[10px] font-telemetry px-1.5 py-0.5 rounded-btn bg-navy-surface text-slate-300 border border-navy-light font-medium">
                YOLOv11
              </span>
            </div>
          </div>

          {/* 3 Clean Navigation Tabs */}
          <nav className="flex space-x-1 sm:space-x-2">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`px-3 py-1.5 text-xs uppercase tracking-wider font-medium transition-all rounded-btn ${
                    isActive
                      ? "text-gold bg-navy-surface font-semibold shadow-sm border border-gold/30"
                      : "text-slate-300 hover:text-white hover:bg-navy-surface/40"
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Compact Telemetry Status Pill */}
          <div className="flex items-center">
            <div
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-btn text-[11px] font-telemetry font-medium border ${statusColor}`}
              title={
                health
                  ? `Active Model: ${health.model_name}`
                  : "Connecting to FastAPI backend..."
              }
            >
              <span className="relative flex h-1.5 w-1.5">
                {!healthError && (
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-75"></span>
                )}
                <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-current"></span>
              </span>
              <StatusIcon className="w-3 h-3" />
              <span className="tracking-wide">{statusText}</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
