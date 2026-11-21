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
    { id: "workbench", label: "Inspection Workbench" },
    { id: "telemetry", label: "Model & Dataset" },
    { id: "history", label: "Audit History" },
  ];

  // Derive status badge
  let statusText = "CHECKING...";
  let statusColor = "bg-slate-500 text-slate-200 border-slate-600";
  let StatusIcon = Activity;

  if (healthError) {
    statusText = "BACKEND OFFLINE";
    statusColor = "bg-red-950 text-red-300 border-red-800";
    StatusIcon = WifiOff;
  } else if (!healthLoading && health) {
    if (health.model_loaded) {
      statusText = "SYSTEM ONLINE";
      statusColor = "bg-emerald-950 text-emerald-300 border-emerald-700";
      StatusIcon = ShieldCheck;
    } else {
      statusText = "MODEL PENDING";
      statusColor = "bg-amber-950 text-amber-300 border-amber-700";
      StatusIcon = AlertCircle;
    }
  }

  return (
    <header className="sticky top-0 z-50 bg-navy text-text-inverse border-b border-navy-surface shadow-editorial">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Identity */}
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-btn bg-gold flex items-center justify-center text-navy font-bold shadow-glow">
              <span className="font-telemetry text-sm tracking-wider">MBU</span>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-editorial text-lg tracking-wide font-bold text-white">
                  RailDefect<span className="text-gold">Vision</span>
                </span>
                <span className="text-[10px] font-telemetry px-1.5 py-0.5 rounded-btn bg-navy-surface text-gold border border-navy-light font-semibold">
                  v1.0 • YOLO-11n
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-sans hidden sm:block">
                Mohan Babu University • Dept. of CSE (AI & ML) • Team B6-B3
              </p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden md:flex space-x-1 lg:space-x-2">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`px-3 py-2 text-xs uppercase tracking-wider font-medium transition-all rounded-btn ${
                    isActive
                      ? "text-gold bg-navy-surface border-b-2 border-gold shadow-sm font-semibold"
                      : "text-slate-300 hover:text-white hover:bg-navy-surface/50"
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* System Telemetry Status Pill */}
          <div className="flex items-center space-x-2">
            <div
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-btn text-[11px] font-telemetry font-medium border ${statusColor}`}
              title={
                health
                  ? `Active Model: ${health.model_name} | Weights: ${health.weights_path || "Default"}`
                  : "Connecting to FastAPI backend..."
              }
            >
              <span className="relative flex h-2 w-2">
                {!healthError && (
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-75"></span>
                )}
                <span className="relative inline-flex rounded-full h-2 w-2 bg-current"></span>
              </span>
              <StatusIcon className="w-3.5 h-3.5" />
              <span className="tracking-wider">{statusText}</span>
            </div>
          </div>
        </div>

        {/* Mobile Navigation bar */}
        <div className="flex md:hidden border-t border-navy-surface py-2 overflow-x-auto space-x-1">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`px-2.5 py-1 text-[11px] uppercase tracking-wider whitespace-nowrap rounded-btn transition-colors ${
                  isActive
                    ? "text-gold bg-navy-surface font-semibold"
                    : "text-slate-300 hover:text-white"
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
