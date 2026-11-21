import React from "react";
import { Terminal, Shield, Award, Cpu } from "lucide-react";

export const Footer: React.FC = () => {
  return (
    <footer className="bg-navy-dark text-slate-400 border-t border-navy-surface pt-12 pb-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 pb-10 border-b border-navy-surface">
          {/* Col 1: Project Identity */}
          <div className="md:col-span-2 space-y-3">
            <div className="flex items-center space-x-2">
              <span className="font-editorial text-lg text-white font-bold">
                RailDefect<span className="text-gold">Vision</span>
              </span>
              <span className="text-[10px] font-telemetry px-2 py-0.5 rounded-btn bg-navy-surface text-gold border border-navy-light">
                MBU CSE (AI & ML)
              </span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed max-w-lg">
              AI-Based Multi-Defect Railway Monitoring and Severity Assessment Using YOLO-Based Drone Vision.
              Engineered for high-altitude aerial surveillance, localized geometric defect triage, and automated track safety dispatch.
            </p>
            <div className="flex items-center space-x-4 pt-1 text-xs text-slate-400 font-telemetry">
              <span className="flex items-center space-x-1.5">
                <Award className="w-3.5 h-3.5 text-gold" />
                <span>Team B6-B3</span>
              </span>
              <span className="flex items-center space-x-1.5">
                <Shield className="w-3.5 h-3.5 text-emerald-400" />
                <span>Supervisor: Ms. K. Sandhya</span>
              </span>
            </div>
          </div>

          {/* Col 2: Canonical Defect Classes */}
          <div className="space-y-2">
            <h4 className="text-xs font-telemetry uppercase tracking-wider text-white font-semibold flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5 text-gold" />
              <span>Target Taxonomy</span>
            </h4>
            <ul className="text-xs space-y-1 text-slate-300 font-telemetry">
              <li className="flex items-center space-x-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-telemetry-high"></span>
                <span>0: rail_crack</span>
              </li>
              <li className="flex items-center space-x-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-telemetry-medium"></span>
                <span>1: missing_fastener</span>
              </li>
              <li className="flex items-center space-x-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-telemetry-low"></span>
                <span>2: rail_corrosion</span>
              </li>
              <li className="flex items-center space-x-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-telemetry-high"></span>
                <span>3: obstacle</span>
              </li>
              <li className="flex items-center space-x-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-telemetry-medium"></span>
                <span>4: sleeper_misalignment</span>
              </li>
            </ul>
          </div>

          {/* Col 3: Technology Architecture */}
          <div className="space-y-2">
            <h4 className="text-xs font-telemetry uppercase tracking-wider text-white font-semibold flex items-center space-x-1.5">
              <Terminal className="w-3.5 h-3.5 text-gold" />
              <span>Core Stack</span>
            </h4>
            <div className="text-xs space-y-1 text-slate-300">
              <p>• Ultralytics YOLOv11n Detection</p>
              <p>• OpenCV Vision Annotator & HUD</p>
              <p>• Deterministic Track Risk Index (TRI)</p>
              <p>• FastAPI Asynchronous REST Engine</p>
              <p>• React 18 + Vite + TypeScript Client</p>
            </div>
          </div>
        </div>

        {/* Copyright & Disclaimer */}
        <div className="pt-6 flex flex-col sm:flex-row items-center justify-between text-[11px] text-slate-300 space-y-2 sm:space-y-0">
          <p>
            © {new Date().getFullYear()} Mohan Babu University, Tirupati. Dept. of Computer Science & Engineering (AI & ML).
          </p>
          <div className="flex items-center space-x-4 font-telemetry text-slate-300">
            <span>Anveshana Research Series</span>
            <span>•</span>
            <span>B.Tech Capstone Project</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
