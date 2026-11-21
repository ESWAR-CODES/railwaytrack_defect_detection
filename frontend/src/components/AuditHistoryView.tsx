import React, { useEffect, useState } from "react";
import { History, RefreshCw, FileText, AlertCircle, ShieldCheck } from "lucide-react";
import { getHistory } from "@/services/api";
import { InspectionHistoryItem } from "@/types/api";

export const AuditHistoryView: React.FC = () => {
  const [historyItems, setHistoryItems] = useState<InspectionHistoryItem[]>([]);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getHistory(50);
      setHistoryItems(data.items);
      setTotalCount(data.total_inspections);
    } catch (err: any) {
      setError(err.message || "Failed to load inspection history from server.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const getRiskBadge = (level: string) => {
    switch (level) {
      case "HIGH":
        return "bg-red-100 text-red-800 border-red-300 font-bold";
      case "MEDIUM":
        return "bg-orange-100 text-orange-800 border-orange-300 font-bold";
      case "LOW":
        return "bg-amber-100 text-amber-800 border-amber-300 font-bold";
      default:
        return "bg-emerald-100 text-emerald-800 border-emerald-300 font-bold";
    }
  };

  return (
    <section id="history" className="space-y-8">
      {/* Section Header */}
      <div className="border-b border-border-muted pb-4 flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h2 className="font-editorial text-2xl sm:text-3xl font-bold text-navy">
            Inspection Audit & Telemetry History
          </h2>
          <p className="text-xs text-text-muted mt-1">
            Sequential log of recent aerial drone track inspections, evaluated defect counts, and Track Risk Index scores.
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <span className="text-xs font-telemetry text-slate-500">
            Total Logged: <strong className="text-navy">{totalCount}</strong>
          </span>
          <button
            onClick={fetchHistory}
            className="text-xs font-telemetry px-3 py-1.5 rounded-btn bg-surface hover:bg-slate-200 text-navy border border-border-muted transition-colors flex items-center space-x-1"
          >
            <RefreshCw className={`w-3.5 h-3.5 text-gold ${loading ? "animate-spin" : ""}`} />
            <span>Refresh Audit Log</span>
          </button>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="bg-red-50 border border-red-300 text-red-800 text-xs p-4 rounded-btn flex items-start space-x-2">
          <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
          <div>
            <p className="font-bold">Error Loading Audit Log</p>
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* History Table Container */}
      <div className="bg-surface-card rounded-card border border-border-muted shadow-card overflow-hidden">
        {historyItems.length === 0 && !loading ? (
          <div className="p-12 text-center text-text-muted space-y-2">
            <History className="w-10 h-10 mx-auto text-slate-400" />
            <p className="text-xs font-telemetry font-bold text-navy">No Inspections Recorded Yet</p>
            <p className="text-xs text-slate-500 max-w-sm mx-auto">
              Run an image analysis in the Inspection Workbench to generate logged defect events here.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-surface text-text-muted uppercase font-telemetry text-[10px] tracking-wider border-b border-border-muted">
                  <th className="py-3 px-4">Inspection Event</th>
                  <th className="py-3 px-4">Timestamp (UTC)</th>
                  <th className="py-3 px-4">Resolution</th>
                  <th className="py-3 px-4">Total Defects</th>
                  <th className="py-3 px-4">Track Risk Index (TRI)</th>
                  <th className="py-3 px-4">Overall Risk Level</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border-muted">
                {historyItems.map((item) => (
                  <tr key={item.inspection_id} className="hover:bg-surface/50 transition-colors">
                    <td className="py-3.5 px-4">
                      <div className="flex items-center space-x-2">
                        <FileText className="w-4 h-4 text-slate-400" />
                        <div>
                          <div className="font-telemetry font-bold text-navy">
                            {item.filename}
                          </div>
                          <div className="text-[10px] text-slate-400 font-telemetry">
                            ID: {item.inspection_id.slice(0, 12)}...
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="py-3.5 px-4 font-telemetry text-slate-600">
                      {new Date(item.created_at).toLocaleString()}
                    </td>
                    <td className="py-3.5 px-4 font-telemetry text-slate-500">
                      {item.image_width} × {item.image_height}px
                    </td>
                    <td className="py-3.5 px-4 font-telemetry font-bold text-navy">
                      {item.total_defects}
                    </td>
                    <td className="py-3.5 px-4 font-telemetry">
                      <span className="font-bold text-navy">{item.track_risk_index.toFixed(1)}</span>
                      <span className="text-slate-400 text-[10px]"> / 100</span>
                    </td>
                    <td className="py-3.5 px-4">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-telemetry uppercase border ${getRiskBadge(item.overall_risk_level)}`}>
                        {item.overall_risk_level}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
};
