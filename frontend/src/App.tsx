import React, { useState, useEffect } from "react";
import { Navbar } from "@/components/Navbar";
import { Footer } from "@/components/Footer";
import { HeroOverview } from "@/components/HeroOverview";
import { InspectionWorkbench } from "@/components/InspectionWorkbench";
import { ModelDatasetView } from "@/components/ModelDatasetView";
import { getHealth } from "@/services/api";
import { HealthResponse } from "@/types/api";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>("overview");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);
  const [healthError, setHealthError] = useState<boolean>(false);

  // Poll backend health status on mount and every 30 seconds
  const fetchHealthStatus = async () => {
    try {
      const data = await getHealth();
      setHealth(data);
      setHealthError(false);
    } catch {
      setHealthError(true);
    } finally {
      setHealthLoading(false);
    }
  };

  useEffect(() => {
    fetchHealthStatus();
    const interval = setInterval(fetchHealthStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleLaunchWorkbench = () => {
    setActiveTab("workbench");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="min-h-screen flex flex-col bg-ivory text-text-primary">
      {/* Editorial Sticky Navy Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={(tab) => {
          setActiveTab(tab);
          window.scrollTo({ top: 0, behavior: "smooth" });
        }}
        health={health}
        healthLoading={healthLoading}
        healthError={healthError}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-10">
        {activeTab === "overview" && (
          <HeroOverview onLaunchWorkbench={handleLaunchWorkbench} />
        )}

        {activeTab === "workbench" && <InspectionWorkbench />}

        {activeTab === "metrics" && <ModelDatasetView health={health} />}
      </main>

      {/* Minimal Navy Footer */}
      <Footer
        setActiveTab={(tab) => {
          setActiveTab(tab);
          window.scrollTo({ top: 0, behavior: "smooth" });
        }}
      />
    </div>
  );
};

export default App;
