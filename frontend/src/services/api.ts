/**
 * Typed API Client for Railway Defect Inspection Backend.
 * 
 * Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
 * Phase 5: React Frontend & Full-Stack Integration.
 */

import {
  HealthResponse,
  InspectionHistoryResponse,
  InspectionResponse,
  MetricsResponse,
} from "@/types/api";

const API_BASE_URL =
  import.meta.env.VITE_API_URL?.replace(/\/$/, "") || "http://localhost:8000/api/v1";

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorMessage = `HTTP error ${res.status}`;
    try {
      const errorJson = await res.json();
      if (errorJson.detail) {
        errorMessage = errorJson.detail;
      }
    } catch {
      // Fallback to text or status text
      const text = await res.text().catch(() => "");
      if (text) errorMessage = text;
    }
    throw new ApiError(errorMessage, res.status);
  }
  return res.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  return handleResponse<HealthResponse>(response);
}

export async function inspectImage(
  file: File,
  confThreshold: number = 0.25,
  includeCrops: boolean = true,
  drawHud: boolean = true,
): Promise<InspectionResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("conf_threshold", confThreshold.toString());
  formData.append("include_crops", includeCrops ? "true" : "false");
  formData.append("draw_hud", drawHud ? "true" : "false");

  const response = await fetch(`${API_BASE_URL}/inspect`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<InspectionResponse>(response);
}

export async function getMetrics(): Promise<MetricsResponse> {
  const response = await fetch(`${API_BASE_URL}/metrics`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  return handleResponse<MetricsResponse>(response);
}

export async function getHistory(limit: number = 20): Promise<InspectionHistoryResponse> {
  const response = await fetch(`${API_BASE_URL}/history?limit=${limit}`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  return handleResponse<InspectionHistoryResponse>(response);
}

export { ApiError, API_BASE_URL };
