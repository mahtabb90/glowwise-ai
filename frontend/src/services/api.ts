import type {
  HealthResponse,
  ModelStatus,
  SatisfactionPredictionRequest,
  SatisfactionPredictionResponse,
  TopTermsResponse,
  ClustersResponse,
  InsightsSummaryResponse
} from '../types/api';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = {
  /**
   * Fetches service health status from the backend /health endpoint.
   */
  async getHealth(): Promise<HealthResponse> {
    const response = await fetch(`${API_URL}/health`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  },

  /**
   * Fetches the ML model's loading and configuration status.
   */
  async getModelStatus(): Promise<ModelStatus> {
    const response = await fetch(`${API_URL}/api/model/status`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  },

  /**
   * Submits a customer review for satisfaction prediction classification.
   */
  async predictSatisfaction(payload: SatisfactionPredictionRequest): Promise<SatisfactionPredictionResponse> {
    const response = await fetch(`${API_URL}/api/predict/satisfaction`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }
    return response.json();
  },

  /**
   * Retrieves top positive and negative vocabulary terms driving satisfaction.
   */
  async getTopTerms(): Promise<TopTermsResponse> {
    const response = await fetch(`${API_URL}/api/insights/top-terms`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  },

  /**
   * Retrieves post-hoc profiled customer segments and clusters.
   */
  async getClusters(): Promise<ClustersResponse> {
    const response = await fetch(`${API_URL}/api/insights/clusters`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  },

  /**
   * Retrieves a compact summary dashboard of file and model availabilities.
   */
  async getInsightsSummary(): Promise<InsightsSummaryResponse> {
    const response = await fetch(`${API_URL}/api/insights/summary`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }
};
export type { HealthResponse };
