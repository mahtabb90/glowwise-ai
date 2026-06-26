const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  debug: boolean;
}

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
  }
};
