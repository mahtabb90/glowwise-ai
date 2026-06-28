export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  debug: boolean;
}

export interface ModelStatus {
  model_loaded: boolean;
  model_path: string;
  model_name: string;
  message: string;
}

export interface SatisfactionPredictionRequest {
  review_text: string;
  review_title?: string;
  product_name?: string;
  brand_name?: string;
}

export interface SatisfactionPredictionResponse {
  predicted_label: number;
  predicted_class: string;
  confidence: number;
  probability_low_or_medium: number;
  probability_high_satisfaction: number;
  model_name: string;
  input_preview: string;
}

export interface TermImportance {
  term: string;
  coefficient: number;
  direction: string;
  rank: number;
}

export interface TopTermsResponse {
  available: boolean;
  top_positive_terms: TermImportance[];
  top_negative_terms: TermImportance[];
  message: string;
}

export interface ClusterProfile {
  cluster_id: number;
  persona_name: string;
  size: number;
  percentage: number;
  high_satisfaction_rate: number;
  sentiment_positive: number;
  sentiment_neutral: number;
  sentiment_negative: number;
  avg_text_length: number;
  avg_word_count: number;
  top_brands: string;
  top_terms: string;
}

export interface ClustersResponse {
  available: boolean;
  selected_k: number;
  cluster_profiles: ClusterProfile[];
  message: string;
}

export interface InsightsSummaryResponse {
  top_terms_available: boolean;
  cluster_profiles_available: boolean;
  model_available: boolean;
  model_name: string;
  message: string;
}
export type { ModelStatus as ModelStatusResponse };
