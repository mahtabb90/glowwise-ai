import React from 'react';
import type { ClustersResponse, ClusterProfile } from '../types/api';

interface ClusterCardsProps {
  clusters: ClustersResponse | null;
  loading: boolean;
  error: string | null;
}

const FALLBACK_PROFILES: ClusterProfile[] = [
  {
    cluster_id: 0,
    persona_name: "General Beauty Enthusiasts",
    size: 27608,
    percentage: 0.276,
    high_satisfaction_rate: 0.844,
    sentiment_positive: 0.844,
    sentiment_neutral: 0.063,
    sentiment_negative: 0.093,
    avg_text_length: 262.8,
    avg_word_count: 49.6,
    top_brands: "CLINIQUE, Tatcha, fresh",
    top_terms: "love, face, makeup, product, great, skin, like, use, love love, really"
  },
  {
    cluster_id: 1,
    persona_name: "Daily Skincare Users",
    size: 40467,
    percentage: 0.405,
    high_satisfaction_rate: 0.769,
    sentiment_positive: 0.769,
    sentiment_neutral: 0.095,
    sentiment_negative: 0.136,
    avg_text_length: 362.4,
    avg_word_count: 68.2,
    top_brands: "CLINIQUE, The Ordinary, Drunk Elephant",
    top_terms: "skin, product, using, use, ve, really, like, used, great, face"
  },
  {
    cluster_id: 2,
    persona_name: "Moisture & Texture Fans",
    size: 17495,
    percentage: 0.175,
    high_satisfaction_rate: 0.879,
    sentiment_positive: 0.879,
    sentiment_neutral: 0.062,
    sentiment_negative: 0.060,
    avg_text_length: 352.7,
    avg_word_count: 66.2,
    top_brands: "Tatcha, CLINIQUE, belif",
    top_terms: "skin, moisturizer, feels, oily, dry, cream, like, dry skin, face, great"
  },
  {
    cluster_id: 3,
    persona_name: "Acne & Blemish Care",
    size: 9926,
    percentage: 0.099,
    high_satisfaction_rate: 0.859,
    sentiment_positive: 0.859,
    sentiment_neutral: 0.057,
    sentiment_negative: 0.084,
    avg_text_length: 403.2,
    avg_word_count: 75.3,
    top_brands: "The Ordinary, Murad, Drunk Elephant",
    top_terms: "acne, serum, skin, prone, acne prone, product, prone skin, using, use, love"
  },
  {
    cluster_id: 4,
    persona_name: "Lip Care Seekers",
    size: 4504,
    percentage: 0.045,
    high_satisfaction_rate: 0.839,
    sentiment_positive: 0.839,
    sentiment_neutral: 0.061,
    sentiment_negative: 0.100,
    avg_text_length: 295.7,
    avg_word_count: 56.7,
    top_brands: "LANEIGE, Rosebud Perfume Co., fresh",
    top_terms: "lips, lip, balm, lip balm, love, product, chapped, mask, dry, lip mask"
  }
];

const getClusterIcon = (id: number): string => {
  const icons = ["✨", "🧴", "🌿", "🔬", "💋"];
  return icons[id] || "🌸";
};

export const ClusterCards: React.FC<ClusterCardsProps> = ({ clusters, loading, error }) => {
  const isApiAvailable = !error && clusters?.available;
  const profiles = isApiAvailable && clusters ? clusters.cluster_profiles : FALLBACK_PROFILES;
  const isUsingFallback = !isApiAvailable && !loading;

  return (
    <div className="mb-8">
      <h3 className="text-xl font-bold text-plum mb-2 flex items-center gap-2">
        <span>🌸</span> Beauty Customer Personas
      </h3>
      <p className="text-xs text-muted-plum mb-6">
        Consumer cohorts discovered via unsupervised K-Means clustering of reviews. Mapped using word vectors to capture unique skin concern personas and satisfaction rates.
      </p>

      {loading && (
        <div className="glass-card p-6 rounded-2xl border border-rose-gold/20 shadow-sm text-center py-12 mb-8">
          <span className="w-6 h-6 border-2 border-plum border-t-transparent rounded-full animate-spin inline-block mb-2"></span>
          <p className="text-sm text-muted-plum">Loading customer segments from report...</p>
        </div>
      )}

      {isUsingFallback && (
        <div className="p-3 bg-cream border border-gold/40 text-gold-950 text-xs rounded-xl mb-6">
          <strong>ℹ️ Offline Fallback Data:</strong> The backend clustering report is currently unavailable. Displaying cached dashboard profiles derived from the 100k Sephora review dataset.
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {profiles.map((p) => (
          <div 
            key={`cluster-${p.cluster_id}`}
            className="glass-card p-6 rounded-2xl border border-rose-gold/20 shadow-sm flex flex-col justify-between hover:border-gold hover:shadow-md transition-all duration-300"
          >
            <div>
              {/* Persona title & size */}
              <div className="flex justify-between items-start mb-3">
                <span className="text-[10px] uppercase font-bold text-muted-plum tracking-wide bg-cream px-2 py-0.5 rounded border border-rose-gold/20 font-mono">
                  Cluster {p.cluster_id}
                </span>
                <span className="text-xs font-semibold text-plum">
                  {p.size.toLocaleString()} reviews ({Math.round(p.percentage * 1000) / 10}%)
                </span>
              </div>
              
              <h4 className="text-base font-bold text-plum mb-3 flex items-center gap-2">
                <span>{getClusterIcon(p.cluster_id)}</span> {p.persona_name}
              </h4>

              {/* Core metrics */}
              <div className="grid grid-cols-2 gap-2 bg-cream/40 p-3 rounded-xl border border-rose-gold/10 mb-4 text-xs">
                <div>
                  <span className="block text-[10px] text-muted-plum font-semibold uppercase">High Sat. Rate</span>
                  <span className="font-bold text-gold-dark text-sm">{Math.round(p.high_satisfaction_rate * 1000) / 10}%</span>
                </div>
                <div>
                  <span className="block text-[10px] text-muted-plum font-semibold uppercase">Avg. Word Count</span>
                  <span className="font-bold text-plum text-sm">{Math.round(p.avg_word_count)} words</span>
                </div>
              </div>

              {/* Top Brands */}
              <div className="mb-3">
                <span className="block text-[10px] uppercase font-bold text-muted-plum tracking-wider mb-1">Top Brands</span>
                <span className="text-xs text-plum font-semibold">{p.top_brands}</span>
              </div>
            </div>

            {/* Top Terms chips */}
            <div className="pt-3 border-t border-rose-gold/10">
              <span className="block text-[10px] uppercase font-bold text-muted-plum tracking-wider mb-1.5">Vocabulary Terms</span>
              <div className="flex flex-wrap gap-1">
                {p.top_terms.split(',').slice(0, 6).map((term, idx) => (
                  <span 
                    key={`term-${idx}`}
                    className="text-[10px] bg-cream border border-rose-gold/30 text-plum font-semibold px-2 py-0.5 rounded"
                  >
                    {term.trim()}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
export { FALLBACK_PROFILES };
