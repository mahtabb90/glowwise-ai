import React from 'react';
import type { TopTermsResponse } from '../types/api';

interface TermsSectionProps {
  terms: TopTermsResponse | null;
  loading: boolean;
  error: string | null;
}

export const TermsSection: React.FC<TermsSectionProps> = ({ terms, loading, error }) => {
  if (loading) {
    return (
      <div className="glass-card p-6 rounded-2xl border border-rose-gold/20 shadow-sm text-center py-12">
        <span className="w-6 h-6 border-2 border-plum border-t-transparent rounded-full animate-spin inline-block mb-2"></span>
        <p className="text-sm text-muted-plum">Loading term significance coefficients from report...</p>
      </div>
    );
  }

  if (error || !terms?.available) {
    return (
      <div className="glass-card p-6 rounded-2xl border border-rose-gold/20 shadow-sm text-center py-8 bg-cream/40">
        <span className="text-2xl mb-1 inline-block">📋</span>
        <h4 className="text-sm font-bold text-plum">Explainability Report Unavailable</h4>
        <p className="text-xs text-muted-plum max-w-sm mx-auto mt-1">
          {error || terms?.message || "Explainability results JSON file not found. Make sure you run model_explainability.py first to compile."}
        </p>
      </div>
    );
  }

  const pos = terms.top_positive_terms || [];
  const neg = terms.top_negative_terms || [];

  return (
    <div className="glass-card p-6 rounded-2xl border border-rose-gold/20 shadow-sm mb-8">
      <h3 className="text-xl font-bold text-plum mb-2 flex items-center gap-2">
        <span>🌸</span> What drives love vs disappointment
      </h3>
      <p className="text-xs text-muted-plum mb-1.5">
        Review words learned by the model, showing which terms are most strongly connected to happy customers versus disappointed customers. Higher score means stronger influence in the prediction.
      </p>
      <div className="text-[10px] text-gold-dark font-semibold italic mb-6">
        * Scores show model influence, not percentages.
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Positive terms */}
        <div className="space-y-3">
          <h4 className="text-xs uppercase tracking-wider font-bold text-plum flex items-center gap-1-5">
            <span className="text-green-600">👍</span> Top Love Drivers
          </h4>
          <div className="flex flex-wrap gap-2">
            {pos.length === 0 ? (
              <span className="text-xs text-muted-plum">No terms available.</span>
            ) : (
              pos.map((t, idx) => (
                <div 
                  key={`pos-${idx}`}
                  className="bg-plum text-cream text-xs px-2-5 py-1 rounded-lg border border-plum/30 flex items-center gap-1-5 shadow-sm hover:scale-105 transition-transform"
                >
                  <span className="font-semibold">{t.term}</span>
                  <span className="text-[10px] opacity-80 font-mono">+{t.coefficient.toFixed(1)}</span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Negative terms */}
        <div className="space-y-3">
          <h4 className="text-xs uppercase tracking-wider font-bold text-plum flex items-center gap-1-5">
            <span className="text-red-500">👎</span> Top Disappointment Drivers
          </h4>
          <div className="flex flex-wrap gap-2">
            {neg.length === 0 ? (
              <span className="text-xs text-muted-plum">No terms available.</span>
            ) : (
              neg.map((t, idx) => (
                <div 
                  key={`neg-${idx}`}
                  className="bg-cream text-plum text-xs px-2-5 py-1 rounded-lg border border-gold/40 flex items-center gap-1-5 shadow-sm hover:scale-105 transition-transform"
                >
                  <span className="font-semibold">{t.term}</span>
                  <span className="text-[10px] text-gold font-mono">{t.coefficient.toFixed(1)}</span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
