import React from 'react';
import type { SatisfactionPredictionResponse } from '../types/api';

interface PredictionResultProps {
  result: SatisfactionPredictionResponse | null;
  error: string | null;
}

export const PredictionResult: React.FC<PredictionResultProps> = ({ result, error }) => {
  if (error) {
    return (
      <div className="glass-card p-6 rounded-2xl border border-red-200 bg-red-50 text-red-700 h-full flex flex-col justify-center">
        <h4 className="font-bold text-sm mb-2 flex items-center gap-2">
          <span>⚠️</span> Inference Failure
        </h4>
        <p className="text-xs">{error}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="glass-card p-8 rounded-2xl border border-dashed border-rose-gold/40 text-center text-muted-plum h-full flex flex-col justify-center items-center py-16">
        <span className="text-4xl mb-3 animate-pulse">🔮</span>
        <h4 className="font-bold text-base text-plum mb-2">Skin Intelligence Center</h4>
        <p className="text-xs max-w-xs mx-auto text-muted-plum leading-relaxed">
          Submit a customer skincare review on the left to activate real-time natural language processing and satisfaction analysis.
        </p>
      </div>
    );
  }

  const isSatisfied = result.predicted_label === 1;
  const confidenceVal = result.confidence;
  
  // Confidence interpretation
  let confidenceLabel = "Low confidence";
  if (confidenceVal >= 0.85) {
    confidenceLabel = "Very confident";
  } else if (confidenceVal >= 0.65) {
    confidenceLabel = "Moderately confident";
  }

  return (
    <div className="glass-card p-6 rounded-2xl border border-rose-gold/20 shadow-sm h-full flex flex-col justify-between">
      <div className="space-y-6">
        {/* Product-friendly header icon and title */}
        <h3 className="text-xl font-bold text-plum flex items-center gap-2 pb-3 border-b border-rose-gold/10 mb-4">
          <span>🎯</span> Your GlowWise Result
        </h3>

        {/* Visually elegant main prediction panel */}
        <div className={`p-6 rounded-2xl border shadow-sm transition-all duration-300 mb-8 ${
          isSatisfied 
            ? 'bg-plum text-cream border-plum/30' 
            : 'bg-red-50 text-red-800 border-red-200'
        }`}>
          <div className="flex justify-between items-start gap-4 mb-3">
            <div>
              <span className="text-[10px] uppercase font-bold tracking-wider opacity-85 block mb-1">
                Predicted Sentiment
              </span>
              <div className="text-lg font-black flex items-center gap-1-5">
                {isSatisfied ? (
                  <>👍 High Satisfaction</>
                ) : (
                  <>👎 Low/Medium Satisfaction</>
                )}
              </div>
            </div>
            
            {/* Confidence Badge */}
            <span className={`px-2.5 py-0.5 rounded-full text-[9px] uppercase font-extrabold tracking-widest border ${
              isSatisfied 
                ? 'bg-cream text-plum border-rose-gold'
                : 'bg-red-100 text-red-800 border-red-300'
            }`}>
              {confidenceLabel}
            </span>
          </div>
          
          <p className="text-xs opacity-90 mt-2 font-normal leading-relaxed">
            {isSatisfied 
              ? `Our beauty NLP pipeline predicts this customer review expresses high satisfaction with a model confidence score of ${Math.round(confidenceVal * 100)}%.`
              : `Our beauty NLP pipeline predicts this customer review expresses low or medium satisfaction with a model confidence score of ${Math.round(confidenceVal * 100)}%.`
            }
          </p>
        </div>

        {/* Confidence Analysis Section */}
        <div className="space-y-3 mb-8 pt-2">
          <h4 className="text-xs uppercase tracking-wider font-semibold text-plum flex justify-between items-center">
            <span>📊 Confidence Analysis</span>
            <span className="text-[9px] text-muted-plum font-normal lowercase tracking-normal">NLP probability distribution</span>
          </h4>
          <div className="space-y-4 bg-cream/40 p-4 rounded-xl border border-rose-gold/10">
            <div>
              <div className="flex justify-between text-xs font-semibold text-plum mb-1.5">
                <span>High Satisfaction Score</span>
                <span className="font-mono font-bold">{Math.round(result.probability_high_satisfaction * 100)}%</span>
              </div>
              <div className="w-full bg-cream border border-rose-gold/30 h-2-5 rounded-full overflow-hidden">
                <div 
                  className="bg-plum h-full rounded-full transition-all duration-500" 
                  style={{ width: `${result.probability_high_satisfaction * 100}%` }}
                ></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-semibold text-plum mb-1.5">
                <span>Low/Medium Satisfaction Score</span>
                <span className="font-mono font-bold">{Math.round(result.probability_low_or_medium * 100)}%</span>
              </div>
              <div className="w-full bg-cream border border-rose-gold/30 h-2-5 rounded-full overflow-hidden">
                <div 
                  className="bg-gold h-full rounded-full transition-all duration-500" 
                  style={{ width: `${result.probability_low_or_medium * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>

        {/* Analyzed Review Preview Section */}
        <div className="space-y-2 mb-4 pt-2">
          <h4 className="text-xs uppercase tracking-wider font-semibold text-plum">
            💬 Analyzed Review Preview
          </h4>
          <div className="p-4 bg-cream border border-rose-gold/20 rounded-xl shadow-inner">
            <p className="text-xs text-plum font-sans leading-relaxed italic">
              "{result.input_preview}"
            </p>
          </div>
        </div>
      </div>

      {/* Footer line with served by model */}
      <div className="text-[9px] uppercase tracking-wider text-muted-plum mt-6 pt-3 border-t border-rose-gold/10 text-right">
        Inference served by: <span className="font-semibold text-plum font-mono">{result.model_name}</span>
      </div>
    </div>
  );
};
export default PredictionResult;
