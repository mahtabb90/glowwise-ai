import React, { useState } from 'react';
import type { SatisfactionPredictionRequest } from '../types/api';

interface PredictionFormProps {
  onPredict: (payload: SatisfactionPredictionRequest) => void;
  isAnalyzing: boolean;
  disabled: boolean;
}

export const PredictionForm: React.FC<PredictionFormProps> = ({ onPredict, isAnalyzing, disabled }) => {
  const [text, setText] = useState('');
  const [title, setTitle] = useState('');
  const [product, setProduct] = useState('');
  const [brand, setBrand] = useState('');
  const [validationError, setValidationError] = useState<string | null>(null);

  const presets = {
    positive: {
      title: "Holy grail moisturizer!",
      text: "I absolutely love this moisturizer! It makes my skin feel incredibly smooth and glowing. Highly recommend it.",
      product: "Ultra Facial Cream",
      brand: "Kiehl's"
    },
    negative: {
      title: "Broke me out",
      text: "Highly disappointed with this product. It broke me out in acne all over my face within two days and left a greasy residue.",
      product: "Super Hydrator Gel",
      brand: "Murad"
    }
  };

  const handleApplyPreset = (type: 'positive' | 'negative') => {
    const p = presets[type];
    setTitle(p.title);
    setText(p.text);
    setProduct(p.product);
    setBrand(p.brand);
    setValidationError(null);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) {
      setValidationError("Review text is required to run inference.");
      return;
    }
    setValidationError(null);
    onPredict({
      review_text: text,
      review_title: title,
      product_name: product,
      brand_name: brand
    });
  };

  return (
    <div className="glass-card p-6 rounded-2xl border border-rose-gold/20 shadow-sm h-full">
      <h3 className="text-xl font-bold text-plum mb-4 flex items-center gap-2">
        <span>🌸</span> Try GlowWise
      </h3>

      <div className="flex gap-2 mb-6">
        <button
          type="button"
          onClick={() => handleApplyPreset('positive')}
          disabled={disabled || isAnalyzing}
          className="flex-1 text-xs py-2 px-3 rounded-lg border border-green-200 bg-green-50 text-green-700 hover:bg-green-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          ✨ Apply Positive Example
        </button>
        <button
          type="button"
          onClick={() => handleApplyPreset('negative')}
          disabled={disabled || isAnalyzing}
          className="flex-1 text-xs py-2 px-3 rounded-lg border border-red-200 bg-red-50 text-red-700 hover:bg-red-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          💥 Apply Negative Example
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Optional Metadata Row */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-semibold text-muted-plum mb-1">Product Name</label>
            <input
              type="text"
              placeholder="e.g., Ultra Facial Cream"
              value={product}
              onChange={(e) => setProduct(e.target.value)}
              disabled={disabled || isAnalyzing}
              className="w-full bg-cream border border-rose-gold/30 rounded-xl px-3 py-2 text-sm text-plum placeholder-plum/40 focus:outline-none focus:border-gold transition-colors disabled:opacity-50"
            />
          </div>
          <div>
            <label className="block text-xs font-semibold text-muted-plum mb-1">Brand Name</label>
            <input
              type="text"
              placeholder="e.g., Kiehl's"
              value={brand}
              onChange={(e) => setBrand(e.target.value)}
              disabled={disabled || isAnalyzing}
              className="w-full bg-cream border border-rose-gold/30 rounded-xl px-3 py-2 text-sm text-plum placeholder-plum/40 focus:outline-none focus:border-gold transition-colors disabled:opacity-50"
            />
          </div>
        </div>

        {/* Review Title */}
        <div>
          <label className="block text-xs font-semibold text-muted-plum mb-1">Review Title</label>
          <input
            type="text"
            placeholder="e.g., Best moisturizer ever!"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            disabled={disabled || isAnalyzing}
            className="w-full bg-cream border border-rose-gold/30 rounded-xl px-3 py-2 text-sm text-plum placeholder-plum/40 focus:outline-none focus:border-gold transition-colors disabled:opacity-50"
          />
        </div>

        {/* Review Body */}
        <div>
          <label className="block text-xs font-semibold text-muted-plum mb-1">
            Review Text <span className="text-red-500">*</span>
          </label>
          <textarea
            rows={4}
            placeholder="Write your skincare product experience here..."
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              if (e.target.value.trim()) setValidationError(null);
            }}
            disabled={disabled || isAnalyzing}
            className="w-full bg-cream border border-rose-gold/30 rounded-xl px-3 py-2 text-sm text-plum placeholder-plum/40 focus:outline-none focus:border-gold transition-colors disabled:opacity-50 resize-y"
          />
          {validationError && (
            <p className="text-red-600 text-xs mt-1 font-semibold">{validationError}</p>
          )}
        </div>

        {/* Submit */}
        <button
          type="submit"
          disabled={disabled || isAnalyzing}
          className="w-full bg-plum text-cream hover:bg-light-plum font-bold py-2-5 px-4 rounded-xl text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isAnalyzing ? (
            <>
              <span className="w-4 h-4 border-2 border-cream border-t-transparent rounded-full animate-spin"></span>
              Analyzing Review Sentiment...
            </>
          ) : (
            'Analyze Customer Review'
          )}
        </button>
      </form>
    </div>
  );
};
