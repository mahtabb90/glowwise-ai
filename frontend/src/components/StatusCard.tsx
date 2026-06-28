import React from 'react';
import type { ModelStatus } from '../types/api';

interface StatusCardProps {
  apiStatus: 'online' | 'offline' | 'loading';
  modelStatus: ModelStatus | null;
  loading: boolean;
}

export const StatusCard: React.FC<StatusCardProps> = ({ apiStatus, modelStatus, loading }) => {
  const isApiOnline = apiStatus === 'online';
  const isModelReady = modelStatus?.model_loaded;

  return (
    <div className="glass-card p-4 rounded-2xl mb-8 border border-rose-gold/10 shadow-sm bg-cream/30 flex flex-col md:flex-row justify-between items-center gap-4 transition-all duration-300">
      {/* Title */}
      <div className="flex items-center gap-2">
        <span className="text-base">🌸</span>
        <h3 className="text-xs uppercase font-extrabold tracking-widest text-plum">GlowWise Status</h3>
      </div>
      
      {/* Status Indicators Row */}
      <div className="flex flex-wrap items-center justify-center gap-3">
        {/* API Connection Pill */}
        <div className="flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-semibold bg-cream border border-rose-gold/20">
          <span className="text-muted-plum font-normal">App Link:</span>
          {loading ? (
            <span className="flex items-center gap-1 text-gold-dark">
              <span className="w-1.5 h-1.5 rounded-full bg-gold animate-pulse"></span>
              Checking...
            </span>
          ) : isApiOnline ? (
            <span className="flex items-center gap-1 text-green-600">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500"></span>
              Ready
            </span>
          ) : (
            <span className="flex items-center gap-1 text-red-600">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500"></span>
              Unavailable
            </span>
          )}
        </div>

        {/* Model State Pill */}
        <div className="flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-semibold bg-cream border border-rose-gold/20">
          <span className="text-muted-plum font-normal">Review Intelligence:</span>
          {!isApiOnline ? (
            <span className="flex items-center gap-1 text-red-600">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500"></span>
              Offline
            </span>
          ) : isModelReady ? (
            <span className="flex items-center gap-1 text-green-600">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500"></span>
              Online
            </span>
          ) : (
            <span className="flex items-center gap-1 text-gold-dark">
              <span className="w-1.5 h-1.5 rounded-full bg-gold"></span>
              No Model Loaded
            </span>
          )}
        </div>

        {/* Model Name Pill */}
        {isApiOnline && isModelReady && (
          <div className="flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-semibold bg-cream border border-rose-gold/20">
            <span className="text-muted-plum font-normal">Active Engine:</span>
            <span className="text-plum font-mono font-bold text-[10px]">{modelStatus?.model_name}</span>
          </div>
        )}
      </div>

      {/* Connection Warning Notices */}
      {!isApiOnline && (
        <div className="w-full md:w-auto p-2 bg-red-50 text-red-700 text-[10px] rounded-lg border border-red-100 font-medium">
          ⚠️ Connection Required: run <code className="bg-red-100 px-1 py-0.5 rounded font-mono text-[9px]">cd backend && uvicorn app.main:app --reload</code>
        </div>
      )}

      {isApiOnline && !isModelReady && (
        <div className="w-full md:w-auto p-2 bg-gold/10 text-gold-950 text-[10px] rounded-lg border border-gold/20 font-medium">
          ⚠️ Artifact Missing: run <code className="bg-gold/15 px-1 py-0.5 rounded font-mono text-[9px]">python ml/src/model_comparison.py</code>
        </div>
      )}
    </div>
  );
};
export default StatusCard;
