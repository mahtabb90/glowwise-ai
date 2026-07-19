import { useState, useEffect } from 'react';
import { api } from './services/api';
import type { 
  ModelStatus, 
  SatisfactionPredictionRequest, 
  SatisfactionPredictionResponse, 
  TopTermsResponse, 
  ClustersResponse 
} from './types/api';
import { PredictionForm } from './components/PredictionForm';
import { PredictionResult } from './components/PredictionResult';
import { TermsSection } from './components/TermsSection';
import { ClusterCards } from './components/ClusterCards';
import { ProjectStory } from './components/ProjectStory';

function App() {
  const [healthStatus, setHealthStatus] = useState<'loading' | 'online' | 'offline'>('loading');
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [topTerms, setTopTerms] = useState<TopTermsResponse | null>(null);
  const [clusters, setClusters] = useState<ClustersResponse | null>(null);
  
  const [termsLoading, setTermsLoading] = useState(false);
  const [clustersLoading, setClustersLoading] = useState(false);
  
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [predictionResult, setPredictionResult] = useState<SatisfactionPredictionResponse | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  // Load API data on mount
  useEffect(() => {
    async function checkBackendAndLoad() {
      try {
        setHealthStatus('loading');
        // Health check ping
        await api.getHealth();
        setHealthStatus('online');

        // Fetch model status
        const mStatus = await api.getModelStatus();
        setModelStatus(mStatus);

        // Fetch explainability report
        setTermsLoading(true);
        const termsData = await api.getTopTerms();
        setTopTerms(termsData);
        setTermsLoading(false);

        // Fetch clustering segments
        setClustersLoading(true);
        const clustersData = await api.getClusters();
        setClusters(clustersData);
        setClustersLoading(false);
      } catch (error) {
        console.error("Backend connection error:", error);
        setHealthStatus('offline');
        setTermsLoading(false);
        setClustersLoading(false);
      }
    }
    checkBackendAndLoad();
  }, []);

  const handlePredict = async (payload: SatisfactionPredictionRequest) => {
    setIsAnalyzing(true);
    setPredictionError(null);
    setPredictionResult(null);
    try {
      const res = await api.predictSatisfaction(payload);
      setPredictionResult(res);
    } catch (err: any) {
      console.error("Prediction failed:", err);
      setPredictionError(err.message || "Failed to retrieve satisfaction predictions.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const isOffline = healthStatus === 'offline';
  const isModelMissing = healthStatus === 'online' && !modelStatus?.model_loaded;

  return (
    <div className="min-h-screen bg-cream text-plum font-sans antialiased selection:bg-rose-gold selection:text-plum">
      {/* Premium Glass Header */}
      <header className="sticky top-0 z-50 backdrop-blur-md bg-cream/80 border-b border-rose-gold/20 py-4 px-6 md:px-12 flex justify-between items-center shadow-sm">
        <div className="flex items-center gap-3">
          <span className="text-2xl font-bold bg-plum text-cream px-2-5 py-1 rounded-xl shadow-sm">🌸</span>
          <div>
            <h1 className="text-lg font-black text-plum tracking-tight uppercase">GlowWise AI</h1>
            <p className="text-[9px] font-bold text-gold-dark uppercase tracking-wider">Beauty & Skincare Review Intelligence</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase font-bold text-muted-plum bg-cream border border-rose-gold/30 px-2 py-1 rounded-lg">
            v0.1.0
          </span>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-6 md:px-12 py-10">
        
        {/* Hero Section */}
        <section className="hero-banner p-6 md:p-8 rounded-2xl mb-10 text-center">
          {/* Ambient Glow & Decorative Curves */}
          <div className="hero-decorations-wrapper">
            {/* Left Side Glow & SVG */}
            <div className="glow-blob glow-blob-left-1" />
            <div className="glow-blob glow-blob-left-2" />
            <div className="glow-blob glow-blob-left-3" />
            
            <svg className="hero-svg-deco hero-svg-deco-left" viewBox="0 0 300 400" fill="none" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="left-grad-1" x1="0%" y1="100%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#FFEAD8" stopOpacity="0" />
                  <stop offset="30%" stopColor="#FFE5D9" stopOpacity="0.9" />
                  <stop offset="70%" stopColor="#FCD5CE" stopOpacity="0.95" />
                  <stop offset="100%" stopColor="#FFEAD8" stopOpacity="0" />
                </linearGradient>
                <linearGradient id="left-grad-2" x1="0%" y1="100%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#E8D3C4" stopOpacity="0" />
                  <stop offset="50%" stopColor="#FFFFFF" stopOpacity="0.85" />
                  <stop offset="100%" stopColor="#FFEAD8" stopOpacity="0" />
                </linearGradient>
                <linearGradient id="leaf-grad-left" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#FFE5D9" stopOpacity="0.95" />
                  <stop offset="100%" stopColor="#FCD5CE" stopOpacity="0.8" />
                </linearGradient>
                <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="5" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
              </defs>

              {/* Main swooping curves */}
              <path d="M -20,380 C 100,360 30,180 150,110 C 200,75 220,60 250,20" stroke="url(#left-grad-1)" strokeWidth="3.5" filter="url(#glow)" />
              <path d="M -20,380 C 100,360 30,180 150,110 C 200,75 220,60 250,20" stroke="#FFFFFF" strokeWidth="1" opacity="0.95" />
              <path d="M -40,350 C 70,330 10,160 120,90 C 160,65 190,50 210,10" stroke="url(#left-grad-2)" strokeWidth="1.5" opacity="0.85" />
              <path d="M 20,390 C 130,370 70,220 180,140" stroke="url(#left-grad-1)" strokeWidth="1" opacity="0.75" />

              {/* Leaf branch 1 */}
              <path d="M 50,270 Q 75,250 85,220" stroke="url(#left-grad-1)" strokeWidth="1.5" />
              <g transform="translate(85, 220) rotate(-35) scale(1.05)">
                <path d="M0,0 Q12,-15 24,-12 Q20,3 0,0 Z" fill="url(#leaf-grad-left)" />
                <path d="M0,0 Q12,-15 24,-12" stroke="#FFFFFF" strokeWidth="0.5" opacity="0.7" />
              </g>
              <g transform="translate(68, 245) rotate(-60) scale(0.85)">
                <path d="M0,0 Q12,-15 24,-12 Q20,3 0,0 Z" fill="url(#leaf-grad-left)" />
              </g>

              {/* Leaf branch 2 (higher up) */}
              <path d="M 105,170 Q 130,150 145,115" stroke="url(#left-grad-1)" strokeWidth="1.5" />
              <g transform="translate(145, 115) rotate(-15) scale(1.2)">
                <path d="M0,0 Q12,-15 24,-12 Q20,3 0,0 Z" fill="url(#leaf-grad-left)" />
                <path d="M0,0 Q12,-15 24,-12" stroke="#FFFFFF" strokeWidth="0.5" opacity="0.7" />
              </g>
              <g transform="translate(125, 140) rotate(-40) scale(0.95)">
                <path d="M0,0 Q12,-15 24,-12 Q20,3 0,0 Z" fill="url(#leaf-grad-left)" />
              </g>
              <g transform="translate(112, 160) rotate(20) scale(0.8)">
                <path d="M0,0 Q12,-15 24,-12 Q20,3 0,0 Z" fill="url(#leaf-grad-left)" opacity="0.95" />
              </g>

              {/* Sparkles / Stars */}
              <g className="sparkle-element" transform="translate(110, 260) scale(1.0)">
                <path d="M0,-10 L2,-2 L10,0 L2,2 L0,10 L-2,2 L-10,0 L-2,-2 Z" fill="#FFFFFF" opacity="0.95" />
                <circle cx="0" cy="0" r="3" fill="#FFEAD8" filter="url(#glow)" />
              </g>
              <g className="sparkle-element" style={{ animationDelay: '1.5s' }} transform="translate(170, 80) scale(1.25)">
                <path d="M0,-10 L2,-2 L10,0 L2,2 L0,10 L-2,2 L-10,0 L-2,-2 Z" fill="#FFFFFF" opacity="0.95" />
                <circle cx="0" cy="0" r="3" fill="#C39B6F" filter="url(#glow)" />
              </g>
              <g className="sparkle-element" style={{ animationDelay: '0.7s' }} transform="translate(40, 180) scale(0.8)">
                <path d="M0,-10 L2,-2 L10,0 L2,2 L0,10 L-2,2 L-10,0 L-2,-2 Z" fill="#FFFFFF" opacity="0.9" />
              </g>
              <g className="sparkle-element" style={{ animationDelay: '2.2s' }} transform="translate(210, 150) scale(0.85)">
                <path d="M0,-10 L2,-2 L10,0 L2,2 L0,10 L-2,2 L-10,0 L-2,-2 Z" fill="#FFFFFF" opacity="0.95" />
                <circle cx="0" cy="0" r="2" fill="#FFEAD8" filter="url(#glow)" />
              </g>
              
              {/* Soft round light orbs */}
              <circle cx="90" cy="310" r="5" fill="#FFFFFF" opacity="0.75" filter="url(#glow)" />
              <circle cx="195" cy="140" r="4" fill="#FFFFFF" opacity="0.65" filter="url(#glow)" />
            </svg>

            {/* Right Side Glow & SVG */}
            <div className="glow-blob glow-blob-right-1" />
            <div className="glow-blob glow-blob-right-2" />
            <div className="glow-blob glow-blob-right-3" />

            <svg className="hero-svg-deco hero-svg-deco-right" viewBox="0 0 300 400" fill="none" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="right-grad-1" x1="100%" y1="100%" x2="0%" y2="0%">
                  <stop offset="0%" stopColor="#FFEAD8" stopOpacity="0" />
                  <stop offset="30%" stopColor="#FFE5D9" stopOpacity="0.9" />
                  <stop offset="70%" stopColor="#FCD5CE" stopOpacity="0.95" />
                  <stop offset="100%" stopColor="#FFEAD8" stopOpacity="0" />
                </linearGradient>
                <linearGradient id="right-grad-2" x1="100%" y1="100%" x2="0%" y2="0%">
                  <stop offset="0%" stopColor="#E8D3C4" stopOpacity="0" />
                  <stop offset="50%" stopColor="#FFFFFF" stopOpacity="0.85" />
                  <stop offset="100%" stopColor="#FFEAD8" stopOpacity="0" />
                </linearGradient>
                <linearGradient id="leaf-grad-right" x1="100%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#FFE5D9" stopOpacity="0.95" />
                  <stop offset="100%" stopColor="#FCD5CE" stopOpacity="0.8" />
                </linearGradient>
                <filter id="glow-right" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="5" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
              </defs>

              {/* Main swooping curves */}
              <path d="M 320,380 C 200,360 270,180 150,110 C 100,75 80,60 50,20" stroke="url(#right-grad-1)" strokeWidth="3.5" filter="url(#glow-right)" />
              <path d="M 320,380 C 200,360 270,180 150,110 C 100,75 80,60 50,20" stroke="#FFFFFF" strokeWidth="1" opacity="0.95" />
              <path d="M 340,350 C 230,330 290,160 180,90 C 140,65 110,50 90,10" stroke="url(#right-grad-2)" strokeWidth="1.5" opacity="0.85" />
              <path d="M 280,390 C 170,370 230,220 120,140" stroke="url(#right-grad-1)" strokeWidth="1" opacity="0.75" />

              {/* Leaf branch 1 */}
              <path d="M 250,270 Q 225,250 215,220" stroke="url(#right-grad-1)" strokeWidth="1.5" />
              <g transform="translate(215, 220) rotate(-145) scale(1.05)">
                <path d="M0,0 Q12,-15 24,-12 Q20,3 0,0 Z" fill="url(#leaf-grad-right)" />
                <path d="M0,0 Q12,-15 24,-12" stroke="#FFFFFF" strokeWidth="0.5" opacity="0.7" />
              </g>
              <g transform="translate(232, 245) rotate(-120) scale(0.85)">
                <path d="M0,0 Q12,-15 24,-12 Q20,3 0,0 Z" fill="url(#leaf-grad-right)" />
              </g>

              {/* Leaf branch 2 (higher up) */}
              <path d="M 195,170 Q 170,150 155,115" stroke="url(#right-grad-1)" strokeWidth="1.5" />
              <g transform="translate(155, 115) rotate(-165) scale(1.2)">
                <path d="M0,0 Q12,-15 24,-12 Q20,3 0,0 Z" fill="url(#leaf-grad-right)" />
                <path d="M0,0 Q12,-15 24,-12" stroke="#FFFFFF" strokeWidth="0.5" opacity="0.7" />
              </g>
              <g transform="translate(175, 140) rotate(-140) scale(0.95)">
                <path d="M0,0 Q12,-15 24,-12 Q20,3 0,0 Z" fill="url(#leaf-grad-right)" />
              </g>
              <g transform="translate(188, 160) rotate(-200) scale(0.8)">
                <path d="M0,0 Q12,-15 24,-12 Q20,3 0,0 Z" fill="url(#leaf-grad-right)" opacity="0.95" />
              </g>

              {/* Sparkles / Stars */}
              <g className="sparkle-element" style={{ animationDelay: '0.5s' }} transform="translate(190, 260) scale(1.0)">
                <path d="M0,-10 L2,-2 L10,0 L2,2 L0,10 L-2,2 L-10,0 L-2,-2 Z" fill="#FFFFFF" opacity="0.9" />
                <circle cx="0" cy="0" r="3" fill="#FFEAD8" filter="url(#glow-right)" />
              </g>
              <g className="sparkle-element" style={{ animationDelay: '2s' }} transform="translate(130, 80) scale(1.15)">
                <path d="M0,-10 L2,-2 L10,0 L2,2 L0,10 L-2,2 L-10,0 L-2,-2 Z" fill="#FFFFFF" opacity="0.95" />
                <circle cx="0" cy="0" r="3" fill="#C39B6F" filter="url(#glow-right)" />
              </g>
              <g className="sparkle-element" style={{ animationDelay: '1.2s' }} transform="translate(260, 180) scale(0.8)">
                <path d="M0,-10 L2,-2 L10,0 L2,2 L0,10 L-2,2 L-10,0 L-2,-2 Z" fill="#FFFFFF" opacity="0.8" />
              </g>
              <g className="sparkle-element" style={{ animationDelay: '1.8s' }} transform="translate(90, 150) scale(0.85)">
                <path d="M0,-10 L2,-2 L10,0 L2,2 L0,10 L-2,2 L-10,0 L-2,-2 Z" fill="#FFFFFF" opacity="0.95" />
                <circle cx="0" cy="0" r="2" fill="#FFEAD8" filter="url(#glow-right)" />
              </g>

              {/* Soft round light orbs */}
              <circle cx="210" cy="310" r="5" fill="#FFFFFF" opacity="0.75" filter="url(#glow-right)" />
              <circle cx="105" cy="140" r="4" fill="#FFFFFF" opacity="0.65" filter="url(#glow-right)" />
            </svg>
          </div>

          <div className="max-w-2xl mx-auto space-y-4">
            <h2 className="text-4xl md:text-5xl font-black text-luxury-gradient tracking-tight leading-none mb-1">
              GlowWise AI
            </h2>
            <p className="text-sm md:text-base text-plum font-medium leading-relaxed">
              Your smart skincare review companion for beauty insights and product feedback.
            </p>
            <p className="text-xs text-muted-plum leading-relaxed max-w-xl mx-auto">
              An end-to-end ML platform featuring classification, explainability, customer personas, transfer learning, anomaly detection and live deployment.
            </p>
          </div>
          
          {/* Centered premium collage layout */}
          <div className="hero-collage-container animate-fade-in">
            {/* Left side card: Skincare Cream Jar */}
            <div className="hero-collage-side-card left-card">
              <img 
                src="/skincare_cream.png" 
                alt="Skincare Cream Jar" 
              />
            </div>
            
            {/* Center main card: Serum Bottle */}
            <div className="hero-image-card">
              <img 
                src="/skincare_hero.png" 
                alt="GlowWise Serum Bottle" 
              />
            </div>

            {/* Right side card: Glowing Skincare Model */}
            <div className="hero-collage-side-card right-card">
              <img 
                src="/skincare_model.png" 
                alt="Glowing Skin Model" 
              />
            </div>
          </div>
        </section>

        {/* Offline Banner */}
        {isOffline && (
          <div className="p-4 bg-red-50 border border-red-200 text-red-800 text-sm rounded-2xl mb-8 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div>
              <strong>🔴 FastAPI Backend Offline:</strong> The React client could not reach the server at 
              <a href="http://localhost:8000" className="underline font-semibold ml-1 text-red-950">http://localhost:8000</a>.
            </div>
            <div className="text-xs font-semibold bg-red-100 px-3 py-1 rounded-lg border border-red-200">
              Start Server: <code className="font-mono ml-1 text-[10px]">uvicorn app.main:app --reload</code>
            </div>
          </div>
        )}

        {/* Inference Workspace Row */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          {/* Prediction form */}
          <PredictionForm 
            onPredict={handlePredict} 
            isAnalyzing={isAnalyzing} 
            disabled={isOffline || isModelMissing} 
          />

          {/* Inference results */}
          <PredictionResult 
            result={predictionResult} 
            error={predictionError} 
          />
        </section>

        {/* Model explainability terms section */}
        <TermsSection 
          terms={topTerms} 
          loading={termsLoading} 
          error={isOffline ? "API server is offline. Cannot load explainability drivers." : null} 
        />

        {/* Unsupervised segments profiles cards */}
        <ClusterCards 
          clusters={clusters} 
          loading={clustersLoading} 
          error={isOffline ? "API server is offline. Displaying dashboard fallback personas." : null} 
        />

        {/* Project story roadmap */}
        <ProjectStory />

      </main>

      {/* Pearl Glass Footer */}
      <footer className="border-t border-rose-gold/20 py-8 text-center bg-cream/40 text-xs text-muted-plum">
        <p>© 2026 GlowWise AI. Built for wellness intelligence. Data source: Sephora Skincare Reviews.</p>
        <p className="mt-1 font-semibold text-[10px] text-gold uppercase tracking-widest">Natural Language Processing Inference & Clustering App</p>
      </footer>
    </div>
  );
}

export default App;
