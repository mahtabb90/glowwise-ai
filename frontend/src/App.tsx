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
          <div className="max-w-2xl mx-auto space-y-4">
            <h2 className="text-4xl md:text-5xl font-black text-luxury-gradient tracking-tight leading-none mb-1">
              GlowWise AI
            </h2>
            <p className="text-sm md:text-base text-plum font-medium leading-relaxed">
              Your smart skincare review companion for beauty insights and product feedback.
            </p>
            <p className="text-xs text-muted-plum leading-relaxed">
              Discover what people truly feel about skincare products.
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
