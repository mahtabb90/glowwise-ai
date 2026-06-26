import { useState, useEffect } from 'react';
import { api, type HealthResponse } from './services/api';
import './App.css';

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthStatus, setHealthStatus] = useState<'loading' | 'online' | 'offline'>('loading');
  const [reviewInput, setReviewInput] = useState('');
  const [predictionResult, setPredictionResult] = useState<any | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Fetch API Health Status on mount
  useEffect(() => {
    async function checkSystemHealth() {
      try {
        const data = await api.getHealth();
        setHealth(data);
        setHealthStatus('online');
      } catch (error) {
        console.error("Health check error:", error);
        setHealthStatus('offline');
      }
    }
    checkSystemHealth();
  }, []);

  // Preset reviews for the interactive test
  const presets = [
    {
      title: "🌟 Glowing Results",
      text: "I absolutely love this serum! It leaves my skin glowing, super hydrated, and did not cause any breakouts. The packaging is gorgeous, but the dropper could be slightly better."
    },
    {
      title: "⚠️ Skin Irritation",
      text: "Disappointed with this product. It felt extremely sticky, had a strong perfume scent, and broke me out after just two days. Definitely not for sensitive skin. Waste of money."
    },
    {
      title: "🧴 Neutral Review",
      text: "It is an okay cleanser. The texture is nice and creamy, but it leaves my skin feeling a bit dry. It gets the job done, but it is not worth the premium price tag."
    }
  ];

  const handlePresetSelect = (text: string) => {
    setReviewInput(text);
    handleInferenceSimulate(text);
  };

  const handleInferenceSimulate = (text: string) => {
    if (!text.trim()) return;
    setIsAnalyzing(true);
    
    // Simulate API delay for a polished UX
    setTimeout(() => {
      const cleaned = text.toLowerCase();
      
      const positiveWords = ["love", "glowing", "glow", "hydrated", "best", "gorgeous", "amazing", "smooth", "perfect"];
      const negativeWords = ["disappointed", "sticky", "broke me out", "breakouts", "irritation", "dry", "waste", "strong", "perfume"];
      
      let score = 0.5;
      positiveWords.forEach(word => {
        if (cleaned.includes(word)) score += 0.15;
      });
      negativeWords.forEach(word => {
        if (cleaned.includes(word)) score -= 0.15;
      });
      
      score = Math.max(0.0, Math.min(1.0, score));
      
      let sentiment = 'neutral';
      let confidence = 0.5;
      
      if (score > 0.6) {
        sentiment = 'positive';
        confidence = score;
      } else if (score < 0.4) {
        sentiment = 'negative';
        confidence = 1 - score;
      } else {
        confidence = 0.5 + Math.abs(0.5 - score);
      }
      
      // Determine aspects
      const aspects = {
        efficacy: cleaned.includes("glowing") || cleaned.includes("glow") || cleaned.includes("cleared") ? "Positive" : 
                  cleaned.includes("broke") || cleaned.includes("irritation") ? "Negative" : "Neutral",
        texture: cleaned.includes("sticky") ? "Sticky/Negative" : cleaned.includes("smooth") || cleaned.includes("creamy") ? "Creamy/Positive" : "Neutral",
        packaging: cleaned.includes("dropper") || cleaned.includes("leaks") || cleaned.includes("bottle") ? "Needs Improvement" : "Neutral"
      };

      setPredictionResult({
        sentiment,
        confidence: Math.round(confidence * 100),
        rating: Math.round((1 + score * 4) * 10) / 10,
        aspects
      });
      setIsAnalyzing(false);
    }, 600);
  };

  return (
    <div className="glowwise-app">
      {/* Header */}
      <header className="header">
        <div className="container header-content">
          <div className="logo">
            Glow<span>Wise</span> AI
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span className="form-label" style={{ margin: 0, fontSize: '0.75rem' }}>API Backend:</span>
            {healthStatus === 'loading' && (
              <span className="badge badge-gold">
                <span className="pulse-dot" style={{ backgroundColor: 'var(--status-warning)' }}></span> Loading...
              </span>
            )}
            {healthStatus === 'online' && (
              <span className="badge badge-status">
                <span className="pulse-dot"></span> Online (v{health?.version})
              </span>
            )}
            {healthStatus === 'offline' && (
              <span className="badge badge-status error">
                <span className="pulse-dot error"></span> Offline
              </span>
            )}
          </div>
        </div>
      </header>

      <main style={{ paddingBottom: '80px' }}>
        {/* Hero Section */}
        <section style={{ padding: '80px 0 60px 0', textAlign: 'center', background: 'radial-gradient(circle, rgba(249,235,224,0.3) 0%, rgba(252,250,247,0.1) 70%)' }}>
          <div className="container">
            <span className="badge badge-gold" style={{ marginBottom: '16px' }}>Machine Learning Portfolio Project</span>
            <h1 style={{ fontSize: '3.6rem', marginBottom: '24px', letterSpacing: '-0.5px' }}>
              GlowWise AI
            </h1>
            <p style={{ fontSize: '1.25rem', fontFamily: 'var(--font-serif)', fontStyle: 'italic', color: 'var(--text-muted)', maxWidth: '700px', margin: '0 auto 40px auto' }}>
              Skincare Review Intelligence powered by Machine Learning
            </p>
            <p style={{ maxWidth: '600px', margin: '0 auto 40px auto', color: 'var(--text-muted)', fontSize: '1.05rem' }}>
              Analyze Sephora skincare reviews and product formulation metadata to predict customer satisfaction, extract fine-grained sentiments, and cluster beauty products dynamically.
            </p>
            <div style={{ display: 'flex', gap: '16px', justifyContent: 'center' }}>
              <a href="#interactive-demo" className="btn btn-primary">Try Interactive Demo</a>
              <a href="#architecture" className="btn btn-outline">View Capabilities</a>
            </div>
          </div>
        </section>

        {/* Live Interactive Tester */}
        <section id="interactive-demo" style={{ padding: '60px 0' }}>
          <div className="container">
            <h2 style={{ textAlign: 'center', marginBottom: '16px', fontSize: '2.2rem' }}>Skincare Review Sentiment Tester</h2>
            <p style={{ textAlign: 'center', color: 'var(--text-muted)', marginBottom: '40px', maxWidth: '600px', margin: '0 auto 40px auto' }}>
              Type a custom skincare review or load one of our presets to run a simulated client-side NLP review parsing and satisfaction estimation.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px', alignItems: 'start' }}>
              {/* Review input card */}
              <div className="card">
                <div style={{ marginBottom: '24px' }}>
                  <span className="form-label">Presets</span>
                  <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                    {presets.map((p, idx) => (
                      <button 
                        key={idx} 
                        onClick={() => handlePresetSelect(p.text)}
                        className="btn btn-outline" 
                        style={{ padding: '6px 14px', fontSize: '0.8rem', borderRadius: '15px', border: '1px solid var(--border-color)' }}
                      >
                        {p.title}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="review-textarea">Review Text</label>
                  <textarea
                    id="review-textarea"
                    className="form-input"
                    rows={6}
                    placeholder="Write your review here... (e.g. 'I love the feel of this moisturizer. It keeps my face smooth and glowy without feeling heavy.')"
                    value={reviewInput}
                    onChange={(e) => setReviewInput(e.target.value)}
                    style={{ resize: 'none' }}
                  />
                </div>

                <button 
                  className="btn btn-primary" 
                  onClick={() => handleInferenceSimulate(reviewInput)}
                  disabled={!reviewInput.trim() || isAnalyzing}
                  style={{ width: '100%' }}
                >
                  {isAnalyzing ? 'Analyzing Review Features...' : 'Run Sentiment Classifier'}
                </button>
              </div>

              {/* Analysis result card */}
              <div className="card" style={{ minHeight: '344px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                {!predictionResult && !isAnalyzing ? (
                  <div style={{ textAlign: 'center', padding: '40px 0' }}>
                    <div style={{ fontSize: '3rem', color: 'var(--rose-gold)', marginBottom: '16px' }}>✨</div>
                    <h3 style={{ marginBottom: '8px' }}>Awaiting Inference Input</h3>
                    <p style={{ color: 'var(--text-light)', fontSize: '0.95rem' }}>Write or select a review to preview classification and regression metrics.</p>
                  </div>
                ) : isAnalyzing ? (
                  <div style={{ textAlign: 'center', padding: '40px 0' }}>
                    <div className="pulse-dot" style={{ width: '20px', height: '20px', backgroundColor: 'var(--gold-champagne)', marginBottom: '16px' }}></div>
                    <h3>Parsing aspect parameters...</h3>
                    <p style={{ color: 'var(--text-light)', fontSize: '0.95rem' }}>Computing word weights and ratings.</p>
                  </div>
                ) : (
                  <div>
                    <h3 style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '12px', marginBottom: '20px' }}>Inference Outcomes</h3>
                    
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '24px' }}>
                      <div>
                        <span className="form-label" style={{ fontSize: '0.75rem' }}>Sentiment Class</span>
                        <span 
                          className={`badge ${predictionResult.sentiment === 'positive' ? 'badge-status' : predictionResult.sentiment === 'negative' ? 'badge-status error' : 'badge-gold'}`} 
                          style={{ fontSize: '0.9rem', padding: '6px 16px' }}
                        >
                          {predictionResult.sentiment.toUpperCase()}
                        </span>
                      </div>
                      <div>
                        <span className="form-label" style={{ fontSize: '0.75rem' }}>Confidence Score</span>
                        <span style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--plum-primary)' }}>{predictionResult.confidence}%</span>
                      </div>
                    </div>

                    <div style={{ marginBottom: '24px' }}>
                      <span className="form-label" style={{ fontSize: '0.75rem' }}>Predicted Customer Rating</span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: '2.5rem', fontWeight: 700, fontFamily: 'var(--font-serif)', color: 'var(--plum-primary)' }}>
                          {predictionResult.rating}
                        </span>
                        <span style={{ color: 'var(--text-light)', fontSize: '1.2rem' }}>/ 5.0</span>
                        <div style={{ display: 'flex', color: 'var(--gold-champagne)', marginLeft: '12px' }}>
                          {Array.from({ length: 5 }).map((_, i) => (
                            <span key={i} style={{ fontSize: '1.2rem' }}>
                              {i < Math.floor(predictionResult.rating) ? '★' : '☆'}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div>
                      <span className="form-label" style={{ fontSize: '0.75rem', marginBottom: '12px' }}>Extracted aspect sentiments</span>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', borderBottom: '1px dotted var(--border-color)', paddingBottom: '4px' }}>
                          <span style={{ fontWeight: 500 }}>Formulation Efficacy:</span>
                          <span style={{ color: predictionResult.aspects.efficacy === 'Positive' ? 'var(--status-healthy)' : predictionResult.aspects.efficacy === 'Negative' ? 'var(--status-error)' : 'var(--text-muted)' }}>
                            {predictionResult.aspects.efficacy}
                          </span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', borderBottom: '1px dotted var(--border-color)', paddingBottom: '4px' }}>
                          <span style={{ fontWeight: 500 }}>Texture & Feel:</span>
                          <span style={{ color: 'var(--text-muted)' }}>{predictionResult.aspects.texture}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem' }}>
                          <span style={{ fontWeight: 500 }}>Packaging / Usability:</span>
                          <span style={{ color: 'var(--text-muted)' }}>{predictionResult.aspects.packaging}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* Capabilities Section */}
        <section id="architecture" style={{ padding: '60px 0', backgroundColor: 'var(--bg-accent)' }}>
          <div className="container">
            <h2 style={{ textAlign: 'center', marginBottom: '16px', fontSize: '2.2rem' }}>Core Capabilities</h2>
            <p style={{ textAlign: 'center', color: 'var(--text-muted)', marginBottom: '50px', maxWidth: '600px', margin: '0 auto 50px auto' }}>
              The ML backend leverages textual modeling and clustering techniques to provide holistic reviews intelligence.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '24px' }}>
              <div className="card">
                <span className="badge badge-gold" style={{ marginBottom: '16px' }}>NLP PIPELINE</span>
                <h3 style={{ marginBottom: '12px' }}>Review Sentiment</h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
                  Extract aspect-based sentiment signals from unstructured text. Group reviews by product texture, scent, skin irritation, and package efficiency.
                </p>
              </div>

              <div className="card">
                <span className="badge badge-gold" style={{ marginBottom: '16px' }}>REGRESSION</span>
                <h3 style={{ marginBottom: '12px' }}>Satisfaction Predictor</h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
                  Predict customer satisfaction ratings and NPS score probability based on combination of review length, token density, and text embeddings.
                </p>
              </div>

              <div className="card">
                <span className="badge badge-gold" style={{ marginBottom: '16px' }}>UNSUPERVISED ML</span>
                <h3 style={{ marginBottom: '12px' }}>Product Clustering</h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
                  Apply K-Means clustering algorithm to product ingredient formulas to map products with similar skin concern alignment and profiles.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Dashboard Mock Preview */}
        <section style={{ padding: '60px 0' }}>
          <div className="container">
            <h2 style={{ textAlign: 'center', marginBottom: '16px', fontSize: '2.2rem' }}>Analytics Dashboard Preview</h2>
            <p style={{ textAlign: 'center', color: 'var(--text-muted)', marginBottom: '50px', maxWidth: '600px', margin: '0 auto 50px auto' }}>
              An executive UI mapping the skincare product feedback landscape for marketing and formulation decisions.
            </p>

            <div className="card" style={{ padding: '40px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '16px', marginBottom: '24px' }}>
                <span style={{ fontFamily: 'var(--font-serif)', fontSize: '1.25rem', fontWeight: 600 }}>Product Formulation Matrix (Ingredient PCA)</span>
                <span className="badge badge-gold">Interactive Map Mock</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '3fr 1fr', gap: '32px' }}>
                {/* PCA Plot Mock */}
                <div style={{ height: '300px', backgroundColor: 'var(--bg-cream)', borderRadius: '8px', border: '1px dashed var(--border-color)', position: 'relative', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {/* Grid Lines */}
                  <div style={{ position: 'absolute', top: '50%', left: 0, width: '100%', height: '1px', borderBottom: '1px dotted var(--border-color)' }}></div>
                  <div style={{ position: 'absolute', left: '50%', top: 0, width: '1px', height: '100%', borderRight: '1px dotted var(--border-color)' }}></div>
                  
                  {/* Scatter dots */}
                  <div style={{ position: 'absolute', top: '25%', left: '30%', width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--plum-primary)', opacity: 0.7, cursor: 'pointer' }} title="Acne Salicylic Serum"></div>
                  <div style={{ position: 'absolute', top: '30%', left: '35%', width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--plum-primary)', opacity: 0.6 }} title="Purifying Clay Cleanser"></div>
                  <div style={{ position: 'absolute', top: '28%', left: '25%', width: '14px', height: '14px', borderRadius: '50%', backgroundColor: 'var(--plum-primary)', opacity: 0.8 }} title="BHA Exfoliating Toner"></div>
                  
                  <div style={{ position: 'absolute', top: '70%', left: '70%', width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--gold-champagne)', opacity: 0.7 }} title="Ceramide Rich Moisturizer"></div>
                  <div style={{ position: 'absolute', top: '75%', left: '65%', width: '14px', height: '14px', borderRadius: '50%', backgroundColor: 'var(--gold-champagne)', opacity: 0.8 }} title="Squalane Face Oil"></div>
                  <div style={{ position: 'absolute', top: '65%', left: '75%', width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--gold-champagne)', opacity: 0.6 }} title="Hyaluronic Hydration Serum"></div>

                  <div style={{ position: 'absolute', top: '15%', left: '80%', width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--rose-gold)', opacity: 0.7 }} title="Gentle Milky Cleanser"></div>
                  <div style={{ position: 'absolute', top: '10%', left: '75%', width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--rose-gold)', opacity: 0.6 }} title="Soothing Oatmeal Toner"></div>

                  <span style={{ color: 'var(--text-light)', fontSize: '0.85rem', fontStyle: 'italic', zIndex: 1 }}>Hover or click points to view formulation profiles</span>
                </div>

                {/* Legend / Metrics */}
                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: '20px', textAlign: 'left' }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                      <span style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--plum-primary)' }}></span>
                      <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Active Acids</span>
                    </div>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Targeting exfoliation & oily/acne-prone skins.</p>
                  </div>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                      <span style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--gold-champagne)' }}></span>
                      <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Barrier Repair</span>
                    </div>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Targeting hydration & dry/compromised skins.</p>
                  </div>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                      <span style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--rose-gold)' }}></span>
                      <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Sensitive Care</span>
                    </div>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Fragrance-free soothing cleansers and balms.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer style={{ borderTop: '1px solid var(--border-color)', padding: '40px 0', backgroundColor: 'var(--bg-card)', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <p>© {new Date().getFullYear()} GlowWise AI.</p>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-light)' }}>Skincare Review NLP & Clustering Project</p>
          </div>
          <div>
            <a href="https://github.com" target="_blank" rel="noopener noreferrer" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span>GitHub Repository</span>
              <span style={{ fontSize: '0.8rem' }}>↗</span>
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
