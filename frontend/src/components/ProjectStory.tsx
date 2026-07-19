import React from 'react';

export const ProjectStory: React.FC = () => {
  const steps = [
    {
      phase: "1. Dataset Audit",
      title: "Auditing Sephora Skincare Data",
      icon: "📊",
      desc: "Analyzed review ratings, missing values and class distributions to define the prediction target."
    },
    {
      phase: "2. Preprocessing",
      title: "Creating Clean ML-Ready Datasets",
      icon: "🌿",
      desc: "Merged product metadata, cleaned review text and prepared structured training data."
    },
    {
      phase: "3. Benchmarking",
      title: "Establishing Baselines",
      icon: "🧪",
      desc: "Started with Dummy and TF-IDF + Logistic Regression models to create a reliable baseline."
    },
    {
      phase: "4. Model Optimization",
      title: "Comparing Classical ML Models",
      icon: "🔬",
      desc: "Compared Logistic Regression, LinearSVC, Naive Bayes, SGD and metadata-enhanced models."
    },
    {
      phase: "5. Advanced ML & NLP",
      title: "Testing ANN, Text CNN & Transformer",
      icon: "🧠",
      desc: "Extended the experiments with neural networks, TensorFlow/Keras Text CNN and transformer-based transfer learning embeddings."
    },
    {
      phase: "6. Explainability",
      title: "Explaining Model Decisions",
      icon: "💬",
      desc: "Identified positive and negative word drivers behind customer satisfaction predictions."
    },
    {
      phase: "7. Customer Segments",
      title: "Discovering Beauty Personas",
      icon: "👥",
      desc: "Used unsupervised K-Means clustering to discover review-based customer personas."
    },
    {
      phase: "8. Anomaly Detection",
      title: "Monitoring Unusual Reviews",
      icon: "🛡️",
      desc: "Added an autoencoder experiment to flag unusual skincare reviews using reconstruction error."
    },
    {
      phase: "9. Future Extension",
      title: "TCN Sequence Modeling",
      icon: "🔮",
      desc: "Documented how Temporal Convolutional Networks could model review text as token sequences in future work."
    },
    {
      phase: "10. Serving API",
      title: "Exposing ML Model Endpoints",
      icon: "⚡",
      desc: "Built FastAPI services with lazy model loading and prediction endpoints."
    },
    {
      phase: "11. Skincare UI",
      title: "Interactive Web Dashboard",
      icon: "🌸",
      desc: "Implemented a responsive React dashboard for predictions, explainability and customer insights."
    }
  ];

  return (
    <div className="glass-card p-6 rounded-2xl border border-rose-gold/20 shadow-sm mb-12">
      <h3 className="text-xl font-bold text-plum mb-2 flex items-center gap-2">
        <span>🌸</span> Behind the experience
      </h3>
      <p className="text-xs text-muted-plum mb-8">
        A timeline of the engineering phases from raw Sephora review datasets to a deployed beauty-tech dashboard:
      </p>

      <div className="timeline-container">
        {steps.map((step, idx) => (
          <div key={`story-${idx}`} className="timeline-item">
            {/* Timeline icon circle badge */}
            <span className="timeline-icon-badge text-sm">
              {step.icon}
            </span>
            
            <div className="text-[10px] uppercase font-bold text-gold-dark tracking-widest">{step.phase}</div>
            <h4 className="text-base font-bold text-plum mt-1">{step.title}</h4>
            <p className="text-xs text-muted-plum mt-1 leading-relaxed">{step.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
};
export default ProjectStory;
