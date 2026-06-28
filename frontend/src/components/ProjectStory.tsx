import React from 'react';

export const ProjectStory: React.FC = () => {
  const steps = [
    {
      phase: "1. Dataset Audit",
      title: "Auditing Sephora Skincare Data",
      icon: "📊",
      desc: "Analyzed review ratings and verified class distributions to isolate baseline targets."
    },
    {
      phase: "2. Preprocessing",
      title: "Creating Clean ML-Ready Datasets",
      icon: "🌿",
      desc: "Merged product metadata and filtered duplicates to prepare structured training rows."
    },
    {
      phase: "3. Benchmarking",
      title: "Establishing Baselines",
      icon: "🧪",
      desc: "Fitted a TF-IDF + Logistic Regression baseline achieving an 87.5% Macro F1 score."
    },
    {
      phase: "4. Optimization",
      title: "Comparing 7 ML Architectures",
      icon: "🔬",
      desc: "Ran grid search over regularization strengths, selecting Tuned Logistic Regression."
    },
    {
      phase: "5. Explainability",
      title: "Extracting Word Coefficients",
      icon: "💬",
      desc: "Identified positive satisfaction drivers and isolated confidently wrong classification cases."
    },
    {
      phase: "6. Customer Segments",
      title: "Unsupervised K-Means Clustering",
      icon: "👥",
      desc: "Reduced dimensions using SVD and profiled 5 cohorts with post-hoc sat. rates."
    },
    {
      phase: "7. Serving API",
      title: "Exposing ML Model Endpoints",
      icon: "⚡",
      desc: "Constructed FastAPI services with lazy model loading and robust payload validations."
    },
    {
      phase: "8. Skincare UI",
      title: "Interactive Web Dashboard",
      icon: "🌸",
      desc: "Implemented a responsive, premium React dashboard showcasing predictions and segmentation."
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
