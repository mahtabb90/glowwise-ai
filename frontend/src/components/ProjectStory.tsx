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
      phase: "4. Model Optimization",
      title: "Comparing Classical ML & Deep Learning Models",
      icon: "🔬",
      desc: "Compared Logistic Regression, LinearSVC, KNN, ANN/MLP and Text CNN to select the strongest production model."
    },
    {
      phase: "5. Advanced ML",
      title: "Testing ANN, KNN & Text CNN",
      icon: "🧠",
      desc: "Extended the experiment with neural networks, KNN + SVD and a real TensorFlow/Keras Text CNN trained in Google Colab."
    },
    {
      phase: "6. Explainability",
      title: "Explaining Model Decisions",
      icon: "💬",
      desc: "Identified positive and negative word drivers behind customer satisfaction predictions."
    },
    {
      phase: "7. Customer Segments",
      title: "Unsupervised K-Means Clustering",
      icon: "👥",
      desc: "Reduced dimensions using SVD and profiled 5 cohorts with post-hoc satisfaction rates."
    },
    {
      phase: "8. Serving API",
      title: "Exposing ML Model Endpoints",
      icon: "⚡",
      desc: "Constructed FastAPI services with lazy model loading and robust payload validations."
    },
    {
      phase: "9. Skincare UI",
      title: "Interactive Web Dashboard",
      icon: "🌸",
      desc: "Implemented a responsive, premium React dashboard showcasing predictions and segmentation."
    },
    {
      phase: "10. Live Deployment",
      title: "Deploying a Full-Stack ML Product",
      icon: "🚀",
      desc: "Published the FastAPI backend on Render and the React frontend on Vercel for live prediction."
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
