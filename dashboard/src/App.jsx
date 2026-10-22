import { useCallback, useMemo, useState } from 'react';
import ControlPanel from './components/ControlPanel.jsx';
import ContributionChart from './components/ContributionChart.jsx';
import RiskPanel from './components/RiskPanel.jsx';
import { MODEL, THRESHOLD, engine } from './model/index.js';
import { buildInsights, buildRetentionLevers } from './model/insights.js';
import { DEFAULT_PROFILE, toModelFeatures } from './model/profile.js';

const MODEL_METRICS = [
  { label: 'ROC-AUC', value: MODEL.metrics.roc_auc.toFixed(3) },
  { label: 'PR-AUC', value: MODEL.metrics.pr_auc.toFixed(3) },
  { label: 'Recall', value: `${(MODEL.metrics.recall * 100).toFixed(1)}%` },
  { label: 'Precision', value: `${(MODEL.metrics.precision * 100).toFixed(1)}%` },
  { label: 'Threshold', value: THRESHOLD.toFixed(3) },
];

export default function App() {
  const [profile, setProfile] = useState(DEFAULT_PROFILE);

  const updateProfile = useCallback((patch) => setProfile((current) => ({ ...current, ...patch })), []);
  const applyPreset = useCallback((preset) => setProfile(preset), []);

  const features = useMemo(() => toModelFeatures(profile), [profile]);
  const result = useMemo(() => engine.score(features), [features]);
  const insights = useMemo(() => buildInsights(profile, result), [profile, result]);
  const levers = useMemo(() => buildRetentionLevers(profile, result.probability), [profile, result]);

  return (
    <div className="app">
      <header className="masthead">
        <div className="masthead-text">
          <p className="eyebrow">Machine learning portfolio</p>
          <h1>Telecom Customer Churn Prediction</h1>
          <p className="lede">
            A Logistic Regression model trained on {(MODEL.n_train + MODEL.n_test).toLocaleString()} IBM Telco
            customers, scored live in your browser. Adjust a customer profile and watch the churn risk update.
          </p>
        </div>
        <dl className="metrics" aria-label={`Held-out test set performance (${MODEL.n_test.toLocaleString()} customers)`}>
          {MODEL_METRICS.map((metric) => (
            <div key={metric.label} className="metric">
              <dt>{metric.label}</dt>
              <dd>{metric.value}</dd>
            </div>
          ))}
        </dl>
      </header>

      <main className="layout">
        <ControlPanel profile={profile} onChange={updateProfile} onPreset={applyPreset} />
        <div className="results-column">
          <RiskPanel result={result} insights={insights} levers={levers} />
          <ContributionChart profile={profile} features={features} result={result} />
        </div>
      </main>

      <footer className="site-footer">
        <p>
          Metrics are from a stratified 20% held-out test set ({MODEL.n_test.toLocaleString()} customers). The
          {' '}{THRESHOLD.toFixed(3)} cut-off maximises F1 on 5-fold out-of-fold training predictions. Odds ratios are
          associations learned from historical data, not causal effects.
        </p>
        <p>
          All scoring runs client-side: coefficients, scaler statistics and reference levels are exported from the
          trained scikit-learn pipeline and verified to match its predictions.
        </p>
      </footer>
    </div>
  );
}
