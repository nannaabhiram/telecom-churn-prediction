import { THRESHOLD } from '../model/index.js';

const formatPercent = (p, digits = 1) => `${(p * 100).toFixed(digits)}%`;

function RiskBadge({ isHighRisk }) {
  return isHighRisk ? (
    <span className="badge badge-critical" role="status">
      <span className="badge-icon" aria-hidden="true">▲</span>
      HIGH RISK (Will Churn)
    </span>
  ) : (
    <span className="badge badge-good" role="status">
      <span className="badge-icon" aria-hidden="true">✓</span>
      LOW RISK (Will Retain)
    </span>
  );
}

function ProbabilityMeter({ probability, isHighRisk }) {
  return (
    <div className="meter-wrap">
      <div
        className={`meter ${isHighRisk ? 'is-critical' : 'is-good'}`}
        role="meter"
        aria-label="Churn probability"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(probability * 1000) / 10}
      >
        <div className="meter-fill" style={{ width: `${probability * 100}%` }} />
        <div className="meter-threshold" style={{ left: `${THRESHOLD * 100}%` }} />
      </div>
      <div className="meter-scale" aria-hidden="true">
        <span>0%</span>
        <span className="meter-threshold-label" style={{ left: `${THRESHOLD * 100}%` }}>
          Threshold {THRESHOLD.toFixed(3)}
        </span>
        <span>100%</span>
      </div>
    </div>
  );
}

function StatTile({ label, value, hint }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {hint && <span className="stat-hint">{hint}</span>}
    </div>
  );
}

export default function RiskPanel({ result, insights, levers }) {
  const { probability, logOdds } = result;
  const isHighRisk = probability >= THRESHOLD;
  const margin = probability - THRESHOLD;
  const odds = probability / (1 - probability);

  return (
    <section className="card panel" aria-labelledby="results-title">
      <header className="card-head">
        <p className="eyebrow">Step 2</p>
        <h2 id="results-title">Live Risk Analysis Results</h2>
      </header>

      <div className="hero">
        <div>
          <p className="hero-value" aria-live="polite">
            {formatPercent(probability)}
          </p>
          <p className="hero-label">Churn Probability</p>
        </div>
        <RiskBadge isHighRisk={isHighRisk} />
      </div>

      <ProbabilityMeter probability={probability} isHighRisk={isHighRisk} />

      <div className="stats">
        <StatTile label="Churn odds" value={`${odds.toFixed(2)} : 1`} hint="p / (1 − p)" />
        <StatTile label="Log-odds" value={logOdds.toFixed(3)} hint="before sigmoid" />
        <StatTile
          label="Margin to threshold"
          value={`${margin >= 0 ? '+' : '−'}${Math.abs(margin * 100).toFixed(1)} pts`}
          hint={isHighRisk ? 'above cut-off' : 'below cut-off'}
        />
      </div>

      <div className="subsection">
        <h3>Why this customer scores {isHighRisk ? 'high' : 'low'}</h3>
        <ul className="insights">
          {insights.map((item) => (
            <li key={item.id} className={`insight insight-${item.tone}`}>
              <span className="insight-icon" aria-hidden="true">
                {item.icon}
              </span>
              <span>{item.text}</span>
            </li>
          ))}
        </ul>
      </div>

      {levers.length > 0 && (
        <div className="subsection">
          <h3>Retention levers</h3>
          <ul className="levers">
            {levers.map((lever) => {
              const flips = probability >= THRESHOLD && lever.probability < THRESHOLD;
              return (
                <li key={lever.id} className="lever">
                  <span className="lever-label">{lever.label}</span>
                  <span className="lever-change">
                    {formatPercent(probability)} → <strong>{formatPercent(lever.probability)}</strong>
                    <span className="lever-delta">({(lever.delta * 100).toFixed(1)} pts)</span>
                  </span>
                  {flips && <span className="lever-flag">Flips to low risk</span>}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </section>
  );
}
