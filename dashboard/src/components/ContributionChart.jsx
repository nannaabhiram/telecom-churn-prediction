import { useState } from 'react';
import { BASELINE_PROBABILITY, MODEL } from '../model/index.js';
import { HELD_CONSTANT_FEATURES } from '../model/profile.js';

// Fixed symmetric domain (log-odds) so bars stay comparable as inputs change.
// Every single-feature contribution in this model falls inside it.
const DOMAIN = 1.6;
const TICKS = [0.25, 0.5, 1, 2, 4].map((ratio) => ({ ratio, position: Math.log(ratio) }));

const ROWS = [
  { id: 'tenure', label: 'Tenure', features: ['tenure'], describe: (p) => `${p.tenure} months` },
  {
    id: 'charges',
    label: 'Monthly charges',
    features: ['MonthlyCharges'],
    describe: (p) => `$${p.monthlyCharges.toFixed(2)}`,
  },
  { id: 'contract', label: 'Contract', features: ['Contract'], describe: (p) => p.contract },
  {
    id: 'internet',
    label: 'Internet service',
    features: ['InternetService'],
    describe: (p) => (p.internetService === 'No' ? 'No internet' : p.internetService),
  },
  {
    id: 'payment',
    label: 'Payment method',
    features: ['PaymentMethod'],
    describe: (p) => p.paymentMethod.replace(' (automatic)', ''),
  },
  { id: 'security', label: 'Online security', features: ['OnlineSecurity'], describe: (_, f) => f.OnlineSecurity },
  { id: 'tv', label: 'Streaming TV', features: ['StreamingTV'], describe: (_, f) => f.StreamingTV },
  { id: 'movies', label: 'Streaming movies', features: ['StreamingMovies'], describe: (_, f) => f.StreamingMovies },
  { id: 'paperless', label: 'Paperless billing', features: ['PaperlessBilling'], describe: (_, f) => f.PaperlessBilling },
  {
    id: 'other',
    label: 'Other attributes',
    features: HELD_CONSTANT_FEATURES,
    describe: () => `${HELD_CONSTANT_FEATURES.length} held constant`,
  },
];

const toPercent = (logOdds) => ((Math.max(-DOMAIN, Math.min(DOMAIN, logOdds)) + DOMAIN) / (2 * DOMAIN)) * 100;
const formatRatio = (ratio) => `×${ratio < 1 ? ratio.toFixed(2) : ratio.toFixed(1)}`;
const formatSigned = (value) => `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(3)}`;

export default function ContributionChart({ profile, features, result }) {
  const [activeRow, setActiveRow] = useState(null);
  const [showTable, setShowTable] = useState(false);

  const rows = ROWS.map((row) => {
    const logOdds = row.features.reduce((sum, name) => sum + result.contributions[name], 0);
    return { ...row, value: row.describe(profile, features), logOdds, ratio: Math.exp(logOdds) };
  });

  return (
    <section className="card" aria-labelledby="drivers-title">
      <header className="card-head card-head-row">
        <div>
          <p className="eyebrow">Explainability</p>
          <h2 id="drivers-title">What is driving this score</h2>
        </div>
        <button type="button" className="text-button" onClick={() => setShowTable((v) => !v)}>
          {showTable ? 'View as chart' : 'View as table'}
        </button>
      </header>

      <p className="chart-caption">
        Each bar is that input&apos;s push on the churn odds, relative to the model baseline of{' '}
        {(BASELINE_PROBABILITY * 100).toFixed(1)}% (intercept {MODEL.intercept.toFixed(3)}: DSL, month-to-month,
        bank transfer, average tenure and charges, no add-ons).
      </p>

      {showTable ? (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Factor</th>
                <th scope="col">Value</th>
                <th scope="col" className="num">Log-odds</th>
                <th scope="col" className="num">Odds ×</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id}>
                  <th scope="row">{row.label}</th>
                  <td>{row.value}</td>
                  <td className="num">{formatSigned(row.logOdds)}</td>
                  <td className="num">{formatRatio(row.ratio)}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <th scope="row">Total</th>
                <td>intercept {MODEL.intercept.toFixed(3)}</td>
                <td className="num">{formatSigned(result.logOdds)}</td>
                <td className="num">{(result.probability * 100).toFixed(1)}%</td>
              </tr>
            </tfoot>
          </table>
        </div>
      ) : (
        <div className="chart">
          <div className="legend" aria-hidden="true">
            <span className="legend-item">
              <span className="legend-swatch swatch-increase" />
              Raises churn odds
            </span>
            <span className="legend-item">
              <span className="legend-swatch swatch-decrease" />
              Lowers churn odds
            </span>
          </div>

          <ul className="bars" aria-label="Per-factor contribution to churn odds">
            {rows.map((row) => {
              const isZero = Math.abs(row.logOdds) < 1e-9;
              const increases = row.logOdds > 0;
              const start = toPercent(Math.min(0, row.logOdds));
              const end = toPercent(Math.max(0, row.logOdds));
              const tipLeft = increases ? end : start;
              return (
                <li
                  key={row.id}
                  className={`bar-row${activeRow === row.id ? ' is-active' : ''}`}
                  tabIndex={0}
                  aria-label={`${row.label}: ${row.value}, churn odds ${formatRatio(row.ratio)}`}
                  onMouseEnter={() => setActiveRow(row.id)}
                  onMouseLeave={() => setActiveRow(null)}
                  onFocus={() => setActiveRow(row.id)}
                  onBlur={() => setActiveRow(null)}
                >
                  <span className="bar-label">
                    {row.label}
                    <small>{row.value}</small>
                  </span>
                  <span className="bar-track">
                    {TICKS.map((tick) => (
                      <span
                        key={tick.ratio}
                        className={`gridline${tick.ratio === 1 ? ' is-zero' : ''}`}
                        style={{ left: `${toPercent(tick.position)}%` }}
                      />
                    ))}
                    {!isZero && (
                      <span
                        className={`bar ${increases ? 'bar-increase' : 'bar-decrease'}`}
                        style={{ left: `${start}%`, width: `${end - start}%` }}
                      />
                    )}
                    {activeRow === row.id && (
                      <span className="tooltip" role="tooltip" style={{ left: `${Math.min(80, Math.max(20, tipLeft))}%` }}>
                        <strong>{row.label}</strong>
                        <span>{row.value}</span>
                        <span>
                          Log-odds {formatSigned(row.logOdds)} · Odds {formatRatio(row.ratio)}
                        </span>
                      </span>
                    )}
                  </span>
                  <span className={`bar-value${isZero ? ' is-muted' : ''}`}>
                    {isZero ? 'baseline' : formatRatio(row.ratio)}
                  </span>
                </li>
              );
            })}
          </ul>

          <div className="axis" aria-hidden="true">
            <span />
            <span className="axis-track">
              {TICKS.map((tick) => (
                <span key={tick.ratio} className="axis-tick" style={{ left: `${toPercent(tick.position)}%` }}>
                  ×{tick.ratio}
                </span>
              ))}
            </span>
            <span />
          </div>
          <p className="axis-title">Odds multiplier (log scale)</p>
        </div>
      )}
    </section>
  );
}
