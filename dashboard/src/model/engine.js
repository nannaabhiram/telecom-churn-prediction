/**
 * Logistic Regression scoring engine.
 *
 * Re-implements sklearn's Pipeline.predict_proba from the parameters exported
 * by churn_pipeline.py (--export-model), so scoring runs entirely in the browser:
 *
 *   log_odds    = intercept
 *               + sum(coef * (x - mean) / std)   for scaled numeric features
 *               + sum(coef * x)                  for 0/1 binary features
 *               + sum(coef[level])               for one-hot categorical features
 *                                                (the reference level has coef 0)
 *   probability = 1 / (1 + exp(-log_odds))
 */

export const sigmoid = (logOdds) => 1 / (1 + Math.exp(-logOdds));

export function createEngine(model) {
  const { intercept, numeric, binary, categorical } = model;

  /** Log-odds contribution of every model feature for a complete feature record. */
  function contributions(features) {
    const parts = {};

    for (const [name, { coef, mean, scale }] of Object.entries(numeric)) {
      const value = features[name];
      if (!Number.isFinite(value)) throw new Error(`Missing numeric feature "${name}"`);
      parts[name] = coef * ((value - mean) / scale);
    }

    for (const [name, { coef }] of Object.entries(binary)) {
      parts[name] = coef * Number(features[name]);
    }

    for (const [name, { levels }] of Object.entries(categorical)) {
      const level = features[name];
      if (!(level in levels)) throw new Error(`Unknown level "${level}" for feature "${name}"`);
      parts[name] = levels[level];
    }

    return parts;
  }

  /** Score a complete feature record: log-odds, probability, and per-feature contributions. */
  function score(features) {
    const parts = contributions(features);
    const logOdds = Object.values(parts).reduce((sum, part) => sum + part, intercept);
    return { logOdds, probability: sigmoid(logOdds), contributions: parts };
  }

  return { score };
}
