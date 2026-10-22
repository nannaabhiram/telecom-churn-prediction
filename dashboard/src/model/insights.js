/**
 * Turns a scored profile into plain-English explanations and retention levers.
 * Every number shown is derived from the exported model coefficients.
 */
import { MODEL, engine, levelOddsRatio } from './index.js';
import { hasInternet, toModelFeatures } from './profile.js';

// Effects smaller than this (in log-odds) are not worth a bullet point.
const MIN_EFFECT = 0.1;

const formatMultiplier = (ratio) => `${ratio < 1 ? ratio.toFixed(2) : ratio.toFixed(1)}×`;
const formatOddsChange = (ratio) => {
  const pct = Math.round((ratio - 1) * 100);
  return `${pct >= 0 ? '+' : '−'}${Math.abs(pct)}%`;
};
const formatMoney = (value) => `$${value.toFixed(2)}`;

/**
 * Build the list of reasons behind this profile's score.
 * Returns items of shape { id, tone: 'risk' | 'protect' | 'action', icon, text, weight }.
 */
export function buildInsights(profile, result) {
  const { contributions } = result;
  const insights = [];
  const add = (id, tone, icon, text, weight) => insights.push({ id, tone, icon, text, weight });

  // Contract
  if (profile.contract === 'Month-to-month') {
    const vsTwoYear = 1 / levelOddsRatio('Contract', 'Two year');
    const vsOneYear = 1 / levelOddsRatio('Contract', 'One year');
    add(
      'contract',
      'risk',
      '⚠️',
      `Month-to-month contract: ${formatMultiplier(vsTwoYear)} the churn odds of a two-year commitment (${formatMultiplier(vsOneYear)} vs. one-year).`,
      Math.log(vsTwoYear),
    );
  } else {
    const ratio = levelOddsRatio('Contract', profile.contract);
    add(
      'contract',
      'protect',
      '✅',
      `${profile.contract.replace(' ', '-')} contract: ${formatOddsChange(ratio)} churn odds vs. month-to-month.`,
      Math.abs(Math.log(ratio)),
    );
  }

  // Internet service
  if (profile.internetService === 'Fiber optic') {
    const ratio = levelOddsRatio('InternetService', 'Fiber optic');
    add(
      'internet',
      'risk',
      '🚨',
      `Fiber-optic service detected: ${formatOddsChange(ratio)} churn odds vs. DSL (${formatMultiplier(ratio)}).`,
      Math.log(ratio),
    );
  } else if (profile.internetService === 'No') {
    const ratio = levelOddsRatio('InternetService', 'No');
    add('internet', 'protect', '✅', `No internet service: ${formatOddsChange(ratio)} churn odds vs. DSL.`, Math.abs(Math.log(ratio)));
  }

  // Payment method
  const paymentRatio = levelOddsRatio('PaymentMethod', profile.paymentMethod);
  if (profile.paymentMethod === 'Electronic check') {
    add(
      'payment',
      'risk',
      '⚠️',
      `Electronic check payment: ${formatOddsChange(paymentRatio)} churn odds vs. automatic bank transfer.`,
      Math.log(paymentRatio),
    );
  } else if (profile.paymentMethod.includes('automatic')) {
    const echeckRatio = levelOddsRatio('PaymentMethod', 'Electronic check') / paymentRatio;
    add(
      'payment',
      'protect',
      '✅',
      `Automatic payment: avoids the ${formatOddsChange(echeckRatio)} churn odds seen with electronic checks.`,
      Math.log(echeckRatio),
    );
  }

  // Tenure (relative to the average training customer)
  const tenureEffect = contributions.tenure;
  const avgTenure = Math.round(MODEL.numeric.tenure.mean);
  if (Math.abs(tenureEffect) >= MIN_EFFECT) {
    const ratio = Math.exp(tenureEffect);
    const relation = profile.tenure < avgTenure ? 'below' : 'above';
    add(
      'tenure',
      tenureEffect > 0 ? 'risk' : 'protect',
      tenureEffect > 0 ? '⏳' : '✅',
      `Tenure of ${profile.tenure} months is ${relation} the ${avgTenure}-month average: churn odds ${formatMultiplier(ratio)}.`,
      Math.abs(tenureEffect),
    );
  }

  // Monthly charges (a conditional effect: services are already accounted for)
  const chargesEffect = contributions.MonthlyCharges;
  if (Math.abs(chargesEffect) >= MIN_EFFECT) {
    const ratio = Math.exp(chargesEffect);
    add(
      'charges',
      chargesEffect > 0 ? 'risk' : 'protect',
      chargesEffect > 0 ? '💲' : '✅',
      `Monthly charges of ${formatMoney(profile.monthlyCharges)} vs. the ${formatMoney(MODEL.numeric.MonthlyCharges.mean)} average: churn odds ${formatMultiplier(ratio)}, after accounting for the services subscribed.`,
      Math.abs(chargesEffect),
    );
  }

  // Add-ons
  const internet = hasInternet(profile);
  const securityRatio = levelOddsRatio('OnlineSecurity', 'Yes');
  if (internet && profile.onlineSecurity) {
    add('security', 'protect', '🛡️', `Online Security subscribed: ${formatOddsChange(securityRatio)} churn odds.`, Math.abs(Math.log(securityRatio)));
  } else if (internet) {
    add('security', 'action', '💡', `No Online Security: customers with it show ${formatOddsChange(securityRatio)} churn odds.`, Math.abs(Math.log(securityRatio)));
  }

  for (const [key, feature, label] of [
    ['streamingTV', 'StreamingTV', 'Streaming TV'],
    ['streamingMovies', 'StreamingMovies', 'Streaming Movies'],
  ]) {
    if (internet && profile[key]) {
      const ratio = levelOddsRatio(feature, 'Yes');
      add(key, 'risk', '📺', `${label} subscriber: ${formatOddsChange(ratio)} churn odds.`, Math.log(ratio));
    }
  }

  if (profile.paperlessBilling) {
    const ratio = levelOddsRatio('PaperlessBilling', 'Yes');
    add('paperless', 'risk', '📧', `Paperless billing: ${formatOddsChange(ratio)} churn odds vs. paper statements.`, Math.log(ratio));
  }

  const toneOrder = { risk: 0, protect: 1, action: 2 };
  return insights.sort((a, b) => toneOrder[a.tone] - toneOrder[b.tone] || b.weight - a.weight);
}

/**
 * Single-change "what if" scenarios a retention team could act on, scored by
 * the model. Only changes that lower churn probability are returned.
 */
export function buildRetentionLevers(profile, currentProbability) {
  const scenarios = [];

  if (profile.contract === 'Month-to-month') {
    scenarios.push({ id: 'one-year', label: 'Upgrade to a one-year contract', patch: { contract: 'One year' } });
    scenarios.push({ id: 'two-year', label: 'Upgrade to a two-year contract', patch: { contract: 'Two year' } });
  } else if (profile.contract === 'One year') {
    scenarios.push({ id: 'two-year', label: 'Upgrade to a two-year contract', patch: { contract: 'Two year' } });
  }

  if (!profile.paymentMethod.includes('automatic')) {
    scenarios.push({
      id: 'autopay',
      label: 'Switch to automatic credit-card payment',
      patch: { paymentMethod: 'Credit card (automatic)' },
    });
  }

  if (hasInternet(profile) && !profile.onlineSecurity) {
    scenarios.push({ id: 'security', label: 'Bundle in Online Security', patch: { onlineSecurity: true } });
  }

  return scenarios
    .map((scenario) => {
      const { probability } = engine.score(toModelFeatures({ ...profile, ...scenario.patch }));
      return { ...scenario, probability, delta: probability - currentProbability };
    })
    .filter((scenario) => scenario.delta < -0.005)
    .sort((a, b) => a.delta - b.delta);
}
