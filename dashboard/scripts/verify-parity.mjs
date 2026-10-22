/**
 * Checks that the browser scoring engine reproduces scikit-learn's
 * predict_proba on the test-set rows exported with the model.
 *
 *   npm run verify
 */
import { readFileSync } from 'node:fs';
import { createEngine } from '../src/model/engine.js';

const TOLERANCE = 1e-9;

const model = JSON.parse(readFileSync(new URL('../src/model/model.json', import.meta.url), 'utf-8'));
const engine = createEngine(model);

let worstError = 0;
for (const [index, { features, probability }] of model.parity_cases.entries()) {
  const error = Math.abs(engine.score(features).probability - probability);
  worstError = Math.max(worstError, error);
  if (error > TOLERANCE) {
    console.error(`Case ${index}: sklearn ${probability} vs. engine ${engine.score(features).probability}`);
  }
}

const passed = worstError <= TOLERANCE;
console.log(
  `${passed ? 'PASS' : 'FAIL'}: ${model.parity_cases.length} test-set customers, ` +
    `max |sklearn - engine| = ${worstError.toExponential(2)}`,
);
process.exit(passed ? 0 : 1);
