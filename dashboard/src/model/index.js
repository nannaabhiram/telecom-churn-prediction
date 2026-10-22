import modelParams from './model.json';
import { createEngine, sigmoid } from './engine.js';

export const MODEL = modelParams;
export const THRESHOLD = modelParams.threshold;
export const BASELINE_PROBABILITY = sigmoid(modelParams.intercept);
export const engine = createEngine(modelParams);

/** Odds ratio of a categorical level relative to that feature's reference level. */
export const levelOddsRatio = (feature, level) =>
  Math.exp(modelParams.categorical[feature].levels[level]);
