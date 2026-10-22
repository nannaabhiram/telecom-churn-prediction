/**
 * Maps the dashboard's form state onto the model's 18 input features.
 *
 * The form exposes the features recruiters care about; the rest are held at
 * the most common value in the training data (exported as typical_profile).
 */
import { MODEL } from './index.js';

export const TENURE_RANGE = { min: 0, max: 72, step: 1 };
export const MONTHLY_CHARGES_RANGE = { min: 18, max: 120, step: 0.5 };

export const CONTRACT_OPTIONS = [
  { value: 'Month-to-month', label: 'Month-to-month' },
  { value: 'One year', label: 'One year' },
  { value: 'Two year', label: 'Two year' },
];

export const INTERNET_OPTIONS = [
  { value: 'Fiber optic', label: 'Fiber optic' },
  { value: 'DSL', label: 'DSL' },
  { value: 'No', label: 'No internet' },
];

export const PAYMENT_OPTIONS = [
  { value: 'Electronic check', label: 'Electronic check' },
  { value: 'Credit card (automatic)', label: 'Credit card' },
  { value: 'Bank transfer (automatic)', label: 'Bank transfer' },
  { value: 'Mailed check', label: 'Mailed check' },
];

export const ADDON_OPTIONS = [
  { key: 'onlineSecurity', feature: 'OnlineSecurity', label: 'Online Security', requiresInternet: true },
  { key: 'streamingTV', feature: 'StreamingTV', label: 'Streaming TV', requiresInternet: true },
  { key: 'streamingMovies', feature: 'StreamingMovies', label: 'Streaming Movies', requiresInternet: true },
  { key: 'paperlessBilling', feature: 'PaperlessBilling', label: 'Paperless Billing', requiresInternet: false },
];

export const DEFAULT_PROFILE = {
  tenure: 12,
  monthlyCharges: 65,
  contract: 'Month-to-month',
  internetService: 'Fiber optic',
  paymentMethod: 'Electronic check',
  onlineSecurity: false,
  streamingTV: false,
  streamingMovies: false,
  paperlessBilling: true,
};

export const PRESETS = [
  {
    id: 'new-fiber',
    label: 'New fiber streamer',
    profile: {
      tenure: 3,
      monthlyCharges: 95,
      contract: 'Month-to-month',
      internetService: 'Fiber optic',
      paymentMethod: 'Electronic check',
      onlineSecurity: false,
      streamingTV: true,
      streamingMovies: true,
      paperlessBilling: true,
    },
  },
  {
    id: 'loyal-family',
    label: 'Loyal long-term',
    profile: {
      tenure: 60,
      monthlyCharges: 85,
      contract: 'Two year',
      internetService: 'DSL',
      paymentMethod: 'Credit card (automatic)',
      onlineSecurity: true,
      streamingTV: true,
      streamingMovies: false,
      paperlessBilling: false,
    },
  },
  {
    id: 'phone-only',
    label: 'Phone-only basic',
    profile: {
      tenure: 24,
      monthlyCharges: 20,
      contract: 'One year',
      internetService: 'No',
      paymentMethod: 'Mailed check',
      onlineSecurity: false,
      streamingTV: false,
      streamingMovies: false,
      paperlessBilling: false,
    },
  },
];

// Model features the form controls directly; everything else is held constant.
const CONTROLLED_FEATURES = new Set([
  'tenure',
  'MonthlyCharges',
  'Contract',
  'InternetService',
  'PaymentMethod',
  ...ADDON_OPTIONS.map((addon) => addon.feature),
]);

const INTERNET_DEPENDENT_FEATURES = ['OnlineBackup', 'DeviceProtection', 'TechSupport'];

export const HELD_CONSTANT = Object.fromEntries(
  Object.entries(MODEL.typical_profile).filter(([name]) => !CONTROLLED_FEATURES.has(name)),
);

export const HELD_CONSTANT_FEATURES = Object.keys(HELD_CONSTANT);

export const hasInternet = (profile) => profile.internetService !== 'No';

const yesNo = (flag) => (flag ? 'Yes' : 'No');

/** Convert form state into the complete feature record the model expects. */
export function toModelFeatures(profile) {
  const internet = hasInternet(profile);
  const features = {
    ...HELD_CONSTANT,
    tenure: profile.tenure,
    MonthlyCharges: profile.monthlyCharges,
    Contract: profile.contract,
    InternetService: profile.internetService,
    PaymentMethod: profile.paymentMethod,
  };

  // Mirrors the training-time cleaning: without internet, internet add-ons are "No".
  for (const addon of ADDON_OPTIONS) {
    const enabled = profile[addon.key] && (internet || !addon.requiresInternet);
    features[addon.feature] = yesNo(enabled);
  }
  if (!internet) {
    for (const name of INTERNET_DEPENDENT_FEATURES) features[name] = 'No';
  }
  return features;
}

const HELD_CONSTANT_LABELS = {
  SeniorCitizen: (v) => (v ? 'Senior citizen' : 'Not a senior citizen'),
  gender: (v) => v,
  Partner: (v) => (v === 'Yes' ? 'Has a partner' : 'No partner'),
  Dependents: (v) => (v === 'Yes' ? 'Has dependents' : 'No dependents'),
  PhoneService: (v) => (v === 'Yes' ? 'Phone service' : 'No phone service'),
  MultipleLines: (v) => (v === 'Yes' ? 'Multiple lines' : 'Single line'),
  OnlineBackup: (v) => (v === 'Yes' ? 'Online backup' : 'No online backup'),
  DeviceProtection: (v) => (v === 'Yes' ? 'Device protection' : 'No device protection'),
  TechSupport: (v) => (v === 'Yes' ? 'Tech support' : 'No tech support'),
};

/** Human-readable list of the attributes held at their training-set mode. */
export const HELD_CONSTANT_SUMMARY = HELD_CONSTANT_FEATURES.map((name) =>
  HELD_CONSTANT_LABELS[name] ? HELD_CONSTANT_LABELS[name](HELD_CONSTANT[name]) : `${name}: ${HELD_CONSTANT[name]}`,
);
