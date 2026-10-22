# Telecom Customer Churn Prediction

An end-to-end churn prediction project: a leakage-safe scikit-learn **Logistic Regression** pipeline, plus an
interactive **React dashboard** that scores customers live in the browser using the trained model's coefficients.

**Live demo:** https://nannaabhiram.github.io/telecom-churn-prediction/

## Results

Held-out test set: 1,409 customers, stratified 20% split.

| Metric | Score |
|---|---|
| ROC-AUC | **0.839** |
| PR-AUC | **0.628** (a random guess scores 0.265) |
| Churn recall | **74.6%** |
| Churn precision | **52.4%** |
| Churn F1 | **0.616** |
| Decision threshold | **0.546** (F1-optimal on 5-fold out-of-fold training predictions) |

## Key business insights

Odds ratios from the model's coefficients:

| Raises churn odds | Change | Lowers churn odds | Change |
|---|---|---|---|
| Fiber-optic internet (vs. DSL) | +206% | Two-year contract (vs. month-to-month) | −76% |
| Electronic check payment | +49% | No internet service (vs. DSL) | −65% |
| Streaming movies | +48% | Tenure (+1 standard deviation) | −53% |
| Streaming TV | +45% | One-year contract | −51% |
| Paperless billing | +40% | Online security | −29% |

These are associations learned from historical data, not causal effects.

## Repository structure

```
├── backend/
│   ├── churn_pipeline.py                     <- Python ML pipeline (EDA → training → evaluation → insights)
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv  <- IBM Telco Customer Churn dataset
│   └── requirements.txt
├── frontend/                                 <- React + Vite dashboard
│   ├── src/
│   ├── package.json
│   └── README.md
├── .github/workflows/deploy-frontend.yml     <- builds and publishes the dashboard to GitHub Pages
└── README.md
```

## Backend: the ML pipeline

```bash
pip install -r backend/requirements.txt
python backend/churn_pipeline.py
```

1. **EDA & cleaning:** casts `TotalCharges` from text and sets the 11 blank values to 0 (all are customers with
   zero tenure). Merges the redundant "No internet service" and "No phone service" categories into "No".
2. **Preprocessing:** a `ColumnTransformer` handles imputation, `StandardScaler` and one-hot encoding with
   `drop="first"`. It sits inside one `Pipeline`, so nothing is fitted on test data. `TotalCharges` is dropped
   because it is almost exactly tenure × monthly charges (correlation 1.000).
3. **Training & threshold tuning:** `class_weight="balanced"` handles the class imbalance. The decision threshold
   is tuned for F1 on cross-validated training predictions, never on the test set.
4. **Evaluation:** classification report, confusion matrix, ROC-AUC and PR-AUC on the held-out test set.
5. **Interpretability:** coefficients converted to odds ratios, with the top 5 churn drivers and protectors.

## Frontend: the live dashboard

```bash
cd frontend
npm install
npm run dev       # http://localhost:5173
npm run verify    # checks the JS scoring engine against sklearn's probabilities
```

The pipeline exports the model's parameters to `frontend/src/model/model.json`:

```bash
python backend/churn_pipeline.py --export-model frontend/src/model/model.json
```

The dashboard re-implements `predict_proba` in JavaScript. It matches scikit-learn to within 1e-16 on
test-set customers.

## Deployment (GitHub Pages)

The workflow in `.github/workflows/deploy-frontend.yml` verifies, builds and publishes `frontend/` on every push to
`main` that changes it. To enable it, go to *Settings → Pages → Source* and choose **GitHub Actions**.

## Tech stack

Python · pandas · scikit-learn · React 18 · Vite · GitHub Actions

Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
