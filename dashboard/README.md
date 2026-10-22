# Churn Risk Dashboard

An interactive React dashboard for the Telco churn Logistic Regression model. It runs entirely in the browser:
there is no backend. The model's intercept, coefficients, scaler statistics and one-hot reference levels are
exported from the trained scikit-learn pipeline into [`src/model/model.json`](src/model/model.json), and
[`src/model/engine.js`](src/model/engine.js) re-implements `predict_proba`:

```
log_odds    = intercept + Σ coef·(x − mean)/std  (tenure, monthly charges)
                        + Σ coef·x               (senior citizen flag)
                        + Σ coef[level]          (one-hot categories; reference level = 0)
probability = 1 / (1 + e^(−log_odds))
```

A customer is flagged **HIGH RISK** when `probability ≥ 0.546`. That cut-off maximised F1 on 5-fold out-of-fold
predictions during training.

## Run locally

```bash
npm install
npm run dev       # http://localhost:5173
npm run verify    # checks the JS engine against sklearn's probabilities on 25 test-set customers
npm run build     # production build in dist/
```

## Retrain / refresh the model

From the project root:

```bash
python churn_pipeline.py --export-model dashboard/src/model/model.json
cd dashboard && npm run verify
```

## Deploy

**Vercel:** import the repository, set *Root Directory* to `dashboard`. Vercel detects Vite automatically
(build `npm run build`, output `dist`).

**GitHub Pages:** the workflow in `.github/workflows/deploy-dashboard.yml` builds and publishes on every push
to `main`. In the repository settings, set *Pages → Source* to **GitHub Actions**. Asset paths are relative
(`base: './'` in `vite.config.js`), so the site works under `https://<user>.github.io/<repo>/`.
