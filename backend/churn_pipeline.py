"""
Telco Customer Churn Prediction: End-to-End Logistic Regression Pipeline
=========================================================================

An interpretable churn model built in five phases:

    1. EDA & data cleaning        - fix data-type anomalies, handle missing values
    2. Feature engineering        - leakage-safe ColumnTransformer preprocessing
    3. Training & threshold tuning - class-balanced model, F1-optimal cut-off chosen
                                     on out-of-fold predictions (never the test set)
    4. Evaluation                 - classification report, confusion matrix, ROC-AUC, PR-AUC
    5. Business insights          - coefficients converted to odds ratios

Usage
-----
    python backend/churn_pipeline.py

    # Also export model parameters for the React frontend:
    python backend/churn_pipeline.py --export-model frontend/src/model/model.json

Dataset: IBM Telco Customer Churn (Kaggle: blastchar/telco-customer-churn).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET_COL = "Churn"
ID_COL = "customerID"

# Continuous features: scaled with StandardScaler.
NUMERIC_FEATURES = ["tenure", "MonthlyCharges"]

# Cleaned, then excluded from the model: TotalCharges is roughly
# tenure x MonthlyCharges, so keeping it splits the same effect across
# overlapping columns and flips coefficient signs (it showed TotalCharges
# raising churn odds while MonthlyCharges lowered them).
REDUNDANT_FEATURES = ["TotalCharges"]

# Already 0/1 in the raw data: passed through unscaled so its odds ratio reads
# as "senior vs. non-senior" rather than "per standard deviation".
BINARY_FEATURES = ["SeniorCitizen"]

# Service add-ons whose "No internet service" level duplicates InternetService == "No".
INTERNET_DEPENDENT_FEATURES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# "string" covers the pandas 3 default text dtype; "object" covers pandas 2.
TEXT_DTYPES = ["object", "string"]

LINE_WIDTH = 72


@dataclass
class ThresholdResult:
    """Outcome of the decision-threshold search."""

    threshold: float
    precision: float
    recall: float
    f1: float


# --------------------------------------------------------------------------- #
# Console formatting helpers
# --------------------------------------------------------------------------- #

def print_header(title: str) -> None:
    print("\n" + "=" * LINE_WIDTH)
    print(f" {title}")
    print("=" * LINE_WIDTH)


def print_subheader(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, LINE_WIDTH - len(title) - 5))


# --------------------------------------------------------------------------- #
# Phase 1: EDA & Data Cleaning
# --------------------------------------------------------------------------- #

def load_data(path: Path) -> pd.DataFrame:
    """Load the raw Telco CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download it from Kaggle "
            "(blastchar/telco-customer-churn) or pass --data <path>."
        )
    return pd.read_csv(path)


def run_eda(df: pd.DataFrame) -> None:
    """Print a short exploratory summary of the raw data."""
    print_header("PHASE 1: EXPLORATORY DATA ANALYSIS & CLEANING")

    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    print(f"Duplicate customer IDs: {df[ID_COL].duplicated().sum()}")

    print_subheader("Column data types")
    print(df.dtypes.value_counts().to_string())

    # TotalCharges arrives as text because some rows hold a blank string.
    blank_counts = df.apply(
        lambda col: col.astype(str).str.strip().eq("").sum()
    )
    blank_counts = blank_counts[blank_counts > 0]
    print_subheader("Blank-string anomalies")
    print(blank_counts.to_string() if not blank_counts.empty else "None found.")

    print_subheader("Target distribution (Churn)")
    churn_dist = df[TARGET_COL].value_counts()
    churn_pct = df[TARGET_COL].value_counts(normalize=True).mul(100)
    for label in churn_dist.index:
        print(f"  {label:<4}: {churn_dist[label]:>5,}  ({churn_pct[label]:.1f}%)")

    print_subheader("Churn rate by contract type")
    churn_flag = df[TARGET_COL].eq("Yes")
    contract_churn = churn_flag.groupby(df["Contract"]).mean().mul(100).sort_values(ascending=False)
    for contract, rate in contract_churn.items():
        print(f"  {contract:<16}: {rate:5.1f}%")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix data-type anomalies, apply deterministic row-level fixes, and encode the target.

    Only row-level rules are applied here. Anything that learns statistics from
    the data (e.g. median imputation) happens inside the sklearn Pipeline, so
    it is fitted on the training split only.
    """
    df = df.copy()

    # 1. Strip whitespace from text columns so " " becomes "".
    text_cols = df.select_dtypes(include=TEXT_DTYPES).columns
    df[text_cols] = df[text_cols].apply(lambda col: col.str.strip())

    # 2. TotalCharges: blank strings -> NaN -> float.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace("", np.nan), errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    print_subheader("Cleaning log")
    print(f"  TotalCharges missing after numeric cast: {n_missing}")

    # 3. Domain rule: customers with tenure 0 have not been billed yet, so their
    #    TotalCharges is 0, not unknown. This is a per-row fact, not a learned
    #    statistic, so it leaks nothing across the train/test split.
    new_customer_mask = df["TotalCharges"].isna() & df["tenure"].eq(0)
    df.loc[new_customer_mask, "TotalCharges"] = 0.0
    print(f"  Set to 0 for brand-new customers (tenure == 0): {new_customer_mask.sum()}")
    print(f"  Still missing: {df['TotalCharges'].isna().sum()}")

    # 4. Collapse redundant levels. "No internet service" in the six add-on
    #    columns is the same information as InternetService == "No". Keeping it
    #    makes the dummies perfectly collinear and the coefficients unreliable.
    for col in INTERNET_DEPENDENT_FEATURES:
        df[col] = df[col].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    print("  Collapsed 'No internet/phone service' levels into 'No' (multicollinearity fix)")

    # 5. Target: Yes/No -> 1/0.
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    if df[TARGET_COL].isna().any():
        raise ValueError("Unexpected values in target column 'Churn'.")
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # 6. Log the evidence for dropping redundant features.
    charges_proxy = df["tenure"] * df["MonthlyCharges"]
    print(
        f"  corr(TotalCharges, tenure)                  = "
        f"{df['TotalCharges'].corr(df['tenure']):.3f}"
    )
    print(
        f"  corr(TotalCharges, tenure x MonthlyCharges) = "
        f"{df['TotalCharges'].corr(charges_proxy):.3f}"
    )
    print(f"  Dropped as redundant (multicollinearity): {REDUNDANT_FEATURES}")

    # 7. Drop the identifier (no signal) and the redundant features.
    df = df.drop(columns=[ID_COL] + REDUNDANT_FEATURES)

    print(f"  Final clean shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# --------------------------------------------------------------------------- #
# Phase 2: Feature Engineering & Preprocessing
# --------------------------------------------------------------------------- #

def get_feature_groups(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Split predictor columns into numeric, binary, and categorical groups."""
    predictors = df.drop(columns=[TARGET_COL])
    categorical = [
        col for col in predictors.select_dtypes(include=TEXT_DTYPES).columns
        if col not in NUMERIC_FEATURES + BINARY_FEATURES
    ]
    return NUMERIC_FEATURES, BINARY_FEATURES, categorical


def build_pipeline(
    numeric_features: list[str],
    binary_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """
    Build the preprocessing + model pipeline.

    Wrapping everything in one Pipeline means the imputer, scaler, and encoder
    are fitted only on the data passed to .fit() (the training split or a CV
    fold), so no test-set information leaks into preprocessing.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # drop="first" removes one level per feature to avoid the dummy-variable
        # trap (perfect multicollinearity with the intercept).
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("bin", "passthrough", binary_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = LogisticRegression(
        class_weight="balanced",  # reweights classes inversely to their frequency
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])


# --------------------------------------------------------------------------- #
# Phase 3: Model Training & Threshold Tuning
# --------------------------------------------------------------------------- #

def find_optimal_threshold(y_true: pd.Series, y_proba: np.ndarray) -> ThresholdResult:
    """Return the probability cut-off that maximises F1 on the given predictions."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # precision/recall have one more element than thresholds; drop the last point.
    precision, recall = precision[:-1], recall[:-1]
    denom = precision + recall
    f1_scores = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)

    best_idx = int(np.argmax(f1_scores))
    return ThresholdResult(
        threshold=float(thresholds[best_idx]),
        precision=float(precision[best_idx]),
        recall=float(recall[best_idx]),
        f1=float(f1_scores[best_idx]),
    )


def tune_threshold(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> ThresholdResult:
    """
    Choose the decision threshold on out-of-fold training predictions.

    Every training row is scored by a model that never saw it (stratified
    k-fold), so the chosen threshold is not overfit to the training data, and
    the test set stays untouched until final evaluation.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    best = find_optimal_threshold(y_train, oof_proba)

    print_subheader(f"Threshold sweep ({CV_FOLDS}-fold out-of-fold, training set)")
    print(f"  {'Threshold':>9} | {'Precision':>9} | {'Recall':>7} | {'F1':>6}")
    print(f"  {'-' * 9}-+-{'-' * 9}-+-{'-' * 7}-+-{'-' * 6}")
    for threshold in np.arange(0.30, 0.80, 0.05):
        preds = (oof_proba >= threshold).astype(int)
        print(
            f"  {threshold:>9.2f} | "
            f"{precision_score(y_train, preds, zero_division=0):>9.3f} | "
            f"{recall_score(y_train, preds):>7.3f} | "
            f"{f1_score(y_train, preds):>6.3f}"
        )

    print(f"\n  CV PR-AUC (average precision): {average_precision_score(y_train, oof_proba):.4f}")
    print(f"  F1-optimal threshold:          {best.threshold:.4f}")
    print(
        f"    -> Precision {best.precision:.3f} | Recall {best.recall:.3f} | F1 {best.f1:.3f}"
    )
    return best


# --------------------------------------------------------------------------- #
# Phase 4: Comprehensive Model Evaluation
# --------------------------------------------------------------------------- #

def print_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> None:
    """Print a labelled 2x2 confusion matrix with business-friendly annotations."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"  {'':>18}{'Pred: Stay':>12}{'Pred: Churn':>13}")
    print(f"  {'Actual: Stay':<18}{tn:>12,}{fp:>13,}")
    print(f"  {'Actual: Churn':<18}{fn:>12,}{tp:>13,}")
    print(f"\n  True Positives  (churners caught):         {tp:,}")
    print(f"  False Negatives (churners missed):         {fn:,}")
    print(f"  False Positives (loyal customers flagged): {fp:,}")
    print(f"  True Negatives  (loyal customers cleared): {tn:,}")


def evaluate_model(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float
) -> dict[str, float]:
    """Evaluate the fitted pipeline on the held-out test set."""
    print_header("PHASE 4: MODEL EVALUATION (HELD-OUT TEST SET)")

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred_tuned = (y_proba >= threshold).astype(int)
    y_pred_default = (y_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print_subheader("Threshold-independent metrics")
    print(f"  ROC-AUC:                       {roc_auc:.4f}")
    print(f"  PR-AUC (average precision):    {pr_auc:.4f}")
    print(f"  Baseline PR-AUC (churn rate):  {y_test.mean():.4f}")

    print_subheader(f"Classification report @ tuned threshold = {threshold:.3f}")
    print(classification_report(
        y_test, y_pred_tuned, target_names=["No Churn (0)", "Churn (1)"], digits=3
    ))

    print_subheader("Confusion matrix @ tuned threshold")
    print_confusion_matrix(y_test, y_pred_tuned)

    print_subheader("Default (0.50) vs. tuned threshold: churn class")
    print(f"  {'Metric':<10} | {'@ 0.50':>8} | {f'@ {threshold:.3f}':>8}")
    print(f"  {'-' * 10}-+-{'-' * 8}-+-{'-' * 8}")
    for name, metric in [("Precision", precision_score), ("Recall", recall_score), ("F1", f1_score)]:
        print(
            f"  {name:<10} | {metric(y_test, y_pred_default):>8.3f} | "
            f"{metric(y_test, y_pred_tuned):>8.3f}"
        )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision_score(y_test, y_pred_tuned),
        "recall": recall_score(y_test, y_pred_tuned),
        "f1": f1_score(y_test, y_pred_tuned),
        "threshold": threshold,
    }


# --------------------------------------------------------------------------- #
# Phase 5: Executive Business Insights & Interpretability
# --------------------------------------------------------------------------- #

def extract_odds_ratios(pipeline: Pipeline) -> pd.DataFrame:
    """Return a table of coefficients and odds ratios, sorted by odds ratio."""
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": classifier.coef_[0],
    })
    coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
    # Percent change in churn odds, e.g. OR 1.8 -> +80%, OR 0.4 -> -60%.
    coef_df["pct_change_in_odds"] = (coef_df["odds_ratio"] - 1) * 100
    return coef_df.sort_values("odds_ratio", ascending=False).reset_index(drop=True)


def print_business_insights(pipeline: Pipeline, top_n: int = 5) -> pd.DataFrame:
    """Print the intercept and the top churn drivers and protectors."""
    print_header("PHASE 5: EXECUTIVE BUSINESS INSIGHTS")

    intercept = pipeline.named_steps["classifier"].intercept_[0]
    print(f"  Model intercept (log-odds): {intercept:+.4f}")
    print(f"  Baseline odds (exp):        {np.exp(intercept):.4f}")
    print(
        "\n  How to read this: numeric features (tenure, monthly charges) are standardised,\n"
        "  so their odds ratio is per +1 standard deviation. Categorical odds ratios\n"
        "  compare against the dropped reference level (e.g. Contract vs. 'Month-to-month')."
    )

    odds_df = extract_odds_ratios(pipeline)

    def _print_table(frame: pd.DataFrame) -> None:
        print(f"  {'Rank':<5}{'Feature':<40}{'Odds Ratio':>11}{'Change in Odds':>16}")
        print(f"  {'-' * 4} {'-' * 39} {'-' * 10} {'-' * 15}")
        for rank, row in enumerate(frame.itertuples(index=False), start=1):
            print(
                f"  {rank:<5}{row.feature:<40}{row.odds_ratio:>11.3f}"
                f"{row.pct_change_in_odds:>+15.1f}%"
            )

    print_subheader(f"Top {top_n} features that INCREASE churn odds")
    _print_table(odds_df.head(top_n))

    print_subheader(f"Top {top_n} features that DECREASE churn odds")
    _print_table(odds_df.sort_values("odds_ratio").head(top_n))

    return odds_df


def print_resume_summary(metrics: dict[str, float], n_train: int, n_test: int) -> None:
    """Print a one-glance summary of the headline results."""
    print_header("RESUME SUMMARY")
    print(f"  Model:            Logistic Regression (class_weight='balanced', L2)")
    print(f"  Data split:       {n_train:,} train / {n_test:,} test (stratified 80/20)")
    print(f"  Decision cut-off: {metrics['threshold']:.3f} (F1-optimal, {CV_FOLDS}-fold CV)")
    print(f"  ROC-AUC:          {metrics['roc_auc']:.3f}")
    print(f"  PR-AUC:           {metrics['pr_auc']:.3f}")
    print(f"  Churn Recall:     {metrics['recall']:.1%}")
    print(f"  Churn Precision:  {metrics['precision']:.1%}")
    print(f"  Churn F1-Score:   {metrics['f1']:.3f}")
    print("=" * LINE_WIDTH)


# --------------------------------------------------------------------------- #
# Model export (consumed by the React frontend)
# --------------------------------------------------------------------------- #

def export_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    metrics: dict[str, float],
    path: Path,
    n_parity_cases: int = 25,
) -> None:
    """
    Write everything needed to score a customer outside Python to JSON.

    Includes the intercept, coefficients, scaler statistics, one-hot reference
    levels, a "typical customer" profile (training means / modes), and a set
    of test-set rows with sklearn's probabilities, so a re-implementation of
    the scoring formula can be checked against the real model.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    coef_by_name = dict(zip(preprocessor.get_feature_names_out(), classifier.coef_[0]))

    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    numeric_features = list(preprocessor.transformers_[0][2])
    binary_features = list(preprocessor.transformers_[1][2])
    categorical_features = list(preprocessor.transformers_[2][2])
    encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]

    numeric = {
        name: {
            "coef": float(coef_by_name[name]),
            "mean": float(scaler.mean_[i]),
            "scale": float(scaler.scale_[i]),
        }
        for i, name in enumerate(numeric_features)
    }
    binary = {name: {"coef": float(coef_by_name[name])} for name in binary_features}

    # drop="first": the first category of each feature is the reference (coef 0).
    categorical = {}
    for name, levels in zip(categorical_features, encoder.categories_):
        reference = str(levels[0])
        categorical[name] = {
            "reference": reference,
            "levels": {reference: 0.0} | {
                str(level): float(coef_by_name[f"{name}_{level}"]) for level in levels[1:]
            },
        }

    typical_profile = (
        {name: float(X_train[name].mean()) for name in numeric_features}
        | {name: int(X_train[name].mode()[0]) for name in binary_features}
        | {name: str(X_train[name].mode()[0]) for name in categorical_features}
    )

    feature_cols = numeric_features + binary_features + categorical_features
    sample = X_test[feature_cols].head(n_parity_cases)
    parity_cases = [
        {"features": row, "probability": float(prob)}
        for row, prob in zip(
            sample.to_dict(orient="records"), pipeline.predict_proba(sample)[:, 1]
        )
    ]

    payload = {
        "model": "LogisticRegression(class_weight='balanced', L2)",
        "intercept": float(classifier.intercept_[0]),
        "threshold": round(metrics["threshold"], 3),
        "numeric": numeric,
        "binary": binary,
        "categorical": categorical,
        "typical_profile": typical_profile,
        "metrics": {
            key: round(float(metrics[key]), 4)
            for key in ("roc_auc", "pr_auc", "precision", "recall", "f1")
        },
        "n_train": len(X_train),
        "n_test": len(X_test),
        "parity_cases": parity_cases,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  Model parameters exported to: {path}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def main(data_path: Path, export_path: Path | None = None) -> None:
    # Phase 1
    raw_df = load_data(data_path)
    run_eda(raw_df)
    df = clean_data(raw_df)

    # Phase 2
    print_header("PHASE 2: FEATURE ENGINEERING & PREPROCESSING")
    numeric_features, binary_features, categorical_features = get_feature_groups(df)
    print(f"  Numeric (scaled)       [{len(numeric_features)}]: {numeric_features}")
    print(f"  Binary (passthrough)   [{len(binary_features)}]: {binary_features}")
    print(f"  Categorical (one-hot)  [{len(categorical_features)}]: {categorical_features}")
    pipeline = build_pipeline(numeric_features, binary_features, categorical_features)

    # Phase 3
    print_header("PHASE 3: MODEL TRAINING & THRESHOLD TUNING")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train):,} rows | churn rate {y_train.mean():.1%}")
    print(f"  Test:  {len(X_test):,} rows | churn rate {y_test.mean():.1%}")

    best_threshold = tune_threshold(pipeline, X_train, y_train)
    pipeline.fit(X_train, y_train)
    n_encoded = len(pipeline.named_steps["preprocessor"].get_feature_names_out())
    print(f"\n  Final model fitted on full training set ({n_encoded} encoded features).")

    # Phase 4
    metrics = evaluate_model(pipeline, X_test, y_test, best_threshold.threshold)

    # Phase 5
    print_business_insights(pipeline)
    print_resume_summary(metrics, len(X_train), len(X_test))

    if export_path is not None:
        export_model(pipeline, X_train, X_test, metrics, export_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telco churn Logistic Regression pipeline.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to the Telco Customer Churn CSV.",
    )
    parser.add_argument(
        "--export-model",
        type=Path,
        default=None,
        help="Optional path to write model parameters as JSON (e.g. frontend/src/model/model.json).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.data, args.export_model)
