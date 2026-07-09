# Expected Metrics ML — Optuna + MLflow xG Modeling System

A production-grade expected-goals (xG) modeling stack: leak-safe **Scikit-Learn** feature engineering, **XGBoost** classification, **Optuna** hyperparameter search, and full experiment lineage in **MLflow** — metrics, model artifacts, feature-importance matrices, and calibration curves per run.

## Overview

Expected-metrics models live or die on two things: features that don't leak the future, and probabilities that are actually calibrated. This repository treats both as first-class engineering concerns. All feature logic is implemented as composable `TransformerMixin` steps refit inside every cross-validation fold (so target encodings and rolling form never leak across folds), and evaluation reports calibration by decile plus Brier skill score against the naive base-rate forecaster — the number that tells a stakeholder whether the model beats "just guess the league conversion rate."

## Architecture

```
 ┌───────────────┐    ┌─────────────────────────────────────────────────┐
 │ Raw shot data  │──▶ │ Feature Pipeline (sklearn)                       │
 │ (x, y, context)│    │  ShotGeometry ─▶ RollingForm ─▶ TargetEncoder    │
 └───────────────┘    │  dist/angle      shift(1)+roll    Bayesian-smooth │
                      └───────────────┬─────────────────────────────────┘
                                      ▼
                      ┌─────────────────────────────────┐    ┌───────────────────────────┐
                      │ XGBClassifier                    │◀──▶│ Optuna TPE study           │
                      │ (binary:logistic, hist)          │    │ 7-dim search space,        │
                      └───────────────┬─────────────────┘    │ stratified 4-fold logloss  │
                                      ▼                      └────────────┬──────────────┘
                      ┌─────────────────────────────────────────────────┐ │ every trial
                      │ MLflow (sqlite:///mlflow.db)                     │◀┘
                      │ params · cv_logloss · final ROC-AUC/Brier/LL     │
                      │ model artifact · feature_importance.csv          │
                      │ calibration_curve.png                            │
                      └─────────────────────────────────────────────────┘
                                      ▼
                      ┌─────────────────────────────────┐
                      │ evaluate.py ─▶ markdown report   │
                      │ Brier skill · decile calibration │
                      └─────────────────────────────────┘
```

| Module | Responsibility |
|---|---|
| `src/features.py` | `ShotGeometryTransformer` (distance, visible goal angle, lateral offset), `RollingFormTransformer` (leak-safe shooter form via `shift(1).rolling(...)`), `SmoothedTargetEncoder` (Bayesian-smoothed category encoding with unseen-category fallback), pipeline factory |
| `src/train.py` | Reproducible synthetic shot generator, Optuna TPE study with per-trial nested MLflow runs, final refit logging model + importance matrix + calibration curve |
| `src/evaluate.py` | Brier score, Brier skill score, ROC-AUC, log loss, decile calibration table, markdown performance report |

## Business & Analytics Impact

- **Recruitment signal over noise.** A calibrated xG model separates finishing luck from chance quality: a striker outperforming xG for three seasons is a skill signal; for three matches, it's variance. That distinction is the difference between a good and a catastrophic transfer fee.
- **Decisions you can audit.** Every candidate model's hyperparameters, cross-validated loss, importance matrix, and calibration curve are versioned in MLflow — when a coach asks "why did the model change its mind," the lineage answers.
- **Calibration is the product.** Aggregated season xG totals (the numbers that reach dashboards and broadcasts) are only meaningful if per-shot probabilities are calibrated; the decile table and Brier skill score make that measurable and monitorable, not assumed.
- **Leak-safe by construction.** Rolling form and target encodings are recomputed inside each CV fold, so offline metrics predict production behavior instead of flattering it — protecting analytics credibility with coaching staff.

## Repository Layout

```
expected-metrics-ml/
├── src/
│   ├── features.py      # leak-safe custom transformers + pipeline factory
│   ├── train.py         # Optuna study + MLflow logging + final artifacts
│   └── evaluate.py      # metrics suite + markdown reporting
├── tests/               # transformer, metrics, and end-to-end study tests
├── .github/workflows/ci.yml
├── Dockerfile           # multi-stage, non-root, libgomp runtime
└── requirements*.txt
```

## Notes

- Training data is synthesized with a known probability structure (distance, angle, defender pressure, shooter skill) so runs are fully reproducible and the model has genuine signal to recover; swap `generate_synthetic_shots` for your event-data loader in production.
- MLflow defaults to `sqlite:///mlflow.db` because the plain filesystem store is deprecated in recent MLflow 3.x releases.
