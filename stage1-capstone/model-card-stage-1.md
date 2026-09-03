# Model Card: House Price Prediction (Capstone BBO Stage 1)

Follows the structure proposed in *Model Cards for Model Reporting* (Mitchell et al., 2019). This repo trains and compares several regression models to predict `SalePrice` for the Kaggle *House Prices — Advanced Regression Techniques* dataset (see `datasheet-stage-1.md` for full dataset documentation). 

Author: May Zune, Imperial College Business School, 2026.

---

## Model Details

**Overview.** Five regression algorithms are explored across two notebooks:

| Notebook | Models trained | Role |
|---|---|---|
| `mz-house-price-prediction-rf_capstone-stage1_ipynb.ipynb` | `RandomForestRegressor` (two versions), `KNeighborsRegressor` | Primary models: Random Forest. KNN is used only as an imputer for missing numeric values, not as a predictor of `SalePrice`. |
| `mz-house-price-prediction-xgb_capstone-stage1.ipynb` | `LinearRegression` (+ `PolynomialFeatures`), `DecisionTreeRegressor`, `XGBRegressor` | Model comparison/selection, culminating in an `XGBRegressor` deployed inside a scikit-learn `Pipeline`. |

**Selected/"final" model.** `XGBRegressor` was judged the best performer during exploration (highest cross-validated score and R² among all candidates tried) and is the model wrapped in the final `Pipeline` (`NATreatment` → `ScaleAndEncode` → `XGBRegressor`) used to generate test-set predictions.

**Model architecture / key hyperparameters:**

- *Random Forest (final version, `rf_capstone-stage1`):* `RandomForestRegressor(n_estimators=100)` (scikit-learn defaults otherwise), trained on `StandardScaler`-normalised, target-mean-encoded features (79 columns).
- *XGBoost (final version, `xgb_capstone-stage1`):* `XGBRegressor(n_estimators=500, learning_rate=0.1, booster='gbtree', max_depth=6, min_child_weight=1, gamma=0, subsample=1, colsample_bytree=1, reg_alpha=0, reg_lambda=1, random_state=42)`. A hyperparameter sweep over `n_estimators ∈ {5,10,50,100,150,200,500,1000,1200}` (all other parameters fixed as above) was run first to select this configuration; 150–1200 estimators gave near-identical, best-in-sweep scores, and 500 was carried forward into the final pipeline.
- *Decision Tree (comparison only):* grid search over `max_depth ∈ {2,3,4,5,7,None}` × `criterion ∈ {squared_error, friedman_mse, absolute_error, poisson}`; best combination found was `max_depth=7, criterion=squared_error`.
- *Linear Regression (comparison only):* plain `LinearRegression`, and a `PolynomialFeatures` (degree 2, 3) + `LinearRegression` pipeline, evaluated with 5-fold CV.

**Software:** Python, pandas, numpy, scikit-learn, xgboost, matplotlib/seaborn (for EDA). Notebooks were run in a local Jupyter/Windows environment (per stack traces).

**Date:** 2026 (Capstone BBO Stage 1, Modules 3–9).

**License / contact:** MIT

---

## Intended Use

**Primary intended uses.** Educational — a practice exercise in the end-to-end regression workflow (EDA → missing-value treatment → categorical/ordinal encoding → dimensionality reduction → model comparison → cross-validation → prediction) as preparation for a later black-box optimisation (BBO) Stage 2 challenge, per the readme-capstone-stage-1.

**Primary intended users.** The author and instructors/peers reviewing the capstone submission.

**Out-of-scope uses.** Not intended for production use, real-world property valuation, mortgage/lending decisions, or any use where a house-price estimate has legal, financial, or safety consequences. Not validated for any geography or period other than Ames, Iowa, 2006–2010. See `datasheet-stage-1.md`, "Uses," for dataset-level caveats that also constrain the models.

---

## Training Data

Ames Housing training set (`train.csv`, 1,460 rows) as distributed by the Kaggle competition; full details in `datasheet-stage-1.md`. Key preprocessing applied before fitting (differs slightly by notebook — see `datasheet-stage-1.md` §4 for the complete breakdown):
- Missing-value treatment split between "genuinely absent" categoricals (filled with the literal string `'None'`) and true unknowns (filled via KNN imputation, group means, or mode, depending on notebook/field).
- Categorical encoding: target-mean rank encoding (RF notebook) or a mix of explicit ordinal maps + label encoding (XGB notebook).
- Feature reduction: four collinear garage columns (`GarageYrBlt`, `GarageQual`, `GarageCond`, `has_garage`) compressed into a single `pca_garage` component via `StandardScaler` + `PCA(n_components=1)`, explaining ~96.8% of their combined variance (XGB notebook only).
- Final feature counts: 79 predictor columns feed both the Random Forest and XGBoost pipelines (after the reductions above).

## Evaluation Data

- Held-out split of `train.csv` (`train_test_split`, 10–20% held out depending on notebook, `random_state` 8 or 42) used for RMSE/R² checks and 5-fold cross-validation (`KFold(n_splits=5, shuffle=True, random_state=42)`).
- The unlabelled Kaggle `test.csv` (1,459 rows) is used to generate predictions, which are then compared — **not formally scored**, since true labels for `test.csv` are not available — against `sample_submission.csv`, a simple benchmark (linear regression on year/month of sale, lot square footage, and bedroom count) rather than ground truth.

---

## Metrics

| Model | Metric | Value | Source |
|---|---|---|---|
| Random Forest (naive, numeric-only features, no categoricals) | RMSE | $29,588 | `rf_capstone-stage1`, initial pass |
| Random Forest (`n_estimators=100`, full feature set, scaled + encoded) | RMSE (10% hold-out) | $19,399 | `rf_capstone-stage1`, final pass |
| Linear Regression | 5-fold CV R² | 0.777 | `xgb_capstone-stage1` |
| Linear Regression + `PolynomialFeatures` (degree 2) | 5-fold CV R² | 0.130 | `xgb_capstone-stage1` |
| Linear Regression + `PolynomialFeatures` (degree 3) | 5-fold CV R² | −76.0 (diverged) | `xgb_capstone-stage1` |
| Decision Tree (best: `max_depth=7`, `squared_error`) | 5-fold CV score / hold-out R² | 0.741 / 0.799 | `xgb_capstone-stage1` |
| XGBoost (`n_estimators=150`, best in sweep) | 5-fold CV score / hold-out R² / MSE | 0.856 / 0.915 / 6.52×10⁸ | `xgb_capstone-stage1` sweep |
| XGBoost (`n_estimators=500`, carried forward) | 5-fold CV score / hold-out R² / MSE | 0.856 / 0.916 / 6.47×10⁸ | `xgb_capstone-stage1` sweep |
| XGBoost (final pipeline) | Training R² | 1.000 | `xgb_capstone-stage1`, final pipeline — a strong indicator of overfitting, see Limitations |

**RMSE unit:** US dollars, on the original `SalePrice` scale (`std(SalePrice) ≈ $79,443`, so an RMSE of ~$19–30k is roughly 25–37% of the target's standard deviation).

**Decision thresholds:** Not applicable — this is a regression task with no classification threshold.

---

## Known Limitations / Ethical Considerations

- **Systematic underestimation of price.** Both the Random Forest and XGBoost pipelines were observed to underestimate `SalePrice` relative to the benchmark `sample_submission.csv`, and this is called out explicitly in the notebooks and readme-capstone-stage-1. On the RF notebook's `test.csv` predictions, the mean signed difference vs. the benchmark was **−9.3%** (range −195% to +62%); on the XGB notebook's, the mean signed difference was **−15.3%** (range −255% to +67%) — both are compared to a simple benchmark, not ground truth, so these numbers indicate directional disagreement rather than proven error.
- **Training R² of 1.000 for the final XGBoost pipeline** strongly suggests overfitting to the training set; no early stopping, regularization tuning beyond defaults, or held-out validation curve was used for the final fit, so the reported cross-validation R² (0.916) is a more trustworthy estimate of generalisation than the training R².
- **Small sample size relative to feature count.** 1,460 training rows against ~79 features (before further engineering) leaves limited data per feature, especially for rare categories within `Neighborhood`, `Exterior1st/2nd`, `RoofMatl`, etc.
- **Skewed target, not transformed.** `SalePrice` is right-skewed (median $163,000 vs. mean $180,921, max $755,000); none of the notebooks apply a log transform, which the underestimation pattern above is consistent with (high-value homes pulling the loss function without being modelled well).
- **Missing feature-importance / neighborhood effects**, noted explicitly in the readme-capstone-stage-1's own reflection: *"the ignorance of feature importance such as neighborhood, building types, house style and house construction types contribute to the poor prediction results."*
- **Limited hyperparameter tuning.** The readme-capstone-stage-1 notes feature engineering and hyperparameter tuning were limited; the XGBoost sweep varied only `n_estimators`, leaving `max_depth`, `learning_rate`, regularisation terms, etc. at fixed values.
- **Encoding leakage risk (early cells).** The initial (non-pipeline) versions of both the target-mean encoding (RF notebook) and label/ordinal encoding (XGB notebook) are fit using statistics that include information not strictly separated from the evaluation split. The XGB notebook's later `NATreatment`/`ScaleAndEncode` transformer classes explicitly fix this by fitting only on training data, but the earlier exploratory cells in both notebooks do not use this safeguard.
- **Geographic and temporal narrowness.** Ames, Iowa, 2006–2010 only; see `datasheet-stage-1.md` for details. No fairness auditing across neighborhoods or property types was performed, and given the readme-capstone-stage-1's own note about ignoring neighborhood effects, differential error rates across `Neighborhood` are plausible but unmeasured.
- **Not intended for deployment.** This is a learning exercise; no monitoring, versioning, or production serving infrastructure accompanies these models.

---

## Recommendations for Future Work (from the readme-capstone-stage-1's own reflection)

- Incorporate categorical features more directly tied to price drivers (`Neighborhood`, `BldgType`, `HouseStyle`, construction type) that the current models under-use.
- Broaden hyperparameter tuning beyond the single-parameter sweep performed for XGBoost.
- Consider a log-transform of `SalePrice` to address the underestimation pattern and right-skew noted above.
- Formalise the leak-safe `NATreatment`/`ScaleAndEncode` pipeline pattern (already implemented late in the XGB notebook) as the standard preprocessing path, replacing the earlier non-pipeline exploratory cells.
