# Datasheet: Ames Housing Dataset (House Prices — Advanced Regression Techniques)

This datasheet documents the dataset used in the **Capstone BBO Stage 1** project (May Zune, Imperial College Business School, 2026). It follows the structure proposed in *Datasheets for Datasets* (Gebru et al., 2018) and describes `train.csv` / `test.csv` as used by the notebooks in this repository. Where the original data-collection process is not documented in the repo, this is stated explicitly rather than assumed.

---

## 1. Motivation

**For what purpose was the dataset created?**
The dataset was assembled to give students and practitioners a richer, more realistic alternative to the small Boston Housing dataset for practicing regression techniques. In this repository it is used as practice material for the Capstone BBO Stage 1 project — building the modelling skills and habits needed for a later black-box optimisation (BBO) challenge (Stage 2).

**Who created the dataset and on behalf of which entity?**
The underlying data was compiled by Dean De Cock for use in data science education, and is distributed on Kaggle as the *House Prices — Advanced Regression Techniques* competition dataset. `data-description.txt` in this repo is described as "originally prepared by Dean De Cock but lightly edited to match the column names used here."

**Who funded the creation of the dataset?**
Not documented in this repository.

---

## 2. Composition

**What do the instances represent?**
Each row (instance) represents a single residential property sale in Ames, Iowa, described by 79 explanatory variables covering almost every aspect of the property (dwelling type, zoning, lot characteristics, quality/condition ratings, room counts, garage/basement/porch details, sale conditions, etc.), plus a target column, `SalePrice`, present only in `train.csv`.

**How many instances are there in total?**
- `train.csv`: 1,460 rows × 81 columns (79 features + `Id` + `SalePrice`), confirmed via `df.shape` in the notebooks.
- `test.csv`: 1,459 rows × 80 columns (79 features + `Id`, no `SalePrice`).
- A `sample_submission.csv` benchmark file (linear regression on year/month of sale, lot square footage and bedroom count) is also referenced by the README as part of the data bundle, though its contents are not explored in the notebooks.

**Does the dataset contain all possible instances, or a sample?**
A sample. It covers residential property sales in Ames, Iowa between 2006 and 2010 (per `YrSold`/`MoSold`); it is not a complete record of all property sales in that period or region.

**What data does each instance consist of?**
Raw, human/machine-readable tabular fields: a mix of
- **Numeric** — e.g. `LotArea`, `GrLivArea`, `TotalBsmtSF`, `GarageArea`, `YearBuilt`, `MoSold`, `YrSold` (35 `int64` + 3 `float64` columns per `df.dtypes.value_counts()`).
- **Categorical (nominal)** — e.g. `MSZoning`, `Neighborhood`, `BldgType`, `SaleType`, `SaleCondition` (43 `object`/string columns).
- **Ordinal, encoded as text** — e.g. `ExterQual`, `KitchenQual`, `BsmtExposure`, `GarageFinish`, `Fence`, using scales such as `Ex/Gd/TA/Fa/Po` or `NA` for "feature absent."

Full field-by-field definitions and category codes are in `data-description.txt` (81 fields).

**Is there a label or target associated with each instance?**
Yes — `SalePrice` (USD), present in `train.csv` only. `test.csv` is unlabelled; predictions for it are what the notebooks/models produce.

**Is any information missing from individual instances?**
Yes, extensively. Per the XGB notebook, the raw training data contains **7,829 missing cells** across the 80 feature columns (`df.isnull().sum().sum()`, before the `Id`/target columns are dropped). Missingness falls into two distinct categories that the notebooks treat differently:
- *Feature genuinely absent* (e.g. `PoolQC`, `Alley`, `FireplaceQu`, `Fence`, `MiscFeature`, and basement/garage quality fields) — `NA` here means "no pool," "no alley access," "no fireplace," etc., not an unknown value.
- *True unknowns* (e.g. `LotFrontage`, `MasVnrArea`, `GarageYrBlt`, `Electrical`) — the value exists in reality but was not recorded.

**Are relationships between individual instances made explicit?**
No explicit relational structure (e.g. multiple sales of the same property, or sales by the same agent) is provided or explored.

**Are there recommended data splits?**
The repo uses the Kaggle-provided `train.csv` / `test.csv` split. Within `train.csv`, the notebooks further hold out a validation split via `train_test_split` (test sizes of 10%–20% across different notebooks, `random_state` 8 or 42) and 5-fold `KFold` cross-validation (`random_state=42`).

**Are there any errors, sources of noise, or redundancies?**
- The EDA notebook engineers an `Area` column (`1stFlrSF + 2ndFlrSF + GrLivArea`) that is redundant by construction, since `GrLivArea` is itself approximately `1stFlrSF + 2ndFlrSF` (+ any finished area above the second floor) — this is flagged as a derived, not raw, feature.
- The XGB notebook's own multicollinearity check (|r| > 0.9) surfaces several near-duplicate signals, e.g. `GarageQual`/`GarageCond`/`GarageYrBlt`/`has_garage` and `PoolQC`/`PoolArea`, which it addresses by compressing the garage fields into a single `pca_garage` component via PCA.
- No duplicate rows were found (`df.duplicated().sum() == 0`).

**Is the dataset self-contained, or does it link to external resources?**
Self-contained as distributed; it is a static extract hosted on Kaggle (linked in the readme-capstone-stage-1) with no live external dependencies.

**Does the dataset contain confidential data?**
No individual/personal data (names, contact details) is present. Data is at the property-sale level (location generalised to `Neighborhood` within Ames, Iowa), not individual-owner level.

**Does the dataset contain data that might be offensive, insulting, threatening, or anxiety-inducing?**
No.

---

## 3. Collection Process

**How was the data collected?**
Not documented in this repository. Publicly, the dataset is understood to be derived from residential property assessment records for Ames, Iowa (2006–2010), compiled by Dean De Cock for teaching purposes; this repo does not independently verify or describe that process.

**Who was involved in the collection, and were they compensated?**
Not documented in this repository.

**Over what timeframe was the data collected?**
Sales recorded span `YrSold` 2006–2010 based on the `MoSold`/`YrSold` fields explored in the EDA notebook. The date of compilation/publication on Kaggle is not stated in the repo.

**Were any ethical review processes conducted?**
Not documented in this repository. Given the property-level, non-personal nature of the data, none is described as necessary.

---

## 4. Preprocessing / Cleaning / Labeling

Preprocessing is **not applied to the source files**; `train.csv` / `test.csv` in the repo are the raw Kaggle files. All cleaning happens inside the notebooks and differs by notebook:

- **`rf_capstone-stage1`**: fills categorical "genuinely absent" columns with the string `'None'`; fills all other categorical NaNs with the column mode; fills remaining numeric NaNs (e.g. `LotFrontage`, `MasVnrArea`, basement/garage numeric fields) via a custom `FillWithKNN` function (`KNeighborsRegressor`, `k=10`) fit on other numeric columns; target-mean-encodes all categorical columns (each category mapped to its rank when sorted by mean `SalePrice`); scales all features with `StandardScaler`.
- **`xgb_capstone-stage1`**: fills `MasVnrArea` with 0; creates a `has_garage` flag and fills `GarageYrBlt` with `-999` where missing; fills `LotFrontage` with the mean for the matching `Street` group; fills "genuinely absent" categoricals with `'None'`; fills `Electrical` with its mode; applies **ordinal encoding** to quality/condition and shape/exposure/slope fields using explicit rank mappings (e.g. `Ex=5…Po=1`, `None=0`); label-encodes remaining categoricals; compresses four collinear garage columns into one `pca_garage` component via `StandardScaler` + `PCA(n_components=1)`. A later cell reimplements this pipeline as reusable `NATreatment` and `ScaleAndEncode` scikit-learn transformers (fit on training data only, to avoid leakage) wrapped in an sklearn `Pipeline` with the model.

**Is the raw data also available in addition to the "cleaned" data?**
Yes — the original, unmodified `train.csv` / `test.csv` are read directly from disk in every notebook; no separately saved "cleaned" dataset file is produced by this repo (feature engineering happens in-memory, per notebook run).

**Is the software used to preprocess/clean/label the instances available?**
Yes, within the four notebooks in this repository (pandas/numpy/scikit-learn code).

---

## 5. Uses

**Has the dataset been used for any tasks already?**
Yes — regression (predicting `SalePrice`) via `RandomForestRegressor`, `KNeighborsRegressor` (for imputation only), `DecisionTreeRegressor`, `LinearRegression` (with polynomial features and K-Fold CV), and `XGBRegressor`, all documented in this repo's notebooks. See `model-card-stage-1.md` for details and results.

**Is there anything about the composition or collection process that might impact future uses?**
- The data is specific to Ames, Iowa, sales from 2006–2010; models trained on it should not be assumed to generalise to other locations, property types, or time periods (e.g. different market conditions, inflation, or regional pricing dynamics).
- `SalePrice` is right-skewed (train mean ≈ $180,921, median ≈ $163,000, max $755,000; `std` ≈ $79,443), which the notebooks note but do not log-transform — this can bias regression models such as the ones used here toward underestimating high-value homes (observed directly in both model notebooks — see Model Card, "Known Limitations").
- Several fields mix "genuinely not applicable" and "missing/unknown" semantics under the same NaN representation in the raw CSV; treating them identically without domain logic (as a naive `dropna` or blanket imputation would) would discard real information.

**Are there tasks for which the dataset should not be used?**
The dataset should not be used to make claims about housing markets, valuations, or lending/underwriting decisions outside Ames, Iowa, nor to represent a current market (data ends in 2010). It should not be used as a basis for real-world property appraisal or lending decisions without substantial additional validation, given its age, narrow geography, and the sample size relative to the number of feature columns (79 features vs. 1,460 training rows).

---

## 6. Distribution

**How is the dataset distributed?**
Via Kaggle's *House Prices — Advanced Regression Techniques* competition page (linked in `readme-capstone-stage-1.md`); the repo stores local copies referenced as `train.csv`, `test.csv`, and `sample_submission.csv` (not included among the uploaded files reviewed for this datasheet).

**Will the dataset be distributed under a copyright or IP license, and are there any restrictions?**
Governed by Kaggle's competition terms; not restated in this repository.

---

## 7. Maintenance

**Who maintains the dataset?**
The Kaggle competition dataset is maintained by Kaggle / the original compiler (Dean De Cock). This repository (notebooks, readme-capstone-stage-1, this datasheet) is maintained by May Zune (Imperial College Business School, Capstone BBO Stage 1, Modules 3–9).

**Will the dataset be updated?**
Not documented; treat as a static snapshot for this project.

