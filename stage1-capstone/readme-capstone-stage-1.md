# Capstone BBO Stage 1 Project 
- May Zune | Imperial College Business School 2026
- Repository for Module 3 to Module 9

## BACKGROUND
The aim of the Capstone BBO Stage 1 is to build the skills and habits required for the BBO challenge in Stage 2.

For this experiment, [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) from Kaggle is chosen from Kaggle.

## DATA

- **train.csv** - the training set
- **test.csv** - the test set
- **data-description.txt** - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here
- **sample-submission.csv** - a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms


## NOTEBOOK 
- **mz-house-price-prediction-exploratory-data-analysis-train.ipynb**
    - This notebook contains the exploratory data analysis for the **train.csv** file.
- **mz-house-price-prediction-exploratory-data-analysis-test.ipynb**
    - This notebook contains the exploratory data analysis for the **test.csv** file.
- **mz-house-price-prediction-rf_capstone-stage1.ipynb**
    - This notebook contains the prediction for house prices using `RandomForestRegressor`. 
    - The prediction underestimates the sale prices. 
- **mz-house-price-prediction-xgb_capstone-stage1.ipynb**
    - Ordinal variable treatment is conducted for columns with NAN values.
    - `PCA` and `StandardScaler` are used for preprocessing and dimensionality reduction.
    - `LinearRegression` with `K-Fold` Cross-Validation is checked to understand the linear score.
    - `PolynomialFeatures` cross-validation is reviewed and noted poor validation score.
    - The preliminary test from `XGBRegressor` CV_Score and R_Square are found to be the highest.
    - Differences between XGBRegressor Prediction and **Sample submission** are checked.
    - The prediction underestimates the sale prices.


## REFLECTION
- Most codes used in the notebooks are duplicated from Kaggle examples to ease first-time learning. 
- This experiment aims to practise and prepare for a black-box optimisation (BBO) competition. To strengthen the skills, the notebooks are created for experimenting with the pre-defined data sets.
- This experiment aims to understand how to work with different scales of the features using `sklearn.preprocessing`. 
- `train_test_split`, `cross_val_scor` and `KFold` are used to understand how to select the ML model.
- Feature engineering and Hyperparameter tuning was limited in this work.
- The ignorance of feature importance such as neighborhood, building types, house style and house construction types contribute to the poor prediction results.

