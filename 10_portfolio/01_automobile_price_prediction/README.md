# Automobile Price Prediction

**Question:** Which car characteristics drive price, and which regression model predicts it best?

**Data:** Automobile (imports-85) dataset, 201 cars, 27 attributes (IBM Data Analysis with Python).

## Approach

- Data wrangling: missing values, type fixing, normalization, binning, dummy variables
- EDA: descriptive statistics, grouping and pivot tables, box plots
- Pearson correlation with p-values to select significant predictors
- Simple, multiple and polynomial linear regression; scikit-learn Pipelines
- Train/test split, cross-validation, overfitting analysis, Ridge regression, Grid Search

## Results

| Model | R² | MSE |
|---|---|---|
| Simple linear (highway-mpg) | 0.497 | 3.16 × 10⁷ |
| Polynomial (highway-mpg) | 0.674 | 2.05 × 10⁷ |
| **Multiple linear (horsepower, curb-weight, engine-size, highway-mpg)** | **0.809** | **1.2 × 10⁷** |

## Key findings

- Correlation and p-value screening keeps engine size, horsepower, curb weight, width, length, wheel-base, bore, city/highway-mpg and drive-wheels as meaningful predictors of price.
- Multiple linear regression is the best model: R² 0.81 versus 0.50 for the single-variable model.
- High-degree polynomials overfit: around 200 horsepower the curve diverges from the test data. Ridge regularization, tuned with Grid Search, reduces this.

## Stack

pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn

## Run it

Open [`analysis.ipynb`](analysis.ipynb) in Jupyter or VS Code. The data downloads from its public URL in the first cells.
