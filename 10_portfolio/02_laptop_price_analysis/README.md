# Laptop Price Analysis

**Question:** Which laptop specifications explain price, and how well can a regression model generalize?

**Data:** Laptop pricing dataset (IBM Data Analysis with Python): manufacturer, CPU, RAM, storage, GPU, screen, weight, price.

## Approach

- Data wrangling: missing values, unit conversion, normalization, binning and one-hot encoding
- EDA: correlation of CPU frequency, screen size and weight with price; grouping and pivot tables
- Model evaluation: train/test split, cross-validation, polynomial features
- Ridge regression with a sweep over alpha, and Grid Search over several features

## Key findings

- Each candidate predictor (RAM, CPU frequency, SSD storage, screen size, weight, CPU core, OS, GPU, category) is screened with Pearson correlation and p-values against price.
- GPU × CPU-core pivot tables and heatmaps show how hardware combinations relate to average price.
- Model selection compares train and test R² across polynomial degrees and Ridge alphas to detect overfitting.

## Stack

pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn

## Run it

Open [`analysis.ipynb`](analysis.ipynb) in Jupyter or VS Code. The data downloads from its public URL in the first cells.
