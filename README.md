# 📚 Data Science Knowledge Base

Personal reference notebook collection covering advanced Data Science topics — built as runnable Jupyter notebooks with explanations, code examples, and personal course notes.

## ⭐ Portfolio: applied analyses

Sixteen end-to-end projects, each with a notebook, a key chart and a README of findings. See [`10_portfolio/`](10_portfolio/).

| Project | Headline result |
|---|---|
| [Automobile Price Prediction](10_portfolio/01_automobile_price_prediction/) | Multiple regression R² 0.81 |
| [Laptop Price Analysis](10_portfolio/02_laptop_price_analysis/) | Cleaning → EDA → Ridge / Grid Search workflow |
| [CO2 Emissions Regression](10_portfolio/03_co2_emissions_regression/) | R² 0.76, MAE 24.1 g/km |
| [Customer Churn Prediction](10_portfolio/04_customer_churn_prediction/) | Churn drivers from logistic regression coefficients |
| [Obesity Level Classification](10_portfolio/05_obesity_level_classification/) | One-vs-One accuracy 92.2% |
| [Drug Prescription Decision Tree](10_portfolio/06_drug_prescription_decision_tree/) | Accuracy 98.3% |
| [NYC Taxi Tip Prediction](10_portfolio/07_taxi_tip_regression_tree/) | Regression tree; a documented negative result |
| [Credit Card Fraud Detection](10_portfolio/08_credit_card_fraud_detection/) | SVM ROC-AUC 0.986 on 0.17% fraud |
| [Australia Wildfire Analysis](10_portfolio/09_australia_wildfire_analysis/) | EDA + Dash dashboard |
| [Automobile Sales During Recessions](10_portfolio/10_automobile_sales_recession_analysis/) | EDA + Dash dashboard |
| [Impact of Noise on Data Quality](10_portfolio/11_noise_impact_breast_cancer/) | Noise keeps r ≈ 0.89; creates artificial outliers |
| [Principal Component Analysis](10_portfolio/12_pca_dimensionality_reduction/) | 2 components keep 95.8% of Iris variance |
| [t-SNE vs UMAP](10_portfolio/13_tsne_vs_umap/) | UMAP keeps the global layout that t-SNE distorts |
| [California Housing Prices](10_portfolio/14_california_housing_random_forest/) | Random Forest R² 0.80, MAE ≈ $32.8k |
| [K-Means Cluster Evaluation](10_portfolio/15_kmeans_cluster_evaluation/) | Silhouette 0.53; stable across 8 initializations |
| [Regularization: Linear vs Ridge vs Lasso](10_portfolio/16_regularization_ridge_lasso/) | Lasso R² 0.98 vs 0.40 on sparse data |

## 📂 Structure

| Folder | Notebook | Topics |
|---|---|---|
| `01_sql/` | `sql_advanced.ipynb` | DDL/DML, constraints, joins, window functions, CTEs, string patterns, date functions, subqueries |
| `01_sql/` | `sql_practice.ipynb` | 10 exercises on an HR database: WHERE, JOIN, GROUP BY, HAVING, subqueries, UPDATE |
| `02_python/` | `python_advanced.ipynb` | Decorators, generators, itertools, NumPy, pandas, concurrency, DB access |
| `03_statistics/` | `statistics_advanced.ipynb` | Distributions, hypothesis testing, A/B testing, Bayesian inference, bootstrap |
| `04_visualization/` | `visualization_advanced.ipynb` | Matplotlib, Seaborn, Plotly interactive & animated charts |
| `04_visualization/` | `visualization_labs.ipynb`, Final Assignment notebooks | Area plots, treemaps, waffle charts, Folium maps, Dash dashboards |
| `05_ml_modeling/` | `ml_modeling_advanced.ipynb` | Sklearn pipelines, cross-validation, SHAP, XGBoost, LightGBM, class imbalance |
| `06_deep_learning/` | `deep_learning_advanced.ipynb` | PyTorch MLP/CNN/LSTM, training loop, transfer learning, transformers |
| `07_etl_data_wrangling/` | `etl_data_wrangling_advanced.ipynb` | Data profiling, missing data, outliers, string cleaning, ETL pipeline, Polars |
| `07_etl_data_wrangling/` | `data_preprocessing_notes.ipynb` | Formatting, type fixing, normalization (simple / min-max / z-score), binning, encoding |
| `08_git_best_practices/` | `git_best_practices_advanced.ipynb` | Git workflow, project structure, environments, testing, MLflow, Docker/CI |
| `09_generalist/` | `generalist_english.ipynb` | Grammar, critical reasoning, passage analysis |
| `10_portfolio/` | 16 project folders | Applied regression, classification, EDA and dashboard projects |

## 🚀 Getting Started

**Requirements:** Python 3.9+, Jupyter or VS Code with the Jupyter extension.

```bash
# Clone the repo
git clone https://github.com/MateoGomezTamayo/Data-Science-knowledge-base.git
cd Data-Science-knowledge-base

# Install common dependencies
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost lightgbm torch
```

Open any `.ipynb` file in Jupyter Lab, Jupyter Notebook, or VS Code and run the cells.

## 📝 SQL Notebook Highlights

The SQL notebook is the most actively developed. It includes:

- Table creation with **entity, referential, and domain integrity constraints**
- `ALTER TABLE` operations: ADD / DROP / RENAME / MODIFY columns
- `TRUNCATE` vs `DELETE` vs `DROP`
- **String patterns**: `LIKE`, `GLOB`, `SUBSTR`, `REPLACE`, `INSTR`
- **Date functions**: `DATEDIFF`, `DATE_ADD`, `julianday` (SQLite)
- `GROUP BY` + `HAVING` with all aggregate functions
- `ORDER BY` — multi-column, expressions, NULLs
- Window functions, CTEs, JOINs, subqueries, EXPLAIN

> All SQL examples run on **SQLite in-memory** via Python's `sqlite3` — no database server required.

## 🔄 Keeping It Updated

```bash
git add .
git commit -m "Add notes: <topic>"
git push
```

## 👤 Author

**Mateo Gomez Tamayo**  
[GitHub](https://github.com/MateoGomezTamayo)
