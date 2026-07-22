# 📚 Data Science Knowledge Base

Personal reference notebook collection covering advanced Data Science topics — built as runnable Jupyter notebooks with explanations, code examples, and personal course notes.

## 📂 Structure

| Folder | Notebook | Topics |
|---|---|---|
| `01_sql/` | `sql_advanced.ipynb` | DDL/DML, constraints, joins, window functions, CTEs, string patterns, date functions, subqueries |
| `02_python/` | `python_advanced.ipynb` | Decorators, generators, itertools, NumPy, pandas, concurrency, DB access |
| `03_statistics/` | `statistics_advanced.ipynb` | Distributions, hypothesis testing, A/B testing, Bayesian inference, bootstrap |
| `04_visualization/` | `visualization_advanced.ipynb` | Matplotlib, Seaborn, Plotly interactive & animated charts |
| `05_ml_modeling/` | `ml_modeling_advanced.ipynb` | Sklearn pipelines, cross-validation, SHAP, XGBoost, LightGBM, class imbalance |
| `06_deep_learning/` | `deep_learning_advanced.ipynb` | PyTorch MLP/CNN/LSTM, training loop, transfer learning, transformers |
| `07_etl_data_wrangling/` | `etl_data_wrangling_advanced.ipynb` | Data profiling, missing data, outliers, string cleaning, ETL pipeline, Polars |
| `08_git_best_practices/` | `git_best_practices_advanced.ipynb` | Git workflow, project structure, environments, testing, MLflow, Docker/CI |

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
