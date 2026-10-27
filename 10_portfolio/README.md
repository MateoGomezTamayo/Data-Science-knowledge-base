# Data Science Portfolio: Applied Analyses

Sixteen end-to-end analyses built during the IBM Data Science Professional Certificate. Each folder has a notebook with the full code and saved outputs, a key chart, and a README with the question, approach, results and findings.

| # | Project | Type | Techniques | Headline result |
|---|---|---|---|---|
| 01 | [Automobile Price Prediction](01_automobile_price_prediction/) | Regression | Correlation + p-values, SLR / MLR / polynomial, Pipelines, Ridge, Grid Search | Multiple regression R² **0.81** (vs 0.50 for one variable) |
| 02 | [Laptop Price Analysis](02_laptop_price_analysis/) | Wrangling + regression | Cleaning, binning, encoding, Pearson tests, cross-validation, Ridge | Full cleaning-to-model-selection workflow |
| 03 | [CO2 Emissions Regression](03_co2_emissions_regression/) | Regression | Simple and multiple linear regression, standardization | R² **0.76**, MAE 24.1 g/km |
| 04 | [Customer Churn Prediction](04_customer_churn_prediction/) | Binary classification | Logistic regression, log loss, coefficient interpretation | Tenure lowers churn, equipment rental raises it |
| 05 | [Obesity Level Classification](05_obesity_level_classification/) | Multiclass classification | One-vs-All vs One-vs-One, one-hot encoding, stratified split | One-vs-One accuracy **92.2%** |
| 06 | [Drug Prescription Decision Tree](06_drug_prescription_decision_tree/) | Multiclass classification | Decision tree (entropy), depth tuning | Accuracy **98.3%** |
| 07 | [NYC Taxi Tip Prediction](07_taxi_tip_regression_tree/) | Regression | Regression tree, correlation analysis | R² 0.03, trip data doesn't explain tips |
| 08 | [Credit Card Fraud Detection](08_credit_card_fraud_detection/) | Imbalanced classification | Decision tree vs linear SVM, class weights, ROC-AUC | SVM ROC-AUC **0.986** |
| 09 | [Australia Wildfire Analysis](09_australia_wildfire_analysis/) | EDA + dashboard | Matplotlib, Seaborn, Folium, Dash + Plotly | 26,406 records, 7 regions, interactive dashboard |
| 10 | [Automobile Sales During Recessions](10_automobile_sales_recession_analysis/) | EDA + dashboard | Comparative visual analysis, Dash | Sales fall by more than half in recessions |
| 11 | [Impact of Noise on Data Quality](11_noise_impact_breast_cancer/) | Data quality | Gaussian noise simulation, distribution and correlation analysis | Noisy feature keeps r ≈ **0.89** with the original |
| 12 | [Principal Component Analysis](12_pca_dimensionality_reduction/) | Dimensionality reduction | PCA, projections, explained variance | 2 components keep **95.8%** of Iris variance |
| 13 | [t-SNE vs UMAP](13_tsne_vs_umap/) | Dimensionality reduction | t-SNE, UMAP, 3-D Plotly | UMAP keeps the global layout that t-SNE distorts |
| 14 | [California Housing Prices](14_california_housing_random_forest/) | Regression | Random Forest, residual diagnostics | R² **0.80**, MAE ≈ $32.8k |
| 15 | [K-Means Cluster Evaluation](15_kmeans_cluster_evaluation/) | Clustering | Silhouette, Davies-Bouldin, stability across initializations | Silhouette **0.53**; same inertia in 8 runs |
| 16 | [Regularization: Linear vs Ridge vs Lasso](16_regularization_ridge_lasso/) | Regression | Ridge, Lasso, outliers, sparse features | Lasso R² **0.98** vs 0.40 on sparse data |

**Stack:** Python, pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, Plotly, Dash, Folium, umap-learn.

Datasets come from the IBM Skills Network course materials and public sources (UCI, Kaggle). Each notebook downloads its data from a public URL.
