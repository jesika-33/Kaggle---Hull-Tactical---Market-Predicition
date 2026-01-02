/******************************************************************************
 *  Project: Hull Tactical Market Prediction – HGBR Model
 *
 *  Description:
 *  This project implements a HistGradientBoostingRegressor (HGBR) pipeline
 *  for the Hull Tactical Market Prediction Competition. While the broader
 *  project explored multiple modeling approaches (including OLS and MLP),
 *  this component represents the highest-performing and most stable model,
 *  achieving superior risk-adjusted returns.
 *
 *  Objective:
 *  Forecast S&P 500 excess returns using a high-dimensional financial dataset
 *  comprising market dynamics, technical indicators, macroeconomic variables,
 *  and investor sentiment features. Model performance is evaluated using the
 *  Sharpe Ratio.
 *
 *  Model Overview:
 *  - Model Type: HistGradientBoostingRegressor (HGBR)
 *  - Learning Paradigm: Supervised regression
 *  - Evaluation Metric: Sharpe Ratio
 *
 *  Key Contributions:
 *  1. Asymmetric Tanh Signal Rule:
 *     Implemented a custom asymmetric tanh transformation to generate trading
 *     signals. This approach gradually increases leverage while explicitly
 *     penalizing excessive exposure, balancing return potential and downside
 *     risk.
 *
 *  2. Advanced Data Imputation Strategy:
 *     - EWMA-based imputation to emphasize recent observations while preserving
 *       temporal causality.
 *     - Forward-fill with global mean fallback to ensure continuity without
 *       introducing future look-ahead bias.
 *
 *  3. Feature Selection via Permutation Importance:
 *     Utilized permutation importance to retain only the most predictive
 *     features, reducing noise and mitigating overfitting in a
 *     high-dimensional setting.
 *
 *  4. Hyperparameter Optimization:
 *     Performed grid search over:
 *       - Number of boosting iterations (max_iter)
 *       - Number of selected features (top_k)
 *       - Asymmetric tanh signal parameters
 *     Optimization objective: maximize validation Sharpe Ratio.
 *
 *  Performance Summary:
 *  --------------------------------------------------------------------------
 *   Model                  | Public Kaggle Sharpe Ratio
 *  --------------------------------------------------------------------------
 *   HGBR (This Model)       | 2.252
 *   OLS Baseline            | 1.287
 *  --------------------------------------------------------------------------
 *
 *  Repository Contents:
 *  - HGBR_Final_Submission.ipynb
 *      Complete end-to-end pipeline including data preprocessing, feature
 *      selection, HGBR training, and signal generation.
 *
 *  - DDA3020_Project_Report.pdf
 *      Full technical report with mathematical derivations and comparative
 *      analysis across models.
 *
 *  Dependencies:
 *  - polars, pandas        : Data processing
 *  - scikit-learn          : HGBR, permutation importance
 *  - joblib                : Model persistence
 *
 *  Usage:
 *  Run the `HGBR_Final_Submission.ipynb` notebook to reproduce the full training
 *  and inference pipeline. The notebook is configured to automatically handle
 *  the competition's training and testing datasets.
 *
 *  Project Context:
 *  Collaborative coursework project for DDA3020 / Hull Tactical Market
 *  Prediction Competition.
 *
 ******************************************************************************/
