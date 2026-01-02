# 📈 Hull Tactical Market Prediction — HGBR Model

## Overview
This repository contains the **HistGradientBoostingRegressor (HGBR)** pipeline developed for the **Hull Tactical Market Prediction Competition**.  
While the broader project explored multiple modeling approaches (including OLS and MLP), this repository focuses on the **tree-based model** that achieved the **most stable performance and highest risk-adjusted returns**.

The objective is to forecast **S&P 500 excess returns** using a high-dimensional financial dataset and generate trading signals optimized for the **Sharpe Ratio**.

---

## 📌 Problem Statement
Given hundreds of financial indicators—including:
- Market dynamics
- Technical indicators
- Macroeconomic variables
- Investor sentiment measures

the task is to predict **future excess returns** and construct a signal that maximizes **risk-adjusted performance**, evaluated using the **Sharpe Ratio**.

---

## 🚀 My Contribution: HGBR Pipeline
I designed and implemented a robust **end-to-end HGBR modeling pipeline** tailored for noisy and non-stationary financial time-series data.

### Why HistGradientBoostingRegressor?
- Captures complex **non-linear relationships**
- Native handling of **missing values**
- Built-in **L2 regularization** for improved generalization
- Efficient training on large feature spaces

---

## 🔬 Key Features & Innovations

### 1. Asymmetric Tanh Signal Rule
A custom asymmetric tanh transformation is applied to model predictions to generate trading signals.  
This design cautiously increases leverage while explicitly penalizing excessive exposure, balancing return potential with downside volatility.

### 2. Advanced Data Imputation Strategy
A multi-stage imputation pipeline was implemented to preserve temporal causality:
- **EWMA (Exponentially Weighted Moving Average)** to emphasize recent observations
- **Forward-fill with global mean fallback** to ensure continuity without introducing future look-ahead bias

### 3. Feature Selection via Permutation Importance
Instead of using the full feature set, **Permutation Importance** was used to retain only the most predictive features, reducing noise and mitigating overfitting.

### 4. Hyperparameter Optimization
A grid search was conducted over:
- Number of boosting iterations (`max_iter`)
- Number of selected features (`top_k`)
- Asymmetric tanh signal parameters

The optimization objective was to **maximize validation Sharpe Ratio**.

---

## 📊 Performance Results

| Model            | Public Kaggle Sharpe Ratio |
|------------------|---------------------------|
| **HGBR (This Model)** | **2.252** |
| OLS Baseline     | 1.287 |

The HGBR model significantly outperformed baseline and alternative approaches in terms of **risk-adjusted returns**.

---

## 📁 Repository Structure
```text
.
├── HGBR_Final_Submission.ipynb
└── README.md
