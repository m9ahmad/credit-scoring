#  Corporate Credit Risk Early Warning System (EWS)
### Multivariate Solvency Modeling & Predictive Risk Scoring

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app/)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](#)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](#)

## 🎯 Project Overview
This AI-driven platform provides automated **Credit Risk Assessment** for corporate portfolios. By integrating liquidity, solvency, and profitability ratios, the system predicts a firm's risk of default before it occurs. 

The core engine uses a **Random Forest Regressor** to identify the "Risk Drivers" that lead to financial distress, allowing credit officers to perform real-time **Stress Testing** on client financials.

---

## 🧪 Model Validation & Stress Test Findings
During the validation phase, the model was subjected to three distinct corporate archetypes to verify its predictive accuracy:

| Archetype | Key Metrics | Model Output | Financial Logic |
| :--- | :--- | :--- | :--- |
| **"The Zombie"** | D/E: 12.0, ROA: -10% | 🔴 **High Risk** | Excessive leverage + Negative returns = High Default Probability. |
| **"The Expansion"** | D/E: 4.5, Growth: 35% | 🟠 **Medium Risk** | High debt is offset by aggressive revenue growth and stable margins. |
| **"The Cash Cow"** | D/E: 0.2, QR: 3.5 | 🟢 **Low Risk** | Superior liquidity and minimal debt burden indicate peak stability. |

---

## 🚀 Key Features
* **Risk Driver Analytics:** Automatically identifies which ratios (e.g., Debt-to-Equity vs. Interest Coverage) are the primary indicators of risk in the current market.
* **Industry Benchmarking:** Compares firm-specific risk scores against industry-wide averages to filter out systemic vs. idiosyncratic risk.
* **Interactive Stress Lab:** A "What-If" simulator for Credit Officers to input pro-forma financials and see immediate risk rating changes.
* **Liquidity-Solvency Heatmap:** A 4-quadrant visualization identifying firms in the "Danger Zone" (Low Liquidity + High Leverage).

---

## 🛠️ Technical Stack
* **Engine:** Python 3.14 (Optimized for Apple Silicon M4)
* **ML Library:** Scikit-Learn (Random Forest Ensemble)
* **Visuals:** Plotly Express & Graph Objects
* **Deployment:** Streamlit Cloud

---

## 📂 Repository Structure
```text
├── app.py                     # Main AI Dashboard & ML Engine
├── requirements.txt           # Dependency Manifest
├── financial_risk_dataset.csv # Multivariate Corporate Financials
└── README.md                  # Technical Documentation
