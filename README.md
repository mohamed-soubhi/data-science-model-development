# 🚗💻 Data Science Model Development & Predictive Analytics

[![GitHub Pages](https://img.shields.io/badge/Live_Presentation-GitHub_Pages-2ea44f?style=for-the-badge&logo=github)](https://mohamed-soubhi.github.io/data-science-model-development/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

A comprehensive review, diagnostic audit, and pedagogical enhancement of practical regression modeling for **Automobile** and **Laptop** pricing datasets.

👉 **[View the Live Interactive Presentation Dashboard](https://mohamed-soubhi.github.io/data-science-model-development/)**

---

## 📑 Project Highlights

This repository transforms standard introductory regression labs into a production-grade machine learning and statistical evaluation framework:

1. **Beyond In-Sample $R^2$**: Demonstrates why training set metrics create an illusion of predictive power and uses **80/20 train/test splits** and **5-fold cross-validation** to measure true generalization.
2. **Exposing the Overfitting Traps**:
   - **Runge's Phenomenon (High-Degree Polynomials)**: Shows how an 11th-order polynomial wiggles at boundary conditions and collapses on unseen test data ($RMSE = \$7,370$).
   - **Curse of Dimensionality**: Shows how expanding 7 laptop features into 35 polynomial interaction terms leads to negative test $R^2$ ($-0.048$) on a small dataset ($n=190$).
3. **Statistical Regression Diagnostics**:
   - **Multicollinearity (VIF)**: Uncovers why `highway-mpg` flips sign between Simple and Multiple Linear Regression due to high correlation with `engine-size` and `curb-weight` ($VIF > 5.5$).
   - **Gauss-Markov Assumptions**: Q-Q plots, Shapiro-Wilk normality tests, and residuals vs. fitted values.
   - **Inference & Hypothesis Testing**: Closed-form OLS parameter estimates with standard errors, $t$-statistics, and $p$-values.
4. **Automated Quality Assurance**: 10 programmatic unit tests verifying data integrity, mathematical bounds, pipeline equivalence, and prediction completeness.
5. **Universal Portability**: Replaced browser-locked dependencies (`piplite`, `pyodide`) with self-healing data loaders compatible with standard Python, VS Code, JupyterLab, and Google Colab.

---

## 📊 Benchmark Leaderboard

### Automobile Price Prediction (201 cars)
| Model Architecture | Train $R^2$ | Test $R^2$ (Unseen) | Test RMSE ($) | Test MAE ($) | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Multiple Linear Regression (4 features)** | **0.8089** | **0.7637** | **$5,377** | **$3,680** | 🏆 **Top Overall Model** |
| **Regularized Ridge (Poly Deg 2)** | 0.8315 | 0.7680 | $5,328 | $3,546 | 🛡️ Best Regularized Alternative |
| **Polynomial Degree 2 (Pipeline)** | 0.8526 | 0.7561 | $5,462 | $3,634 | Slight overfitting |
| **Simple Linear Regression (engine-size)** | 0.7635 | 0.7102 | $5,934 | $4,098 | Strong single-variable baseline |
| **Polynomial Degree 3 (highway-mpg)** | 0.6933 | 0.6014 | $6,983 | $4,321 | Captures univariate curvature |
| **Simple Linear Regression (highway-mpg)** | 0.5013 | 0.4247 | $8,390 | $5,618 | Underfits; high error |
| **Polynomial Degree 11 (highway-mpg)** | 0.7303 | 0.5560 | $7,370 | $4,845 | ❌ **Overfitted (Runge's Phenomenon)** |

### Laptop Price Prediction (238 laptops)
| Model Architecture | Train $R^2$ | Test $R^2$ (Unseen) | Test RMSE ($) | Test MAE ($) | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Multiple Linear Regression (7 features)** | **0.5212** | **0.2685** | **$410.58** | **$332.66** | 🏆 **Top Overall Model** |
| **Poly Degree 3 (CPU_frequency)** | 0.2711 | 0.1065 | $453.78 | $376.77 | Captures CPU clock non-linearity |
| **Poly Degree 5 (CPU_frequency)** | 0.3131 | 0.1004 | $455.34 | $374.77 | Mild overfitting on CPU curve |
| **Ridge Regularized (Poly 2)** | 0.6707 | -0.0134 | $483.26 | $328.69 | Negative $R^2$ on test data |
| **Simple Linear Regression (CPU_freq)** | 0.1384 | -0.0372 | $488.91 | $389.68 | Inadequate baseline |
| **Polynomial Degree 2 (7 features, 35 terms)** | 0.6746 | -0.0475 | $491.33 | $328.12 | ❌ **Severe Dimensionality Overfitting** |

---

## 📁 Repository Structure

```text
├── index.html                                   # Interactive Presentation Dashboard (GitHub Pages)
├── presentation_dashboard.html                  # Standalone offline presentation deck with embedded base64 charts
├── DA0101EN-4-Review-Model-Development.ipynb    # Enhanced Automobile Price Prediction Lab (fully executed)
├── practice_model_development_laptops.ipynb     # Enhanced Laptop Price Prediction Lab (fully executed)
├── automobileEDA.csv                            # Automobile dataset (201 observations)
├── laptops.csv                                  # Laptop pricing dataset (238 observations)
├── build_enhanced_notebook.py                   # Automated builder for Automobile notebook
├── build_enhanced_laptops_notebook.py           # Automated builder for Laptop notebook
├── build_presentation_html.py                   # Automated builder for the HTML presentation dashboard
└── README.md                                    # Project documentation
```

---

## 🖥️ Interactive Presentation Dashboard Features

The generated [`index.html`](index.html) is an executive slide presentation with:
- **Slide Navigation & Tabs**: Switch between 5 presentation slides via top tabs or keyboard arrows (`Left`/`Right` or `Spacebar`).
- **Interactive KPIs**: Live metrics summarizing models, $R^2$ scores, and pricing drivers.
- **Embedded Visualizations**: 8 self-contained, high-resolution base64 plots (no external CDN dependencies required).
- **Fullscreen Mode & PDF Export**: Instant toggle for conference screen sharing or one-click printable executive reports.

---

## 🚀 How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mohamed-soubhi/data-science-model-development.git
   cd data-science-model-development
   ```

2. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy nbformat nbconvert
   ```

3. **Launch the notebooks**:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

4. **Open the Presentation Dashboard**:
   Simply open `index.html` or `presentation_dashboard.html` in any web browser!

---

## 📜 License
This project is released under the MIT License for educational and research purposes.
