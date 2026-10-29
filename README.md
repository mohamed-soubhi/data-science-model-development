# 🚗💻 Data Science Exploratory Analysis & Predictive Modeling

[![GitHub Pages](https://img.shields.io/badge/Live_Presentation-GitHub_Pages-2ea44f?style=for-the-badge&logo=github)](https://mohamed-soubhi.github.io/data-science-model-development/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Statistics-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)

A rigorous, end-to-end statistical audit, exploratory data analysis (EDA), and machine learning regression framework for **Automobile** and **Laptop** pricing datasets.

👉 **[Launch the Live Interactive Presentation Dashboard](https://mohamed-soubhi.github.io/data-science-model-development/)**

---

## 📑 Core Modules in this Repository

| Module | Notebook | Domain | Key Methodologies & Additions |
| :--- | :--- | :--- | :--- |
| **EDA Lab** | [`Exploratory_data_analysis_cars.ipynb`](Exploratory_data_analysis_cars.ipynb) | Automobile | Pearson correlation ($r$ & $p$-values), ANOVA $F$-tests (`drive-wheels`, `body-style`, `engine-location`), 2D Pivot Heatmaps, Value Counts, 5 automated unit tests |
| **EDA Lab (Copy 2)** | [`Exploratory_data_analysis_cars2.ipynb`](Exploratory_data_analysis_cars2.ipynb) | Automobile | Identical verified, production-grade review notebook |
| **Model Dev Lab** | [`DA0101EN-4-Review-Model-Development.ipynb`](DA0101EN-4-Review-Model-Development.ipynb) | Automobile | SLR, MLR (4 features), Poly 2/3/11, Train/Test split (80/20), 5-Fold CV, VIF multicollinearity, OLS hypothesis testing, 5 automated unit tests |
| **Model Dev Practice** | [`practice_model_development_laptops.ipynb`](practice_model_development_laptops.ipynb) | Laptop | SLR (`CPU_frequency`), MLR (7 features), Polynomial Pipelines, Dimensionality Collapse detection, VIF, OLS $p$-values, 5 automated unit tests |

---

## 🔍 Major Analytical Insights

### 1. Exploratory Data Analysis (Automobiles)
- **Continuous Drivers**: `engine-size` ($r = +0.872, p < 10^{-63}$), `curb-weight` ($r = +0.834, p < 10^{-52}$), and `horsepower` ($r = +0.810, p < 10^{-47}$) are the dominant linear predictors of car price.
- **Fuel Economy**: `highway-mpg` ($r = -0.705$) and `city-mpg` ($r = -0.687$) exhibit strong inverse relationships with price (economy commuter cars vs. luxury vehicles).
- **Negligible Predictors**: `peak-rpm` ($r = -0.102, p = 0.151$) and `stroke` ($r = +0.082, p = 0.250$) are statistically non-significant ($p > 0.05$) with flat regression lines.
- **Categorical Drivetrain**: One-way ANOVA confirms `drive-wheels` significantly separates price ($F = 67.95, p < 10^{-22}$). Pairwise tests prove `rwd vs fwd` is massively distinct ($F = 130.55$), whereas `4wd vs fwd` has no significant price difference ($F = 0.67, p = 0.418$).
- **Class Imbalance Warning**: Only 3 cars out of 201 (1.5%) have rear engines. While rear-engine cars appear expensive, this tiny sample size is prone to severe modeling distortion.

### 2. Model Development & Overfitting Traps
- **Runge's Phenomenon (Automobiles - Degree 11)**: Fitting an 11th-order polynomial minimizes training error, but introduces severe oscillations at the boundaries ($40 - 55$ mpg), causing out-of-sample Test RMSE to spike to **$\$7,370$**.
- **Curse of Dimensionality (Laptops - Degree 2)**: Expanding 7 features into degree-2 polynomials created **35 terms**. With only 190 training samples, the model memorized noise, producing a negative Test $R^2$ (**$-0.048$**). Linear MLR proved far superior on unseen test data (**Test $R^2 = 0.269$, RMSE = $\$410.58$**).
- **Multicollinearity & Sign Flipping**: In automobiles, `engine-size` and `curb-weight` have $VIF > 5.5$ ($r = 0.85$). In SLR, `highway-mpg` has a negative slope ($-822$), but in MLR its coefficient becomes statistically non-significant ($p = 0.627$) because fuel efficiency is redundant once engine size and weight are controlled.

---

## 🏆 Predictive Model Leaderboard

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

## 🛡️ Automated Quality Assurance (15 / 15 Passed)
All notebooks include automated assertion test suites:
- **Automobile EDA Suite (5 Tests):** Dataset integrity, Pearson correlation bounds ($[-1, 1]$), verified engine-size ($r > 0.8$) and mpg ($r < -0.6$) bounds, pivot table completeness, ANOVA statistical significance ($p < 0.05$).
- **Automobile Modeling Suite (5 Tests):** In-sample $R^2$ bounds, $R^2_{\text{MLR}} > R^2_{\text{SLR}}$, prediction vector completeness, Scikit-Learn Pipeline equivalence, empirical overfitting detection.
- **Laptop Modeling Suite (5 Tests):** Dataset contract, price positivity, prediction completeness, pipeline standardization equivalence, MLR out-of-sample dominance over Poly 2.

---

## 📁 Repository Structure

```text
├── index.html                                   # Interactive Presentation Dashboard (GitHub Pages)
├── presentation_dashboard.html                  # Standalone offline presentation deck with embedded base64 charts
├── Exploratory_data_analysis_cars.ipynb         # Enhanced Automobile EDA Notebook (fully executed)
├── Exploratory_data_analysis_cars2.ipynb        # Enhanced Automobile EDA Notebook (Copy 2, fully executed)
├── DA0101EN-4-Review-Model-Development.ipynb    # Enhanced Automobile Price Prediction Lab (fully executed)
├── practice_model_development_laptops.ipynb     # Enhanced Laptop Price Prediction Lab (fully executed)
├── automobileEDA.csv                            # Automobile dataset (201 observations)
├── laptops.csv                                  # Laptop pricing dataset (238 observations)
├── build_enhanced_eda_notebook.py               # Automated builder for EDA notebooks
├── build_enhanced_notebook.py                   # Automated builder for Automobile Model Dev notebook
├── build_enhanced_laptops_notebook.py           # Automated builder for Laptop notebook
├── build_presentation_html.py                   # Automated builder for the HTML presentation dashboard
└── README.md                                    # Comprehensive project documentation
```

---

## 🖥️ Interactive Presentation Dashboard Features

The generated [`index.html`](index.html) is an executive slide presentation with:
- **Slide Navigation & Tabs**: Switch between 6 presentation slides via top tabs or keyboard arrows (`Left`/`Right` or `Spacebar`).
- **Interactive KPIs**: Live metrics summarizing models, $R^2$ scores, ANOVA statistics, and pricing drivers.
- **Embedded Visualizations**: 10 self-contained, high-resolution base64 plots (zero external CDN dependencies).
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
