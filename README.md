# 🚗💻🏥🏡 Data Science Analytics, Predictive Modeling & Refinement Portfolio

[![GitHub Pages](https://img.shields.io/badge/Live_Presentation-GitHub_Pages-2ea44f?style=for-the-badge&logo=github)](https://mohamed-soubhi.github.io/data-science-model-development/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Statistics-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)

An enterprise-grade, end-to-end data science portfolio encompassing **Exploratory Data Analysis (EDA)**, **Statistical Hypothesis Testing (ANOVA, Pearson, OLS Inference)**, **Predictive Machine Learning Regression**, **Overfitting Diagnostics (Runge's Phenomenon, Curse of Dimensionality)**, and **Automated Hyperparameter Optimization (`GridSearchCV`)** across four distinct industries:
1. **Automotive Market Pricing** (`automobileEDA.csv`, 201 vehicles)
2. **Personal Computing Hardware Pricing** (`laptops.csv`, 238 laptops)
3. **Healthcare Insurance Actuarial Risk** (`insurance.csv`, 2,772 health records)
4. **King County Residential Real Estate Pricing** (`housing.csv`, 21,613 homes)

👉 **[Launch the Live Interactive Presentation Dashboard](https://mohamed-soubhi.github.io/data-science-model-development/)**

---

## 📑 Portfolio Curriculum & Notebook Architecture

| Domain | Notebook | Phase | Focus & Key Methodologies |
| :--- | :--- | :--- | :--- |
| **Automobile** | [`Exploratory_data_analysis_cars.ipynb`](Exploratory_data_analysis_cars.ipynb) | Exploratory Analysis | Pearson correlation ranking ($r$ & $p$-values), one-way & pairwise ANOVA ($F=67.95$), 2D pivot heatmaps, extreme class imbalance audit |
| **Automobile** | [`Exploratory_data_analysis_cars2.ipynb`](Exploratory_data_analysis_cars2.ipynb) | Exploratory Analysis | Verified duplicate notebook for redundant lab tracking |
| **Automobile** | [`DA0101EN-4-Review-Model-Development.ipynb`](DA0101EN-4-Review-Model-Development.ipynb) | Predictive Modeling | SLR vs. MLR (4 features), Polynomial fits (Deg 2, 3, 11), VIF multicollinearity, OLS inference, 80/20 Train/Test split, 5-Fold CV |
| **Laptop** | [`practice_model_development_laptops.ipynb`](practice_model_development_laptops.ipynb) | Model Development | SLR (`CPU_frequency`), MLR (7 features), Polynomial Pipelines, Curse of Dimensionality exposure, VIF, OLS parameter testing |
| **Laptop** | [`practice_model_evaluation.ipynb`](practice_model_evaluation.ipynb) | Evaluation & Tuning | K-Fold Cross Validation (`cv=4`), Polynomial Overfitting sweeps (Deg 1-5 collapse), Ridge alpha sweep, `GridSearchCV` ($\alpha=100$) |
| **Insurance** | [`practice_project.ipynb`](practice_project.ipynb) | Capstone Project | End-to-end data wrangling, clinical EDA (smoking surcharge +$23,800), SLR, MLR, Standardized Polynomial Pipeline at scale ($R^2=0.834$) |
| **Housing** | [`House_Sales_in_King_Count_USA.ipynb`](House_Sales_in_King_Count_USA.ipynb) | Capstone Project | 10/10 questions solved: Data hygiene (null imputations, dropping IDs), Exploratory visualizations (Boxplots, Regplots), SLR ($R^2=0.493$), MLR ($R^2=0.658$), Standardized Polynomial Pipeline ($R^2=0.751$), Ridge Regularization ($R^2=0.700$) |

---

## 🔍 Major Empirical Discoveries & Methodological Insights

### 1. The Interaction of Sample Size vs. Model Complexity ($n \gg p$ vs. $n \approx p$)
- **The Dimensionality Trap on Small Datasets (Laptops, $n=190$ train):** Expanding 7 features into degree-2 polynomials created **35 terms**. Because the ratio of samples to features was under $6:1$, the model memorized noise: in-sample $R^2$ was $0.675$, but out-of-sample Test $R^2$ collapsed to **$-0.048$** (worse than predicting the mean).
- **Polynomial Superiority at Scale (Insurance, $n=2,217$ train):** Expanding 6 features into 27 terms on 2,217 samples ($n \gg p$) allowed the model to capture genuine non-linear clinical risk compounding (`smoker * bmi`), lifting out-of-sample Test $R^2$ from **$0.740$ to $0.834$** and reducing Test RMSE by over $\$1,100$.

### 2. Overfitting & Runge's Phenomenon
- In the Automobile study, fitting an 11th-order polynomial on fuel efficiency minimized training error, but produced severe boundary oscillations ($40 - 55$ mpg), spiking out-of-sample Test RMSE to **$\$7,370$**.
- In the Laptop evaluation study, sweeping polynomial degrees from 1 to 5 showed that while Train $R^2$ rose monotonically ($0.168 \rightarrow 0.417$), Test $R^2$ plunged into catastrophic negative territory (**$-0.995$ at degree 4** and **$-1.376$ at degree 5**).

### 3. Hyperparameter Optimization via `GridSearchCV`
- Unregularized degree-2 polynomial expansion on laptops yielded an overfitted test score ($-0.073$).
- Systematic 4-fold `GridSearchCV` selected **$\alpha = 100$** for Ridge regression, shrinking redundant interaction weights and boosting out-of-sample Test $R^2$ to **$+0.386$** (reducing Test RMSE from $\$489$ to $\$369$).

### 4. Multicollinearity & Confounding Variables (VIF)
- In automobiles, `engine-size` and `curb-weight` have $VIF > 5.5$ ($r = 0.85$). In SLR, `highway-mpg` has a negative slope ($-822$), but in MLR its coefficient becomes statistically non-significant ($p = 0.627$) because fuel efficiency is redundant once engine size and weight are accounted for.

---

## 🏆 Master Cross-Study Benchmark Leaderboard

| Domain | Model Architecture | Train $R^2$ | Test $R^2$ (Unseen) | Test RMSE ($) | Test MAE ($) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Insurance** | **Standardized Poly 2 Pipeline** | **0.8449** | **0.8339** | **$4,978** | **$3,001** | 🏆 **Top Overall Portfolio Model** |
| **Insurance** | Regularized Ridge (Poly 2, $\alpha=0.1$) | 0.8449 | 0.8339 | $4,978 | $3,002 | Highly stable regularized fit |
| **Insurance** | Multiple Linear Regression (6 feats) | 0.7509 | 0.7396 | $6,229 | $4,244 | Strong linear baseline |
| **Housing** | **Standardized Poly 2 Pipeline (11 feats)** | **0.7513** | **0.7513** | — | — | 🏆 **Top Real Estate Model ($n=21,613$)** |
| **Housing** | Regularized Ridge (Poly 2, $\alpha=0.1$) | 0.7003 | 0.7003 | — | — | Robust regularized real estate model |
| **Housing** | Multiple Linear Regression (11 feats) | 0.6577 | 0.6577 | — | — | Strong multi-feature baseline |
| **Automobile** | **Multiple Linear Regression (4 feats)** | **0.8089** | **0.7637** | **$5,377** | **$3,680** | 🏆 **Top Automobile Model** |
| **Automobile** | Regularized Ridge (Poly Deg 2) | 0.8315 | 0.7680 | $5,328 | $3,546 | Robust regularized alternative |
| **Automobile** | Simple Linear Regression (engine-size) | 0.7635 | 0.7102 | $5,934 | $4,098 | Strong single-variable baseline |
| **Housing** | Simple Linear Regression (sqft_living) | 0.4929 | 0.4929 | — | — | Square footage explains ~50% price |
| **Insurance** | Simple Linear Regression (smoker) | 0.6231 | 0.5991 | $7,713 | $5,739 | Smoker alone explains 60% of cost |
| **Automobile** | Polynomial Degree 11 (highway-mpg) | 0.7303 | 0.5560 | $7,370 | $4,845 | ❌ **Overfitted (Runge's Phenomenon)** |
| **Laptop** | **GridSearchCV Tuned Ridge ($\alpha=100$)** | **0.6033** | **0.3861** | **$368.95** | **$283.42** | 🏆 **Top Laptop Model** |
| **Laptop** | Multiple Linear Regression (7 feats) | 0.5212 | 0.2685 | $410.58 | $332.66 | Robust unregularized linear model |
| **Laptop** | Polynomial Degree 2 (Unregularized) | 0.6746 | -0.0475 | $491.33 | $328.12 | ❌ **Curse of Dimensionality** |
| **Laptop** | Polynomial Degree 5 (CPU_frequency) | 0.4173 | -1.3759 | $716.48 | $457.82 | ❌ **Severe Variance Collapse** |

---

## 🛡️ Automated Quality Assurance Suite (25 / 25 Passed)

Every notebook in this repository contains an automated programmatic assertion suite:
1. **Automobile EDA Suite (5 Tests):** Dataset schema, Pearson bounds ($[-1, 1]$), verified physical relationships ($r_{\text{engine}} > 0.8, r_{\text{mpg}} < -0.6$), pivot integrity, ANOVA statistical significance ($p < 0.05$).
2. **Automobile Modeling Suite (5 Tests):** In-sample $R^2$ constraints, $R^2_{\text{MLR}} > R^2_{\text{SLR}}$, prediction vector completeness, Scikit-Learn Pipeline equivalence, empirical overfitting detection.
3. **Laptop Modeling Suite (5 Tests):** Dimension integrity, price positivity, prediction vector sanity, pipeline scaling match, out-of-sample MLR dominance.
4. **Laptop Evaluation Suite (5 Tests):** Cross-validation vector completeness, overfitting detection ($R^2_{\text{deg 1}} > R^2_{\text{deg 5}}$), GridSearch parameter selection, tuned Ridge dominance.
5. **Insurance Capstone Suite (5 Tests):** Data contract compliance, $R^2$ hierarchy, complete predictions, pipeline transformation equivalence, polynomial superiority at enterprise scale.

---

## 📁 Repository Structure

```text
├── index.html                                   # Master Interactive Presentation Dashboard (GitHub Pages)
├── presentation_dashboard.html                  # Standalone offline presentation deck with embedded base64 charts
├── Exploratory_data_analysis_cars.ipynb         # Automobile EDA Lab (fully executed)
├── Exploratory_data_analysis_cars2.ipynb        # Automobile EDA Lab (Copy 2, fully executed)
├── DA0101EN-4-Review-Model-Development.ipynb    # Automobile Predictive Modeling Lab (fully executed)
├── practice_model_development_laptops.ipynb     # Laptop Model Development Lab (fully executed)
├── practice_model_evaluation.ipynb              # Laptop Evaluation & Tuning Lab (fully executed)
├── practice_project.ipynb                       # Medical Insurance Capstone Lab (fully executed)
├── House_Sales_in_King_Count_USA.ipynb          # King County Real Estate Regression Lab (10/10 questions solved)
├── automobileEDA.csv                            # Automobile dataset (201 observations)
├── laptops.csv                                  # Laptop pricing dataset (238 observations)
├── insurance.csv                                # Medical insurance dataset (2,772 observations)
├── housing.csv                                  # King County housing dataset (21,613 observations)
├── build_enhanced_eda_notebook.py               # Automated builder for EDA notebooks
├── build_enhanced_notebook.py                   # Automated builder for Automobile Model Dev notebook
├── build_enhanced_laptops_notebook.py           # Automated builder for Laptop notebook
├── build_enhanced_model_evaluation.py           # Automated builder for Laptop Model Evaluation notebook
├── build_enhanced_insurance_project.py          # Automated builder for Medical Insurance Capstone notebook
├── solve_house_sales.py                         # Solution injector for House Sales King County project
├── build_presentation_html.py                   # Automated builder for the HTML presentation dashboard
└── README.md                                    # Master project documentation
```

---

## 🖥️ Presentation Dashboard Features

The generated [`index.html`](index.html) is an executive slide presentation with:
- **Slide Navigation & Tabs**: Switch between 7 presentation slides via top tabs or keyboard arrows (`Left`/`Right` or `Spacebar`).
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
   pip install pandas numpy matplotlib seaborn scikit-learn scipy nbformat nbconvert tqdm
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
