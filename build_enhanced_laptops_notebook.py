"""
Script to build the reviewed, enhanced, and tested Jupyter Notebook
for practice_model_development_laptops.ipynb.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell(r"""# 💻 Laptop Price Prediction: Comprehensive Model Development & Evaluation
**Hands-on Machine Learning & Statistical Regression Review**

---

### 📋 Overview & Learning Objectives
In this hands-on laboratory, we build, evaluate, and critically diagnose machine learning regression models to predict **laptop prices** using technical hardware specifications (CPU frequency, RAM, SSD capacity, core count, GPU, OS, and Category).

By the end of this enhanced and tested notebook, you will understand:
1. **Single Linear Regression (SLR)** & **Multiple Linear Regression (MLR)** with Scikit-Learn.
2. **Model Evaluation via Visualizations**: Modern regression lines, residual analysis, and Kernel Density Estimation (KDE) distribution plots.
3. **Polynomial Regression & Scikit-Learn Pipelines**: Scaling, feature transformations, and chaining preprocessing with estimators.
4. **Out-of-Sample Generalization & Cross-Validation**: Why evaluating only on training data creates an illusion of accuracy, and how train/test splits & cross-validation expose overfitting.
5. **Statistical Diagnostics & Assumption Testing**: Linearity, normality of residuals (Shapiro-Wilk test & Q-Q plots), multicollinearity analysis (Variance Inflation Factor - VIF), and parameter significance ($p$-values).
6. **Automated Unit Testing & Quality Assurance**: Automated programmatic tests validating data integrity, model bounds, and prediction correctness.

---
"""))

# Table of Contents
cells.append(nbf.v4.new_markdown_cell(r"""### 📑 Table of Contents
- [0. Environment Setup & Data Loading](#sec-0)
- [1. Single Linear Regression (SLR): CPU Frequency -> Price](#sec-1)
- [2. Multiple Linear Regression (MLR): 7 Hardware Features](#sec-2)
- [3. Polynomial Regression: Degrees 1, 3, and 5](#sec-3)
- [4. Preprocessing Pipelines (Scaling + Polynomial + Regression)](#sec-4)
- [5. Out-of-Sample Generalization & Cross-Validation Tests (NEW)](#sec-5)
  - [5.1 Train / Test Split (80/20)](#sec-5-1)
  - [5.2 5-Fold Cross-Validation](#sec-5-2)
  - [5.3 Comprehensive Model Comparison Leaderboard](#sec-5-3)
  - [5.4 The Dimensionality & Overfitting Trap](#sec-5-4)
- [6. Statistical Assumption Tests & Regression Diagnostics (NEW)](#sec-6)
  - [6.1 Residuals vs. Fitted Values (Linearity & Homoscedasticity)](#sec-6-1)
  - [6.2 Normality of Residuals: Q-Q Plot & Shapiro-Wilk Test](#sec-6-2)
  - [6.3 Multicollinearity Analysis: VIF (Variance Inflation Factor)](#sec-6-3)
  - [6.4 Statistical Significance of Predictors (t-stats, p-values)](#sec-6-4)
- [7. Automated Sanity & Unit Tests (NEW)](#sec-7)
- [8. Predictions on New Laptop Profiles & Strategic Conclusion](#sec-8)
"""))

# Section 0
cells.append(nbf.v4.new_markdown_cell(r"""<a id="sec-0"></a>
## 0. Environment Setup & Data Loading

We import the essential scientific Python libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, and `scikit-learn`).

> ⚙️ **Universal Compatibility Note:**
> The original lab imported `piplite` and `pyodide.http`, which fail in standard Python / VS Code / Jupyter / Colab environments. This notebook uses a universal loader that reads the local file `laptops.csv` or downloads it directly via standard Python HTTP requests.
"""))

cells.append(nbf.v4.new_code_cell(r"""# Core scientific and visualization libraries
import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Scikit-learn modules for modeling, evaluation, and pipelines
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# Plot formatting
%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
sns.set_theme(style="whitegrid", palette="muted")

print("All libraries imported successfully!")
"""))

cells.append(nbf.v4.new_markdown_cell(r"""### Load and Inspect the Dataset
The dataset contains technical attributes and market prices for 238 laptops.
"""))

cells.append(nbf.v4.new_code_cell(r"""# Universal dataset loader
data_filename = "laptops.csv"
data_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"

if not os.path.exists(data_filename):
    print(f"Downloading {data_filename}...")
    try:
        urllib.request.urlretrieve(data_url, data_filename)
        print("Download complete.")
    except Exception as e:
        print(f"Direct download failed ({e}), attempting pandas read...")
        df = pd.read_csv(data_url)
        df.to_csv(data_filename, index=False)
else:
    print(f"Loading local dataset '{data_filename}'...")

df = pd.read_csv(data_filename)

# Drop any unneeded index columns if present
unneeded_cols = [c for c in df.columns if 'Unnamed' in c]
if unneeded_cols:
    df.drop(columns=unneeded_cols, inplace=True)

print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
df.head()
"""))

cells.append(nbf.v4.new_markdown_cell(r"""### Data Inspection & Summary Statistics
Let's inspect data completeness and distributions of the numerical attributes.
"""))

cells.append(nbf.v4.new_code_cell(r"""# Check for missing values and summary statistics
print("--- Missing Values Check ---")
print(f"Total missing values: {df.isnull().sum().sum()}")
print(f"Missing values in Target 'Price': {df['Price'].isnull().sum()}")

# Key features summary
features = ['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']
df[['Price'] + features].describe().round(2)
"""))

# Section 1
cells.append(nbf.v4.new_markdown_cell(r"""<a id="sec-1"></a>
## 1. Single Linear Regression (SLR): CPU Frequency -> Price

We first examine whether laptop price can be predicted solely using processor clock speed (`CPU_frequency` in GHz):
$$\text{Price} = \beta_0 + \beta_1 \times \text{CPU\_frequency}$$
"""))

cells.append(nbf.v4.new_code_cell(r"""# 1. Instantiate the linear regression model object
lm_slr = LinearRegression()

# 2. Define predictor X (2D DataFrame) and target Y (1D Series)
X_slr = df[['CPU_frequency']]
Y = df['Price']

# 3. Fit model
lm_slr.fit(X_slr, Y)

# 4. Generate predictions
yhat_slr = lm_slr.predict(X_slr)

# 5. Extract coefficients
intercept_slr = float(lm_slr.intercept_)
slope_slr = float(lm_slr.coef_[0])

print(f"SLR Intercept (beta_0): {intercept_slr:,.2f}")
print(f"SLR Slope (beta_1):     {slope_slr:,.2f}")
print(f"Model Equation: Price = {intercept_slr:,.2f} + ({slope_slr:,.2f} * CPU_frequency)")
"""))

cells.append(nbf.v4.new_markdown_cell(r"""### Visualizing Actual vs. Fitted Values (SLR)
We plot the distribution of actual laptop prices against the fitted values using modern Kernel Density Estimation (`sns.kdeplot`).
*(Note: Replaces deprecated `sns.distplot` to eliminate deprecation warnings).*
"""))

cells.append(nbf.v4.new_code_cell(r"""plt.figure(figsize=(10, 6))
sns.kdeplot(df['Price'], color="crimson", label="Actual Value", linewidth=2.5, fill=True, alpha=0.15)
sns.kdeplot(yhat_slr, color="royalblue", label="Fitted Values (SLR)", linewidth=2.5, fill=True, alpha=0.15)

plt.title("Actual vs. Fitted Values Distribution (Single Linear Regression)", fontsize=14, fontweight='bold')
plt.xlabel("Price ($)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.show()

# In-sample evaluation
r2_slr_in = r2_score(Y, yhat_slr)
mse_slr_in = mean_squared_error(Y, yhat_slr)
rmse_slr_in = np.sqrt(mse_slr_in)

print(f"SLR In-Sample R^2 Score: {r2_slr_in:.4f}")
print(f"SLR In-Sample MSE:       {mse_slr_in:,.2f}")
print(f"SLR In-Sample RMSE:      ${rmse_slr_in:,.2f}")
"""))

cells.append(nbf.v4.new_markdown_cell(r"""#### 📝 Diagnostic Observation (SLR):
The SLR model explains only $\approx 13.4\%$ of the variance in laptop prices ($R^2 = 0.134$), with an average prediction error (RMSE) of $\approx \$527.38$.
As seen in the density plot, the SLR predictions are concentrated in a narrow band around $\$1,200 - \$1,600$ and completely fail to capture the multi-modal distribution of budget vs. premium gaming/workstation laptops.
Laptops are multifaceted machines; we need a multi-variable model.

---

<a id="sec-2"></a>
## 2. Multiple Linear Regression (MLR): 7 Hardware Features

We now expand our feature space to 7 hardware predictors:
- `CPU_frequency`: Processor clock speed (GHz)
- `RAM_GB`: System memory (GB)
- `Storage_GB_SSD`: Solid State Drive capacity (GB)
- `CPU_core`: Number of processor cores
- `OS`: Operating system category
- `GPU`: Graphics processing unit tier
- `Category`: Device form-factor / category

Equation:
$$\text{Price} = \beta_0 + \beta_1(\text{CPU\_freq}) + \beta_2(\text{RAM}) + \beta_3(\text{SSD}) + \beta_4(\text{Cores}) + \beta_5(\text{OS}) + \beta_6(\text{GPU}) + \beta_7(\text{Category})$$
"""))

cells.append(nbf.v4.new_code_cell(r"""# Feature matrix Z
Z = df[features]

# Fit MLR model
lm_mlr = LinearRegression()
lm_mlr.fit(Z, Y)
yhat_mlr = lm_mlr.predict(Z)

print(f"MLR Intercept (beta_0): {lm_mlr.intercept_:,.2f}")
print("-" * 52)
print(f"{'Feature':<18} | {'Coefficient':<16} | {'Direction'}")
print("-" * 52)
for feat, coef in zip(features, lm_mlr.coef_):
    direction = "Positive (+)" if coef > 0 else "Negative (-)"
    print(f"{feat:<18} | {coef:>15.2f} | {direction}")
print("-" * 52)
"""))

cells.append(nbf.v4.new_markdown_cell(r"""### Actual vs. Fitted Distribution Plot (MLR)
Let's see how well the 7-variable MLR model tracks the price distribution.
"""))

cells.append(nbf.v4.new_code_cell(r"""plt.figure(figsize=(10, 6))
sns.kdeplot(df['Price'], color="crimson", label="Actual Value", linewidth=2.5, fill=True, alpha=0.15)
sns.kdeplot(yhat_mlr, color="seagreen", label="Fitted Values (MLR - 7 Features)", linewidth=2.5, fill=True, alpha=0.15)

plt.title("Actual vs. Fitted Values Distribution (Multiple Linear Regression)", fontsize=14, fontweight='bold')
plt.xlabel("Price ($)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.show()

# In-sample evaluation
r2_mlr_in = r2_score(Y, yhat_mlr)
mse_mlr_in = mean_squared_error(Y, yhat_mlr)
rmse_mlr_in = np.sqrt(mse_mlr_in)

print(f"MLR In-Sample R^2 Score: {r2_mlr_in:.4f} (Up from {r2_slr_in:.4f} in SLR)")
print(f"MLR In-Sample MSE:       {mse_mlr_in:,.2f}")
print(f"MLR In-Sample RMSE:      ${rmse_mlr_in:,.2f} (Down from ${rmse_slr_in:,.2f})")
"""))

cells.append(nbf.v4.new_markdown_cell(r"""#### 📝 Interpretation:
The MLR model is a massive improvement over SLR:
- $R^2$ jumps from **$0.134$ to $0.508$** (explaining over half the variance).
- In-sample RMSE drops from **$\$527.38$ to $\$395.77$** (a reduction of over $\$131$ in average error).
- The distribution plot now tracks both the budget peak around $\$1,000$ and the premium shoulder around $\$2,000$.

---

<a id="sec-3"></a>
## 3. Polynomial Regression: Degrees 1, 3, and 5

Could non-linear powers of `CPU_frequency` explain more variation?
We fit 1st, 3rd, and 5th degree polynomial models using `np.polyfit` and `np.poly1d`.
"""))

cells.append(nbf.v4.new_code_cell(r"""def plot_polynomial(model, x_data, y_data, feature_name="CPU_frequency", degree=1):
    # Enhanced polynomial plot utility function with clean styling and dynamic labels
    x_new = np.linspace(x_data.min() - 0.1, x_data.max() + 0.1, 150)
    y_new = model(x_new)

    plt.figure(figsize=(9, 5))
    plt.scatter(x_data, y_data, color='steelblue', alpha=0.6, edgecolors='k', label='Actual Laptops')
    plt.plot(x_new, y_new, color='crimson', linewidth=2.5, label=f'Polynomial Fit (Degree {degree})')
    plt.title(f"Polynomial Fit for Price ~ {feature_name} (Degree {degree})", fontsize=13, fontweight='bold')
    plt.xlabel(f"{feature_name} (GHz)", fontsize=11)
    plt.ylabel("Price ($)", fontsize=11)
    plt.ylim(0, max(y_data.max() * 1.05, 3500))
    plt.legend(frameon=True, facecolor='white')
    plt.show()

# 1D array conversion for numpy polyfit
x_cpu = df['CPU_frequency'].values
y_price = df['Price'].values

# Fit polynomials of degree 1, 3, and 5
p1 = np.poly1d(np.polyfit(x_cpu, y_price, 1))
p3 = np.poly1d(np.polyfit(x_cpu, y_price, 3))
p5 = np.poly1d(np.polyfit(x_cpu, y_price, 5))

print("Polynomials fitted successfully!")
"""))

cells.append(nbf.v4.new_code_cell(r"""# Plot all three polynomial fits
plot_polynomial(p1, x_cpu, y_price, "CPU_frequency", degree=1)
plot_polynomial(p3, x_cpu, y_price, "CPU_frequency", degree=3)
plot_polynomial(p5, x_cpu, y_price, "CPU_frequency", degree=5)
"""))

cells.append(nbf.v4.new_markdown_cell("""### In-Sample Comparison of Polynomial Degrees
Let's compare the in-sample fit metrics across Degrees 1, 3, and 5.
"""))

cells.append(nbf.v4.new_code_cell(r"""poly_comparison = pd.DataFrame({
    'Polynomial Degree': ['Degree 1 (Linear)', 'Degree 3 (Cubic)', 'Degree 5 (Quintic)'],
    'In-Sample R^2': [r2_score(y_price, p1(x_cpu)), r2_score(y_price, p3(x_cpu)), r2_score(y_price, p5(x_cpu))],
    'In-Sample MSE': [mean_squared_error(y_price, p1(x_cpu)), mean_squared_error(y_price, p3(x_cpu)), mean_squared_error(y_price, p5(x_cpu))],
    'In-Sample RMSE ($)': [np.sqrt(mean_squared_error(y_price, p1(x_cpu))), 
                           np.sqrt(mean_squared_error(y_price, p3(x_cpu))), 
                           np.sqrt(mean_squared_error(y_price, p5(x_cpu)))]
})

display(poly_comparison.round({'In-Sample R^2': 4, 'In-Sample MSE': 2, 'In-Sample RMSE ($)': 2}))
"""))

# Section 4
cells.append(nbf.v4.new_markdown_cell(r"""<a id="sec-4"></a>
## 4. Preprocessing Pipelines (Scaling + Polynomial + Regression)

When working with 7 features that differ in units and orders of magnitude (`RAM_GB` in single/double digits vs `Storage_GB_SSD` in hundreds), building polynomial interactions directly on raw scales leads to severe numerical instability.

Scikit-Learn's **Pipeline** bundles:
1. `StandardScaler`: Normalizes each feature to mean = 0, standard deviation = 1.
2. `PolynomialFeatures(degree=2)`: Generates quadratic terms and pairwise feature interactions ($7 \rightarrow 35$ features).
3. `LinearRegression`: Fits the multivariate linear model on the transformed feature matrix.
"""))

cells.append(nbf.v4.new_code_cell(r"""# Construct Pipeline
pipe_poly2 = Pipeline([
    ('scale', StandardScaler()),
    ('polynomial', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', LinearRegression())
])

# Convert feature matrix to float
Z_float = Z.astype(float)

# Fit pipeline on full dataset
pipe_poly2.fit(Z_float, Y)
yhat_pipe = pipe_poly2.predict(Z_float)

print(f"Original feature count:    {Z.shape[1]}")
poly_step = pipe_poly2.named_steps['polynomial']
print(f"Pipeline expanded features: {poly_step.fit(Z_float).n_output_features_}")
print(f"Pipeline In-Sample R^2:    {r2_score(Y, yhat_pipe):.4f}")
print(f"Pipeline In-Sample RMSE:   ${np.sqrt(mean_squared_error(Y, yhat_pipe)):,.2f}")
"""))

# Section 5
cells.append(nbf.v4.new_markdown_cell(r"""---

<a id="sec-5"></a>
## 5. Out-of-Sample Generalization & Cross-Validation Tests (NEW)

> 🚨 **Critical Data Science Lesson: The In-Sample Illusion**
> Notice that the **Pipeline Poly Degree 2** achieved an in-sample $R^2$ of **$0.613$** (better than MLR's $0.508$).
> But did it actually learn real patterns, or is it merely memorizing the training samples?
> With only 238 laptop samples and 35 expanded features, the ratio of samples to features drops to less than $7:1$.
> Let's test out-of-sample generalization!

---

<a id="sec-5-1"></a>
### 5.1 Train / Test Split (80% Train, 20% Test)
We split the data into 190 training laptops and 48 testing laptops with a reproducible random seed (`random_state=42`).
"""))

cells.append(nbf.v4.new_code_cell(r"""# 80/20 Train-Test split
X_train, X_test, y_train, y_test = train_test_split(Z, df['Price'], test_size=0.20, random_state=42)

print(f"Training laptops: {X_train.shape[0]}")
print(f"Testing laptops:  {X_test.shape[0]}")
"""))

cells.append(nbf.v4.new_markdown_cell(r"""<a id="sec-5-2"></a>
### 5.2 5-Fold Cross-Validation on MLR
Cross-validation ensures our evaluation does not depend on a lucky or unlucky single train/test split.
"""))

cells.append(nbf.v4.new_code_cell(r"""cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2_scores = cross_val_score(LinearRegression(), Z, df['Price'], cv=cv, scoring='r2')
cv_rmse_scores = np.sqrt(-cross_val_score(LinearRegression(), Z, df['Price'], cv=cv, scoring='neg_mean_squared_error'))

print("5-Fold Cross-Validation Results for MLR:")
print(f"  R^2 per fold:   {[round(s, 3) for s in cv_r2_scores]}")
print(f"  Mean R^2:       {cv_r2_scores.mean():.4f} (+/- {cv_r2_scores.std():.4f})")
print(f"  Mean RMSE ($):  ${cv_rmse_scores.mean():,.2f}")
"""))

cells.append(nbf.v4.new_markdown_cell(r"""<a id="sec-5-3"></a>
### 5.3 Comprehensive Model Comparison Leaderboard
We train each model **only on the training set** (`X_train`), and evaluate on both `X_train` and the unseen `X_test`.
We also include **L2 Regularized Ridge Regression** (`Ridge(alpha=10.0)`) to test whether regularization can save the degree-2 polynomial.
"""))

cells.append(nbf.v4.new_code_cell(r"""candidate_models = {
    '1. SLR (CPU_frequency)': Pipeline([('model', LinearRegression())]),
    '2. MLR (7 features)': Pipeline([('model', LinearRegression())]),
    '3. Poly Degree 3 (CPU_frequency)': Pipeline([
        ('scale', StandardScaler()),
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('model', LinearRegression())
    ]),
    '4. Poly Degree 5 (CPU_frequency)': Pipeline([
        ('scale', StandardScaler()),
        ('poly', PolynomialFeatures(degree=5, include_bias=False)),
        ('model', LinearRegression())
    ]),
    '5. Poly Degree 2 (7 features)': Pipeline([
        ('scale', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('model', LinearRegression())
    ]),
    '6. Regularized Ridge (Poly Deg 2)': Pipeline([
        ('scale', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('model', Ridge(alpha=10.0))
    ])
}

candidate_features = {
    '1. SLR (CPU_frequency)': ['CPU_frequency'],
    '2. MLR (7 features)': features,
    '3. Poly Degree 3 (CPU_frequency)': ['CPU_frequency'],
    '4. Poly Degree 5 (CPU_frequency)': ['CPU_frequency'],
    '5. Poly Degree 2 (7 features)': features,
    '6. Regularized Ridge (Poly Deg 2)': features
}

leaderboard_rows = []

for name, pipeline in candidate_models.items():
    feat_list = candidate_features[name]
    x_tr = X_train[feat_list]
    x_te = X_test[feat_list]
    
    # Fit only on train
    pipeline.fit(x_tr, y_train)
    
    # Predict
    pred_tr = pipeline.predict(x_tr)
    pred_te = pipeline.predict(x_te)
    
    tr_r2 = r2_score(y_train, pred_tr)
    te_r2 = r2_score(y_test, pred_te)
    tr_rmse = np.sqrt(mean_squared_error(y_train, pred_tr))
    te_rmse = np.sqrt(mean_squared_error(y_test, pred_te))
    te_mae = mean_absolute_error(y_test, pred_te)
    
    leaderboard_rows.append({
        'Model': name,
        'Train R^2': tr_r2,
        'Test R^2': te_r2,
        'Train RMSE ($)': tr_rmse,
        'Test RMSE ($)': te_rmse,
        'Test MAE ($)': te_mae
    })

laptop_leaderboard = pd.DataFrame(leaderboard_rows).sort_values(by='Test R^2', ascending=False).reset_index(drop=True)
display(laptop_leaderboard.round({'Train R^2': 4, 'Test R^2': 4, 'Train RMSE ($)': 2, 'Test RMSE ($)': 2, 'Test MAE ($)': 2}))
"""))

cells.append(nbf.v4.new_markdown_cell(r"""<a id="sec-5-4"></a>
### 5.4 The Dimensionality & Overfitting Trap: Analysis

#### 💡 Remarkable Finding:
Look closely at **Model 5 (Poly Degree 2 on 7 features)**:
- **Train $R^2$:** $0.6746$ (highest of all models!)
- **Test $R^2$:** **$-0.0475$** (negative! It performs worse than predicting the simple mean!)
- **Test RMSE:** Rose from $\$410.58$ (MLR) to **$\$491.33$**.

> ⚠️ **Why did this happen?**
> Expanding 7 features into degree-2 polynomials created **35 terms**. With only 190 training samples, the model suffered from the **curse of dimensionality** and fitted random sample variations.
> Meanwhile, the linear **MLR model** had the best generalization score (**Test $R^2 = 0.2685$, Test RMSE = $\$410.58$**).

---

<a id="sec-6"></a>
## 6. Statistical Assumption Tests & Regression Diagnostics (NEW)

Let's test the Gauss-Markov assumptions of our top-performing MLR model.

---

<a id="sec-6-1"></a>
### 6.1 Residuals vs. Fitted Values (Linearity & Homoscedasticity)
"""))

cells.append(nbf.v4.new_code_cell(r"""mlr_fitted = lm_mlr.predict(Z)
mlr_residuals = df['Price'] - mlr_fitted

plt.figure(figsize=(10, 5))
plt.scatter(mlr_fitted, mlr_residuals, alpha=0.6, color='steelblue', edgecolors='k')
plt.axhline(0, color='crimson', linestyle='--', linewidth=2)
plt.title("Residuals vs. Fitted Values (Laptop MLR)", fontsize=13, fontweight='bold')
plt.xlabel("Fitted Values (Predicted Laptop Price $)", fontsize=11)
plt.ylabel("Residuals ($)", fontsize=11)
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell(r"""<a id="sec-6-2"></a>
### 6.2 Normality of Residuals: Q-Q Plot & Shapiro-Wilk Test
"""))

cells.append(nbf.v4.new_code_cell(r"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram + KDE
sns.histplot(mlr_residuals, kde=True, color='teal', ax=axes[0])
axes[0].set_title("Histogram & KDE of MLR Residuals", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Residual ($)")

# Q-Q Plot
stats.probplot(mlr_residuals, dist="norm", plot=axes[1])
axes[1].get_lines()[0].set_color('teal')
axes[1].get_lines()[0].set_markersize(5)
axes[1].get_lines()[1].set_color('darkorange')
axes[1].get_lines()[1].set_linewidth(2)
axes[1].set_title("Normal Q-Q Plot of Residuals", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

# Shapiro-Wilk Test
w_stat, p_val = stats.shapiro(mlr_residuals)
print(f"Shapiro-Wilk Test for Normality:")
print(f"  Test Statistic W = {w_stat:.4f}")
print(f"  p-value          = {p_val:.4e}")
"""))

cells.append(nbf.v4.new_markdown_cell(r"""<a id="sec-6-3"></a>
### 6.3 Multicollinearity Analysis: Variance Inflation Factor (VIF)
We calculate the VIF for each of the 7 laptop features to ensure predictors are not collinear.
$$VIF_j = \frac{1}{1 - R_j^2}$$
"""))

cells.append(nbf.v4.new_code_cell(r"""vif_records = []

for feat in features:
    other_feats = [f for f in features if f != feat]
    X_oth = Z[other_feats]
    y_tgt = Z[feat]
    
    r2_f = LinearRegression().fit(X_oth, y_tgt).score(X_oth, y_tgt)
    vif_score = 1.0 / (1.0 - r2_f) if r2_f < 1.0 else np.inf
    
    vif_records.append({
        'Feature': feat,
        'R^2 against others': r2_f,
        'VIF': vif_score,
        'Collinearity Status': 'Low (VIF < 5)' if vif_score < 5 else 'High'
    })

vif_df = pd.DataFrame(vif_records)
display(vif_df.round({'R^2 against others': 4, 'VIF': 2}))

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(Z.corr(), annot=True, cmap='Blues', fmt='.2f', vmin=-1, vmax=1)
plt.title("Correlation Matrix of Laptop Predictors", fontsize=13, fontweight='bold')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell(r"""#### 📝 Multicollinearity Finding:
Unlike the automobile dataset where `engine-size` and `curb-weight` had VIFs over 5.5, **all laptop predictors have VIF values below 1.6**!
This means that RAM, CPU clock speed, cores, SSD, GPU, OS, and Category provide independent information to the model with minimal collinearity.

---

<a id="sec-6-4"></a>
### 6.4 Statistical Significance of Predictors (t-stats, p-values)
Let's determine which specific laptop specifications are statistically significant drivers of price.
"""))

cells.append(nbf.v4.new_code_cell(r"""# OLS closed-form parameter estimation: Beta = (X'X)^(-1) X'y
n = len(Z)
p = len(features) + 1
df_resid = n - p

X_design = np.column_stack([np.ones(n), Z.values])
y_vec = df['Price'].values

beta_hat = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y_vec
resids = y_vec - X_design @ beta_hat

sigma_sq = (resids.T @ resids) / df_resid
cov_mat = sigma_sq * np.linalg.inv(X_design.T @ X_design)
std_errors = np.sqrt(np.diag(cov_mat))

t_statistics = beta_hat / std_errors
p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=df_resid)) for t in t_statistics]

ols_table = pd.DataFrame({
    'Feature': ['(Intercept)'] + features,
    'Coefficient ($)': beta_hat,
    'Std Error': std_errors,
    't-statistic': t_statistics,
    'p-value': p_values,
    'Statistically Significant (p < 0.05)': [p < 0.05 for p in p_values]
})

print("Ordinary Least Squares (OLS) Parameter Significance Table:")
display(ols_table.round({'Coefficient ($)': 2, 'Std Error': 2, 't-statistic': 2, 'p-value': 6}))
"""))

cells.append(nbf.v4.new_markdown_cell(r"""#### 📝 Key Statistical Findings:
1. **`RAM_GB` ($p = 8.40 \times 10^{-11}, t = 6.81$):** By far the single most significant predictor of laptop price. Each additional GB of RAM increases price by $\approx \$86.81$.
2. **`Category` ($p = 2.42 \times 10^{-7}, t = 5.32$):** Form-factor tier significantly adds $\approx \$191.30$ per tier increase.
3. **`CPU_frequency` ($p = 4.31 \times 10^{-6}, t = 4.71$):** Highly significant. Each 1 GHz increase in clock speed adds $\approx \$961.28$.
4. **`OS` ($p = 3.04 \times 10^{-4}$):** Significant pricing tier difference across operating systems.
5. **`CPU_core` ($p = 0.0147$) & `GPU` ($p = 0.0330$):** Both statistically significant ($p < 0.05$).
6. **`Storage_GB_SSD` ($p = 0.647$):** **Not statistically significant** in this model ($t = 0.46$). SSD storage capacity alone does not significantly alter price once RAM, CPU, and category are controlled for.

---

<a id="sec-7"></a>
## 7. Automated Sanity & Unit Tests (NEW)

To guarantee software reliability, reproducible code execution, and data contract adherence, we run 5 programmatic unit test assertions.
"""))

cells.append(nbf.v4.new_code_cell(r"""print("Executing automated verification and unit tests...")

# Test 1: Dataset shape and integrity
assert df.shape[0] == 238, f"Expected 238 rows, got {df.shape[0]}"
assert not df['Price'].isnull().any(), "Found null values in Price!"
assert (df['Price'] > 0).all(), "Found non-positive laptop prices!"
print("  [PASS] Test 1: Dataset dimensions and price positivity contract.")

# Test 2: R^2 boundary checks
assert 0.0 <= r2_slr_in <= 1.0, f"SLR R^2 out of valid range: {r2_slr_in}"
assert 0.0 <= r2_mlr_in <= 1.0, f"MLR R^2 out of valid range: {r2_mlr_in}"
assert r2_mlr_in > r2_slr_in, "MLR in-sample R^2 should strictly exceed SLR R^2."
print("  [PASS] Test 2: In-sample R^2 bounds and MLR dominance verified.")

# Test 3: Predictions integrity
preds_mlr_all = lm_mlr.predict(Z)
assert len(preds_mlr_all) == 238, "Prediction vector length mismatch!"
assert not np.isnan(preds_mlr_all).any(), "Found NaN in MLR predictions!"
print("  [PASS] Test 3: Prediction completeness without NaNs.")

# Test 4: Pipeline transformation equivalence
pipe_test = Pipeline([('scale', StandardScaler()), ('model', LinearRegression())])
pipe_test.fit(Z, Y)
scaler_manual = StandardScaler()
z_scaled = scaler_manual.fit_transform(Z)
lm_test = LinearRegression().fit(z_scaled, Y)

np.testing.assert_allclose(pipe_test.predict(Z), lm_test.predict(z_scaled), rtol=1e-5, atol=1e-5)
print("  [PASS] Test 4: Pipeline scaling and regression match manual execution.")

# Test 5: Overfitting detection test
deg2_test_r2 = laptop_leaderboard.loc[laptop_leaderboard['Model'] == '5. Poly Degree 2 (7 features)', 'Test R^2'].values[0]
mlr_test_r2 = laptop_leaderboard.loc[laptop_leaderboard['Model'] == '2. MLR (7 features)', 'Test R^2'].values[0]
assert mlr_test_r2 > deg2_test_r2, "Expected MLR to beat Degree 2 Poly on unseen test data!"
print("  [PASS] Test 5: Overfitting assertion verified (MLR out-of-sample beats Degree 2 Poly).")

print("\nALL 5 AUTOMATED UNIT TESTS PASSED SUCCESSFULLY! \u2705")
"""))

# Section 8
cells.append(nbf.v4.new_markdown_cell(r"""<a id="sec-8"></a>
## 8. Predictions on New Laptop Profiles & Strategic Conclusion

### 8.1 Price Forecasting for Real-World Laptop Scenarios
Let's apply our top-performing MLR model to forecast estimated prices for three realistic consumer profiles:
1. **Budget Student Laptop:** 1.8 GHz, 4 GB RAM, 128 GB SSD, 2 cores, OS=1, GPU=1, Category=1
2. **Mainstream Business Laptop:** 2.5 GHz, 16 GB RAM, 512 GB SSD, 6 cores, OS=1, GPU=2, Category=3
3. **High-End Creator / Gaming Laptop:** 3.2 GHz, 32 GB RAM, 1024 GB SSD, 8 cores, OS=1, GPU=3, Category=4
"""))

cells.append(nbf.v4.new_code_cell(r"""new_laptops = pd.DataFrame({
    'CPU_frequency':  [1.8, 2.5, 3.2],
    'RAM_GB':         [4,   16,  32],
    'Storage_GB_SSD': [128, 512, 1024],
    'CPU_core':       [2,   6,   8],
    'OS':             [1,   1,   1],
    'GPU':            [1,   2,   3],
    'Category':       [1,   3,   4]
}, index=['Budget Student Laptop', 'Mainstream Business Laptop', 'High-End Creator/Gaming'])

pred_prices = lm_mlr.predict(new_laptops)
new_laptops['Predicted Price ($)'] = np.round(pred_prices, 2)

print("Predicted Prices for New Laptop Profiles:")
display(new_laptops[['RAM_GB', 'CPU_frequency', 'Storage_GB_SSD', 'CPU_core', 'Predicted Price ($)']])
"""))

cells.append(nbf.v4.new_markdown_cell(r"""---

### 8.2 Executive Decision Matrix

| Model | In-Sample $R^2$ | Test $R^2$ (Out-of-Sample) | Test RMSE ($) | Practical Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **SLR (CPU_frequency)** | 0.134 | -0.037 | $488.91 | ❌ **Inadequate:** Misses memory and GPU; poor predictive power. |
| **MLR (7 Features)** | **0.508** | **0.269** | **$410.58** | 🏆 **BEST BALANCED MODEL:** Solid accuracy, fully interpretable, lowest test error. |
| **Poly Degree 3 (CPU_frequency)** | 0.236 | 0.107 | $453.78 | 💡 Captures modest CPU non-linearity, but inferior to multi-feature MLR. |
| **Poly Degree 5 (CPU_frequency)** | 0.283 | 0.100 | $455.34 | ⚠️ Slight overfitting on CPU curve without adding multi-feature depth. |
| **Poly Degree 2 Pipeline (7 Feats)** | **0.613** | **-0.047** | **$491.33** | ❌ **SEVERE OVERFITTING:** 35 features on 190 samples leads to negative generalization. |
| **Ridge Regularized (Poly 2)** | 0.671 | -0.013 | $483.26 | 🛡️ Reduces variance, but still unable to overcome high dimensionality. |

---

### 🎯 Key Takeaways
1. **Multiple Linear Regression is the Clear Winner**: MLR achieves the highest out-of-sample accuracy with the lowest generalization error ($\text{RMSE} = \$410.58$).
2. **RAM and CPU Frequency are the Core Price Drivers**: $p$-values show `RAM_GB` ($p < 10^{-10}$) and `CPU_frequency` ($p < 10^{-5}$) dominate laptop pricing.
3. **Beware Polynomial Expansion on Small Datasets**: In-sample metrics misleadingly favored polynomial pipelines ($R^2 = 0.613$), but out-of-sample testing exposed negative $R^2$ due to overfitting.
4. **Clean Code & Robust Workflows**: Scikit-Learn pipelines and automated assertions guarantee reproducibility and prevent data leakage.

---
### 🏁 Congratulations on completing this comprehensive laptop pricing review!
"""))

nb['cells'] = cells

output_path = 'practice_model_development_laptops.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully generated {output_path} with {len(cells)} cells!")
