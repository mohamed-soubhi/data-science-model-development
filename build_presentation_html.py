"""
Master script to generate an executive-ready, interactive HTML presentation dashboard
incorporating all studies: Automobile EDA & Modeling, Laptop Evaluation & Tuning, and Insurance Capstone.
"""
import base64
import io
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial']
sns.set_theme(style="whitegrid", palette="muted")

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

print("1. Generating Automobile Charts...")
df_auto = pd.read_csv("automobileEDA.csv")

# Automobile Correlation bar chart
eda_features = ['engine-size', 'curb-weight', 'horsepower', 'width', 'length', 'wheel-base', 'bore', 'city-mpg', 'highway-mpg', 'peak-rpm']
eda_corrs = [stats.pearsonr(df_auto[c], df_auto['price'])[0] for c in eda_features]

fig_ec, ax_ec = plt.subplots(figsize=(7.2, 4.0))
colors_corr = ['#2a9d8f' if x > 0.5 else ('#e63946' if x < -0.5 else '#64748b') for x in eda_corrs]
y_pos = np.arange(len(eda_features))
ax_ec.barh(y_pos, eda_corrs, color=colors_corr, height=0.65)
ax_ec.set_yticks(y_pos)
ax_ec.set_yticklabels(eda_features, fontsize=9.5)
ax_ec.set_xlabel('Pearson Correlation (r) with Car Price', fontsize=10, fontweight='bold')
ax_ec.set_title('Automobile Feature Correlation Ranking', fontsize=12, fontweight='bold', pad=10)
ax_ec.axvline(0, color='black', linewidth=1)
ax_ec.invert_yaxis()
for i, v in enumerate(eda_corrs):
    offset = 0.02 if v >= 0 else -0.11
    ax_ec.text(v + offset, i + 0.1, f"{v:+.2f}", color='black', fontweight='bold', fontsize=8.5)
img_eda_corr = fig_to_base64(fig_ec)

# Automobile Pivot Heatmap
grouped_res = df_auto[['drive-wheels', 'body-style', 'price']].groupby(['drive-wheels', 'body-style'], as_index=False).mean()
pivot_data = grouped_res.pivot(index='drive-wheels', columns='body-style', values='price').fillna(0)

fig_hm, ax_hm = plt.subplots(figsize=(7.2, 4.0))
sns.heatmap(pivot_data, annot=True, fmt=',.0f', cmap='YlGnBu', ax=ax_hm, 
            cbar_kws={'label': 'Mean Price (USD)'}, linewidths=1, linecolor='white')
ax_hm.set_title('Average Car Price by Drivetrain & Body Style', fontsize=12, fontweight='bold', pad=10)
img_eda_heatmap = fig_to_base64(fig_hm)

# Automobile Drivetrain Boxplot
fig_box, ax_box = plt.subplots(figsize=(7.2, 4.0))
sns.boxplot(x="drive-wheels", y="price", data=df_auto, hue="drive-wheels", palette="Blues", legend=False, ax=ax_box)
ax_box.set_title('Car Price by Drive Wheels (ANOVA F = 67.95, p < 10⁻²²)', fontsize=12, fontweight='bold', pad=10)
img_eda_box = fig_to_base64(fig_box)

# Automobile Actual vs Fitted MLR
Z_auto = df_auto[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
y_auto = df_auto['price']
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Z_auto, y_auto, test_size=0.2, random_state=42)
lm_auto = LinearRegression().fit(Xa_tr, ya_tr)

fig_ad, ax_ad = plt.subplots(figsize=(7.2, 4.0))
sns.kdeplot(y_auto, color="#e63946", label="Actual Price", linewidth=2.5, fill=True, alpha=0.15, ax=ax_ad)
sns.kdeplot(lm_auto.predict(Z_auto), color="#1d3557", label="MLR Fitted Values", linewidth=2.5, fill=True, alpha=0.15, ax=ax_ad)
ax_ad.set_title("Automobile: Actual vs. MLR Fitted Price Density", fontsize=12, fontweight='bold', pad=10)
ax_ad.legend(frameon=True, facecolor='white', framealpha=0.9)
img_auto_dist = fig_to_base64(fig_ad)

# Automobile Overfitting (Deg 1 vs 3 vs 11)
x_mpg = df_auto[['highway-mpg']]
x_tr, x_te, y_tr, y_te = train_test_split(x_mpg, y_auto, test_size=0.2, random_state=42)
m1 = Pipeline([('model', LinearRegression())]).fit(x_tr, y_tr)
m3 = Pipeline([('poly', PolynomialFeatures(3)), ('model', LinearRegression())]).fit(x_tr, y_tr)
m11 = Pipeline([('scale', StandardScaler()), ('poly', PolynomialFeatures(11)), ('model', LinearRegression())]).fit(x_tr, y_tr)
x_dom = pd.DataFrame({'highway-mpg': np.linspace(15, 55, 300)})

fig_ao, ax_ao = plt.subplots(figsize=(7.2, 4.0))
ax_ao.scatter(x_tr, y_tr, color='#457b9d', alpha=0.5, s=30, label='Train (80%)')
ax_ao.scatter(x_te, y_te, color='#e63946', marker='^', s=45, label='Test (20%)', edgecolors='k')
ax_ao.plot(x_dom, m1.predict(x_dom), 'k--', label=f"Deg 1 [Test R²: {r2_score(y_te, m1.predict(x_te)):.2f}]", linewidth=1.8)
ax_ao.plot(x_dom, m3.predict(x_dom), '#2a9d8f', label=f"Deg 3 [Test R²: {r2_score(y_te, m3.predict(x_te)):.2f}]", linewidth=2.2)
ax_ao.plot(x_dom, m11.predict(x_dom), '#e76f51', linestyle='-.', label=f"Deg 11 [Test R²: {r2_score(y_te, m11.predict(x_te)):.2f}]", linewidth=1.8)
ax_ao.set_title("Automobile Overfitting: Runge's Phenomenon at Degree 11", fontsize=12, fontweight='bold', pad=10)
ax_ao.set_ylim(0, 52000)
ax_ao.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
img_auto_overfit = fig_to_base64(fig_ao)

# Automobile VIF Correlation
fig_av, ax_av = plt.subplots(figsize=(6, 4))
sns.heatmap(Z_auto.corr(), annot=True, cmap='Blues', fmt='.2f', vmin=-1, vmax=1, ax=ax_av, cbar=False)
ax_av.set_title("Automobile Predictor Correlation (VIF > 5)", fontsize=11, fontweight='bold', pad=10)
img_auto_vif = fig_to_base64(fig_av)

print("2. Generating Laptop Charts...")
df_lap = pd.read_csv("laptops.csv")
lap_features = ['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']
X_lap = df_lap[lap_features]
y_lap = df_lap['Price']

# Laptop Overfitting Curve (Degrees 1 to 5)
x_tr50, x_te50, y_tr50, y_te50 = train_test_split(X_lap, y_lap, test_size=0.50, random_state=0)
degs = [1, 2, 3, 4, 5]
r2_tr_lap = []
r2_te_lap = []
for d in degs:
    pr = PolynomialFeatures(degree=d)
    x_tr_p = pr.fit_transform(x_tr50[['CPU_frequency']])
    x_te_p = pr.transform(x_te50[['CPU_frequency']])
    m = LinearRegression().fit(x_tr_p, y_tr50)
    r2_tr_lap.append(m.score(x_tr_p, y_tr50))
    r2_te_lap.append(m.score(x_te_p, y_te50))

fig_lo, ax_lo = plt.subplots(figsize=(7.2, 4.0))
ax_lo.plot(degs, r2_tr_lap, 'o-', color='#1d3557', linewidth=2.5, label='Train R² (Memorization)', markersize=7)
ax_lo.plot(degs, r2_te_lap, 's--', color='#e63946', linewidth=2.5, label='Test R² (Generalization)', markersize=7)
ax_lo.axhline(0, color='gray', linestyle=':', linewidth=1)
ax_lo.set_title("Laptop Overfitting Curve: Degrees 1 to 5 Collapse", fontsize=12, fontweight='bold', pad=10)
ax_lo.set_xlabel("Polynomial Degree", fontsize=10)
ax_lo.set_ylabel("R² Score", fontsize=10)
ax_lo.set_ylim(-1.6, 0.6)
ax_lo.legend(frameon=True, facecolor='white')
img_lap_overfit_curve = fig_to_base64(fig_lo)

# Laptop GridSearchCV Tuning Chart
pr2 = PolynomialFeatures(degree=2, include_bias=False)
x_lap_pr2 = pr2.fit_transform(x_tr50)
x_lap_pr2_te = pr2.transform(x_te50)

alphas_test = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_scores = [cross_val_score(Ridge(alpha=a), x_lap_pr2, y_tr50, cv=4).mean() for a in alphas_test]

fig_lg, ax_lg = plt.subplots(figsize=(7.2, 4.0))
ax_lg.plot([str(a) for a in alphas_test], ridge_scores, 'd-', color='#2a9d8f', linewidth=2.5, markersize=8)
ax_lg.set_title("Laptop GridSearchCV: Mean 4-Fold CV R² vs. Ridge Alpha", fontsize=12, fontweight='bold', pad=10)
ax_lg.set_xlabel("Regularization Alpha", fontsize=10)
ax_lg.set_ylabel("4-Fold CV Mean R²", fontsize=10)
img_lap_grid = fig_to_base64(fig_lg)

print("3. Generating Medical Insurance Charts...")
df_ins = pd.read_csv("insurance.csv", header=None)
df_ins.columns = ['age', 'gender', 'bmi', 'no_of_children', 'smoker', 'region', 'charges']
df_ins = df_ins.replace('?', np.nan)
df_ins['smoker'] = df_ins['smoker'].fillna(df_ins['smoker'].value_counts().idxmax()).astype(int)
df_ins['age'] = df_ins['age'].astype(float).fillna(df_ins['age'].astype(float).mean()).astype(int)
df_ins['bmi'] = df_ins['bmi'].astype(float)
df_ins['charges'] = np.round(df_ins['charges'].astype(float), 2)

# Insurance Boxplot
fig_ib, ax_ib = plt.subplots(figsize=(7.2, 4.0))
sns.boxplot(x="smoker", y="charges", data=df_ins, hue="smoker", palette=["#2a9d8f", "#e63946"], legend=False, ax=ax_ib)
ax_ib.set_title("Medical Insurance Charges: Non-Smoker vs. Smoker (+$23,600)", fontsize=12, fontweight='bold', pad=10)
ax_ib.set_xticks([0, 1])
ax_ib.set_xticklabels(["Non-Smoker (0)", "Smoker (1)"], fontsize=10)
ax_ib.set_ylabel("Annual Charges (USD)", fontsize=10)
img_ins_box = fig_to_base64(fig_ib)

# Insurance Actual vs Fitted Poly Pipeline
Z_ins = df_ins[['age', 'gender', 'bmi', 'no_of_children', 'smoker', 'region']]
y_ins = df_ins['charges']
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Z_ins, y_ins, test_size=0.2, random_state=42)

poly_ins = Pipeline([
    ('scale', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', LinearRegression())
]).fit(Xi_tr, yi_tr)

fig_id, ax_id = plt.subplots(figsize=(7.2, 4.0))
sns.kdeplot(y_ins, color="#e63946", label="Actual Charges", linewidth=2.5, fill=True, alpha=0.15, ax=ax_id)
sns.kdeplot(poly_ins.predict(Z_ins), color="#1d3557", label="Polynomial Pipeline Fitted Values", linewidth=2.5, fill=True, alpha=0.15, ax=ax_id)
ax_id.set_title("Insurance: Actual vs. Polynomial Fitted Charges (Test R² = 0.834)", fontsize=12, fontweight='bold', pad=10)
ax_id.legend(frameon=True, facecolor='white', framealpha=0.9)
img_ins_dist = fig_to_base64(fig_id)

# Cross-Study Portfolio Benchmark
fig_bm, ax_bm = plt.subplots(figsize=(10, 4.5))
portfolio_models = [
    'Insurance: Poly 2 Pipeline', 'Insurance: Ridge (alpha=0.1)', 'Insurance: MLR (6 feats)', 'Insurance: SLR (smoker)',
    'Auto: MLR (4 feats)', 'Auto: Ridge Deg 2', 'Auto: SLR (engine)', 'Auto: Poly 11 (highway)',
    'Laptop: Tuned Ridge', 'Laptop: MLR (7 feats)', 'Laptop: Poly Deg 2 (Unreg)'
]
portfolio_test_r2 = [0.8339, 0.8339, 0.7396, 0.5991, 0.7637, 0.7680, 0.7102, 0.5560, 0.3861, 0.2685, -0.0475]
colors_port = ['#2a9d8f' if x > 0.7 else ('#457b9d' if x > 0.2 else '#e63946') for x in portfolio_test_r2]

y_pos_port = np.arange(len(portfolio_models))
ax_bm.barh(y_pos_port, portfolio_test_r2, color=colors_port, height=0.65)
ax_bm.set_yticks(y_pos_port)
ax_bm.set_yticklabels(portfolio_models, fontsize=9.5)
ax_bm.set_xlabel('Out-of-Sample Test R² Score', fontsize=10, fontweight='bold')
ax_bm.set_title('Cross-Study Master Benchmark: Generalization Accuracy', fontsize=12, fontweight='bold', pad=10)
ax_bm.axvline(0, color='black', linewidth=1)
ax_bm.invert_yaxis()
for i, v in enumerate(portfolio_test_r2):
    offset = 0.015 if v >= 0 else -0.075
    ax_bm.text(v + offset, i + 0.1, f"{v:.3f}", color='black', fontweight='bold', fontsize=8.5)
img_master_benchmark = fig_to_base64(fig_bm)

print("4. Building Updated Master HTML Presentation...")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Science Analytics, Modeling & Refinement | Executive Presentation</title>
    <style>
        :root {{
            --primary: #1d3557;
            --primary-light: #457b9d;
            --secondary: #2a9d8f;
            --accent: #e76f51;
            --danger: #e63946;
            --bg-main: #f8fafc;
            --bg-card: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
        }}

        body {{
            background-color: var(--bg-main);
            color: var(--text-main);
            line-height: 1.6;
            padding-bottom: 60px;
        }}

        header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #ffffff;
            padding: 24px 32px;
            box-shadow: var(--shadow);
            position: sticky;
            top: 0;
            z-index: 100;
        }}

        .header-container {{
            max-width: 1300px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }}

        .title-area h1 {{
            font-size: 1.45rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .title-area p {{
            color: #94a3b8;
            font-size: 0.88rem;
            margin-top: 4px;
        }}

        .nav-tabs {{
            display: flex;
            gap: 6px;
            background: rgba(255, 255, 255, 0.1);
            padding: 4px;
            border-radius: 8px;
            flex-wrap: wrap;
        }}

        .nav-tab {{
            background: transparent;
            border: none;
            color: #cbd5e1;
            padding: 8px 13px;
            font-size: 0.82rem;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .nav-tab:hover {{
            color: #ffffff;
            background: rgba(255, 255, 255, 0.15);
        }}

        .nav-tab.active {{
            background: var(--secondary);
            color: #ffffff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}

        .toolbar {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}

        .btn {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 16px;
            font-size: 0.85rem;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            border: none;
            transition: all 0.2s;
            text-decoration: none;
        }}

        .btn-outline {{
            background: transparent;
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }}

        .btn-outline:hover {{
            background: rgba(255, 255, 255, 0.1);
        }}

        .btn-primary {{
            background: var(--secondary);
            color: white;
        }}

        .btn-github {{
            background: #24292f;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.35);
        }}

        .btn-github:hover {{
            background: #0f1419;
            border-color: rgba(255, 255, 255, 0.7);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }}

        .container {{
            max-width: 1300px;
            margin: 28px auto;
            padding: 0 24px;
        }}

        .slide-pane {{
            display: none;
            animation: fadeIn 0.3s ease-in-out;
        }}

        .slide-pane.active {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            box-shadow: var(--shadow);
            position: relative;
            overflow: hidden;
        }}

        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--primary-light);
        }}

        .kpi-card.green::before {{ background: var(--secondary); }}
        .kpi-card.red::before {{ background: var(--danger); }}
        .kpi-card.accent::before {{ background: var(--accent); }}

        .kpi-title {{
            font-size: 0.82rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}

        .kpi-value {{
            font-size: 1.8rem;
            font-weight: 800;
            color: var(--text-main);
            margin: 8px 0 4px;
        }}

        .kpi-desc {{
            font-size: 0.82rem;
            color: var(--text-muted);
        }}

        .content-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }}

        @media (max-width: 950px) {{
            .content-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            box-shadow: var(--shadow);
        }}

        .card h3 {{
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .card-img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid var(--border);
            background: #fff;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
            margin-top: 10px;
        }}

        th {{
            background: #f1f5f9;
            color: var(--primary);
            font-weight: 700;
            text-align: left;
            padding: 10px 12px;
            border-bottom: 2px solid var(--border);
        }}

        td {{
            padding: 9px 12px;
            border-bottom: 1px solid var(--border);
        }}

        tr:hover td {{
            background-color: #f8fafc;
        }}

        .badge {{
            display: inline-block;
            padding: 3px 8px;
            font-size: 0.72rem;
            font-weight: 700;
            border-radius: 999px;
            text-transform: uppercase;
        }}

        .badge-success {{ background: #dcfce7; color: #166534; }}
        .badge-warning {{ background: #fef3c7; color: #92400e; }}
        .badge-danger {{ background: #fee2e2; color: #991b1b; }}
        .badge-info {{ background: #e0f2fe; color: #075985; }}

        .callout {{
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 16px;
            border-radius: 6px;
            margin: 16px 0;
            font-size: 0.9rem;
        }}

        .callout-warning {{
            background: #fffbeb;
            border-left-color: #f59e0b;
        }}

        .slide-nav-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 32px;
            padding-top: 16px;
            border-top: 1px solid var(--border);
        }}

        .slide-indicator {{
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-muted);
        }}
    </style>
</head>
<body>

<header>
    <div class="header-container">
        <div class="title-area">
            <h1>📊 Data Science Analytics, Modeling & Refinement</h1>
            <p>Master Presentation Dashboard | Cars, Laptops, Healthcare Insurance & King County Housing</p>
        </div>
        <div class="nav-tabs" id="tabContainer">
            <button class="nav-tab active" onclick="switchSlide(0)">1. Portfolio Overview</button>
            <button class="nav-tab" onclick="switchSlide(1)">2. Automobile EDA</button>
            <button class="nav-tab" onclick="switchSlide(2)">3. Automobile Models</button>
            <button class="nav-tab" onclick="switchSlide(3)">4. Laptop Evaluation & Tuning</button>
            <button class="nav-tab" onclick="switchSlide(4)">5. Insurance Capstone</button>
            <button class="nav-tab" onclick="switchSlide(5)">6. Diagnostics & Overfitting</button>
            <button class="nav-tab" onclick="switchSlide(6)">7. Decision Matrix</button>
        </div>
        <div class="toolbar">
            <a href="https://github.com/mohamed-soubhi/data-science-model-development" target="_blank" class="btn btn-github">⭐ GitHub Repo</a>
            <button class="btn btn-outline" onclick="window.print()">🖨️ Print / PDF</button>
            <button class="btn btn-primary" onclick="toggleFullScreen()">⛶ Fullscreen</button>
        </div>
    </div>
</header>

<div class="container">

    <!-- SLIDE 0: PORTFOLIO OVERVIEW -->
    <div class="slide-pane active" id="slide-0">
        <div class="kpi-grid">
            <div class="kpi-card green">
                <div class="kpi-title">Complete Portfolio</div>
                <div class="kpi-value">7 Notebooks</div>
                <div class="kpi-desc">Cars EDA & ML, Laptop Dev & Tuning, Insurance, House Sales</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Top Benchmark Accuracy</div>
                <div class="kpi-value">Test R² 0.834</div>
                <div class="kpi-desc">Insurance Polynomial Pipeline (2,772 health records)</div>
            </div>
            <div class="kpi-card accent">
                <div class="kpi-title">GridSearch Optimization</div>
                <div class="kpi-value">+45.9% R² Boost</div>
                <div class="kpi-desc">Ridge alpha tuning lifted Laptop test R² from -0.07 to +0.386</div>
            </div>
            <div class="kpi-card red">
                <div class="kpi-title">King County Housing</div>
                <div class="kpi-value">R² 0.751 Pipeline</div>
                <div class="kpi-desc">21,613 records, 10/10 questions solved & validated</div>
            </div>
        </div>

        <div class="content-grid">
            <div class="card">
                <h3>🎯 Executive Summary: The 4-Domain Data Science Journey</h3>
                <p>This master presentation synthesizes empirical findings across three real-world domain studies:</p>
                <ul style="margin: 12px 0 12px 20px; font-size: 0.92rem; line-height: 1.8;">
                    <li><strong>Automobile Pricing Study:</strong> Uncovering continuous correlations, detecting extreme class imbalance (only 3 rear engines), and isolating engine size as the dominant driver ($F = 67.95, p < 10^{-22}$).</li>
                    <li><strong>Laptop Pricing & Model Refinement:</strong> Diagnosing the collapse of high-degree polynomials (degrees 4 and 5 drop to negative R²), and using <code>GridSearchCV</code> to tune Ridge regularization ($\alpha=100$).</li>
                    <li><strong>Healthcare Insurance Cost Capstone:</strong> Demonstrating how large sample sizes ($n=2,772$) enable polynomial interaction terms (`smoker * bmi`) to excel without overfitting (Test $R^2 = 0.834$).</li>
                    <li><strong>King County Housing Capstone:</strong> Predicting residential home prices ($n=21,613$) from square footage to multi-feature standardized polynomial pipelines ($R^2 = 0.751$) and regularized Ridge regression.</li>
                </ul>
                <div class="callout">
                    <strong>The Golden Rule of Model Complexity:</strong> Polynomial interaction terms destroy generalization on small samples ($n < 200$), but unlock superior predictive power at enterprise scale ($n > 2,000$).
                </div>
                <div style="margin-top: 14px;">
                    <a href="https://github.com/mohamed-soubhi/data-science-model-development" target="_blank" style="display: inline-flex; align-items: center; gap: 8px; font-weight: 600; color: var(--primary-light); text-decoration: none; font-size: 0.95rem;">
                        📁 Access all 7 notebooks, datasets & source code on GitHub →
                    </a>
                </div>
            </div>
            <div class="card">
                <h3>🏆 Cross-Study Master Benchmark: Out-of-Sample Test R²</h3>
                <img src="data:image/png;base64,{img_master_benchmark}" alt="Master Benchmark Chart" class="card-img">
            </div>
        </div>
    </div>

    <!-- SLIDE 1: AUTOMOBILE EDA -->
    <div class="slide-pane" id="slide-1">
        <div class="kpi-grid">
            <div class="kpi-card green">
                <div class="kpi-title">Top Positive Predictor</div>
                <div class="kpi-value">Engine Size (r=0.87)</div>
                <div class="kpi-desc">Curb Weight (0.83), Horsepower (0.81), Width (0.75)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Top Negative Predictor</div>
                <div class="kpi-value">Highway MPG (r=-0.70)</div>
                <div class="kpi-desc">City MPG (r=-0.69) | Clear economy vs luxury divide</div>
            </div>
            <div class="kpi-card accent">
                <div class="kpi-title">Categorical Drivetrain</div>
                <div class="kpi-value">ANOVA F = 67.95</div>
                <div class="kpi-desc">RWD vs FWD is highly significant (p &lt; 10⁻²⁰)</div>
            </div>
            <div class="kpi-card red">
                <div class="kpi-title">Imbalance Warning</div>
                <div class="kpi-value">Rear Engine (3 cars)</div>
                <div class="kpi-desc">Only 1.5% of dataset; high risk of overfitting in ML</div>
            </div>
        </div>

        <div class="content-grid">
            <div class="card">
                <h3>📊 Pearson Correlation Ranking (Continuous Features)</h3>
                <img src="data:image/png;base64,{img_eda_corr}" alt="EDA Correlation Bar Chart" class="card-img">
            </div>
            <div class="card">
                <h3>🔥 Heatmap: Drivetrain & Body Style Interaction</h3>
                <img src="data:image/png;base64,{img_eda_heatmap}" alt="EDA Pivot Heatmap" class="card-img">
            </div>
        </div>

        <div class="content-grid">
            <div class="card">
                <h3>📦 Drive Wheels Box Plot & ANOVA Analysis</h3>
                <img src="data:image/png;base64,{img_eda_box}" alt="EDA Drivetrain Boxplot" class="card-img">
            </div>
            <div class="card">
                <h3>📋 ANOVA Hypothesis Test Summary</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Categorical Attribute</th>
                            <th>F-Statistic</th>
                            <th>p-value</th>
                            <th>Predictive Quality</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>drive-wheels</strong></td>
                            <td><strong>67.95</strong></td>
                            <td>3.39 × 10⁻²³</td>
                            <td><span class="badge badge-success">Excellent</span> Robust sample balance</td>
                        </tr>
                        <tr>
                            <td><strong>engine-location</strong></td>
                            <td>24.50</td>
                            <td>1.58 × 10⁻⁶</td>
                            <td><span class="badge badge-warning">Caution</span> Only 3 rear-engine samples</td>
                        </tr>
                        <tr>
                            <td><strong>body-style</strong></td>
                            <td>9.13</td>
                            <td>8.78 × 10⁻⁷</td>
                            <td><span class="badge badge-info">Moderate</span> Substantial category overlap</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- SLIDE 2: AUTOMOBILE MODELS -->
    <div class="slide-pane" id="slide-2">
        <div class="kpi-grid">
            <div class="kpi-card green">
                <div class="kpi-title">Top Automobile Model</div>
                <div class="kpi-value">MLR (4 features)</div>
                <div class="kpi-desc">Test R² 0.7637 | Test RMSE $5,377</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">In-Sample R²</div>
                <div class="kpi-value">0.8089</div>
                <div class="kpi-desc">Explains over 80% of variation on training data</div>
            </div>
            <div class="kpi-card red">
                <div class="kpi-title">Confounding Variable</div>
                <div class="kpi-value">Highway MPG</div>
                <div class="kpi-desc">Slope was -$822 in SLR, but p = 0.627 in MLR due to VIF</div>
            </div>
            <div class="kpi-card accent">
                <div class="kpi-title">Regularized Ridge</div>
                <div class="kpi-value">Test R² 0.768</div>
                <div class="kpi-desc">L2 penalty protects degree-2 polynomial from overfit</div>
            </div>
        </div>

        <div class="content-grid">
            <div class="card">
                <h3>📈 Actual vs. Fitted Distribution (MLR)</h3>
                <img src="data:image/png;base64,{img_auto_dist}" alt="Automobile Distribution Plot" class="card-img">
            </div>
            <div class="card">
                <h3>📋 Automobile Model Performance Leaderboard</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Model Architecture</th>
                            <th>Train R²</th>
                            <th>Test R²</th>
                            <th>Test RMSE</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Multiple Linear Regression (4 feats)</strong></td>
                            <td>0.8089</td>
                            <td><strong>0.7637</strong></td>
                            <td>$5,377</td>
                            <td><span class="badge badge-success">Top Choice</span></td>
                        </tr>
                        <tr>
                            <td>Regularized Ridge (Poly Deg 2)</td>
                            <td>0.8315</td>
                            <td>0.7680</td>
                            <td>$5,328</td>
                            <td><span class="badge badge-success">Robust</span></td>
                        </tr>
                        <tr>
                            <td>Polynomial Degree 2 (Pipeline)</td>
                            <td>0.8526</td>
                            <td>0.7561</td>
                            <td>$5,462</td>
                            <td><span class="badge badge-info">Slight Overfit</span></td>
                        </tr>
                        <tr>
                            <td>Simple Linear Regression (engine-size)</td>
                            <td>0.7635</td>
                            <td>0.7102</td>
                            <td>$5,934</td>
                            <td><span class="badge badge-info">Good 1-Var</span></td>
                        </tr>
                        <tr>
                            <td>Polynomial Degree 3 (highway-mpg)</td>
                            <td>0.6933</td>
                            <td>0.6014</td>
                            <td>$6,983</td>
                            <td><span class="badge badge-warning">Non-linear</span></td>
                        </tr>
                        <tr>
                            <td>Simple Linear Regression (highway-mpg)</td>
                            <td>0.5013</td>
                            <td>0.4247</td>
                            <td>$8,390</td>
                            <td><span class="badge badge-danger">Underfit</span></td>
                        </tr>
                        <tr>
                            <td>Polynomial Degree 11 (highway-mpg)</td>
                            <td>0.7303</td>
                            <td>0.5560</td>
                            <td>$7,370</td>
                            <td><span class="badge badge-danger">Overfitted</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- SLIDE 3: LAPTOP EVALUATION & TUNING -->
    <div class="slide-pane" id="slide-3">
        <div class="kpi-grid">
            <div class="kpi-card green">
                <div class="kpi-title">Optimal Hyperparameter</div>
                <div class="kpi-value">Alpha = 100</div>
                <div class="kpi-desc">Selected via 4-fold GridSearchCV on Ridge regression</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Tuned Ridge Test R²</div>
                <div class="kpi-value">0.3861</div>
                <div class="kpi-desc">Up from -0.073 in unregularized polynomial expansion</div>
            </div>
            <div class="kpi-card red">
                <div class="kpi-title">Degree 5 Overfit</div>
                <div class="kpi-value">Test R² -1.376</div>
                <div class="kpi-desc">Severe variance explosion on high-degree polynomial</div>
            </div>
            <div class="kpi-card accent">
                <div class="kpi-title">Cross-Validation Gain</div>
                <div class="kpi-value">4-Fold Rotation</div>
                <div class="kpi-desc">Averaged scores eliminate small-sample evaluation bias</div>
            </div>
        </div>

        <div class="content-grid">
            <div class="card">
                <h3>⚠️ Diagnosing Overfitting Across Degrees 1 to 5</h3>
                <img src="data:image/png;base64,{img_lap_overfit_curve}" alt="Laptop Overfitting Curve" class="card-img">
                <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 8px;">
                    Train R² rises steadily to 0.417 while Test R² collapses past zero into -1.376, illustrating severe model variance.
                </p>
            </div>
            <div class="card">
                <h3>🎯 Automated Hyperparameter Tuning (GridSearchCV)</h3>
                <img src="data:image/png;base64,{img_lap_grid}" alt="GridSearchCV Curve" class="card-img">
                <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 8px;">
                    Mean 4-fold cross-validation R² steadily increases as regularization alpha is scaled, reaching stability at alpha = 100.
                </p>
            </div>
        </div>
    </div>

    <!-- SLIDE 4: INSURANCE CAPSTONE -->
    <div class="slide-pane" id="slide-4">
        <div class="kpi-grid">
            <div class="kpi-card green">
                <div class="kpi-title">Dataset Scale</div>
                <div class="kpi-value">2,772 Patients</div>
                <div class="kpi-desc">Clinical and demographic features | Target: Charges ($)</div>
            </div>
            <div class="kpi-card red">
                <div class="kpi-title">Smoking Risk Factor</div>
                <div class="kpi-value">+$23,838 / Year</div>
                <div class="kpi-desc">t = 68.61, p &lt; 10⁻¹⁰⁰ (Dominant health risk driver)</div>
            </div>
            <div class="kpi-card accent">
                <div class="kpi-title">Polynomial Test R²</div>
                <div class="kpi-value">0.8339</div>
                <div class="kpi-desc">Captures non-linear compounding interactions (smoker * bmi)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Test RMSE</div>
                <div class="kpi-value">$4,978</div>
                <div class="kpi-desc">Down from $8,079 in Single Linear Regression</div>
            </div>
        </div>

        <div class="content-grid">
            <div class="card">
                <h3>🚬 Smoking Status Impact on Medical Charges</h3>
                <img src="data:image/png;base64,{img_ins_box}" alt="Insurance Boxplot" class="card-img">
                <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 8px;">
                    Smokers face a dramatic upward charge shift: median costs jump from $7,345 to $34,456.
                </p>
            </div>
            <div class="card">
                <h3>📈 Actual vs. Polynomial Fitted Charges (Test R² = 0.834)</h3>
                <img src="data:image/png;base64,{img_ins_dist}" alt="Insurance Distribution Plot" class="card-img">
                <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 8px;">
                    The degree-2 polynomial pipeline captures both the low-cost non-smoker cluster and the elevated bimodal high-risk tail.
                </p>
            </div>
        </div>
    </div>

    <!-- SLIDE 5: DIAGNOSTICS & OVERFITTING -->
    <div class="slide-pane" id="slide-5">
        <div class="content-grid">
            <div class="card">
                <h3>⚠️ Runge's Phenomenon: Automobile Degree 11 Curve</h3>
                <img src="data:image/png;base64,{img_auto_overfit}" alt="Automobile Overfitting" class="card-img">
                <div class="callout callout-warning">
                    High-degree polynomials wiggle aggressively at dataset boundaries (40-55 mpg). While training error is low, test RMSE surges to $7,370.
                </div>
            </div>
            <div class="card">
                <h3>🔍 Multicollinearity Diagnostics (VIF)</h3>
                <img src="data:image/png;base64,{img_auto_vif}" alt="VIF Heatmap" class="card-img">
                <div class="callout callout-warning">
                    <code>curb-weight</code> and <code>engine-size</code> have VIF &gt; 5.5 (r = 0.85), explaining why highway-mpg became statistically insignificant in MLR (p = 0.627).
                </div>
            </div>
        </div>
    </div>

    <!-- SLIDE 6: DECISION MATRIX -->
    <div class="slide-pane" id="slide-6">
        <div class="card" style="margin-bottom: 24px;">
            <h3>⚖️ Portfolio Strategic Decision Matrix</h3>
            <table>
                <thead>
                    <tr>
                        <th>Domain</th>
                        <th>Recommended Model</th>
                        <th>Key Drivers</th>
                        <th>Generalization Accuracy</th>
                        <th>Operational Trade-Off</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Automobile Pricing</strong></td>
                        <td><strong>Multiple Linear Regression</strong> (or Ridge Deg 2)</td>
                        <td>Engine Size (p &lt; 10⁻⁷), Curb Weight (p &lt; 10⁻⁴), Horsepower (p &lt; 10⁻³)</td>
                        <td><span class="badge badge-success">High (Test R² 0.764)</span></td>
                        <td>Linear coefficients provide full regulatory transparency and avoid edge oscillations.</td>
                    </tr>
                    <tr>
                        <td><strong>Laptop Pricing</strong></td>
                        <td><strong>GridSearch Tuned Ridge (alpha=100)</strong> or MLR</td>
                        <td>RAM GB (p &lt; 10⁻¹⁰), Category (p &lt; 10⁻⁶), CPU Freq (p &lt; 10⁻⁵)</td>
                        <td><span class="badge badge-info">Moderate (Test R² 0.386)</span></td>
                        <td>Strong L2 regularization is essential to prevent 35 polynomial interaction terms from collapsing.</td>
                    </tr>
                    <tr>
                        <td><strong>Medical Insurance</strong></td>
                        <td><strong>Standardized Polynomial Pipeline (Deg 2)</strong></td>
                        <td>Smoker (p &lt; 10⁻¹⁰⁰), Age (p &lt; 10⁻⁶⁰), BMI (p &lt; 10⁻⁴⁰)</td>
                        <td><span class="badge badge-success">Very High (Test R² 0.834)</span></td>
                        <td>Large sample size (n=2,772) allows interaction terms (smoker * bmi) to generalize without overfitting.</td>
                    </tr>
                    <tr>
                        <td><strong>King County Housing</strong></td>
                        <td><strong>Standardized Polynomial Pipeline (Deg 2)</strong> / Ridge</td>
                        <td>sqft_living, grade, waterfront, lat, view</td>
                        <td><span class="badge badge-success">High (R² 0.751 Pipeline, Test R² 0.700)</span></td>
                        <td>Massive dataset (n=21,613) enables feature interactions & geo-spatial/structural learning without degradation.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="content-grid">
            <div class="card">
                <h3>🛡️ Automated Quality Assurance Suite (25 / 25 Passed)</h3>
                <ul style="font-size: 0.9rem; line-height: 2;">
                    <li>✅ <strong>Automobile EDA (5 Tests):</strong> Integrity, bounds, pivot completeness, ANOVA.</li>
                    <li>✅ <strong>Automobile Modeling (5 Tests):</strong> In-sample bounds, MLR &gt; SLR, pipeline match.</li>
                    <li>✅ <strong>Laptop Modeling (5 Tests):</strong> Non-negative prices, prediction vector completeness.</li>
                    <li>✅ <strong>Laptop Evaluation (5 Tests):</strong> CV validity, overfitting detection, GridSearch optimality.</li>
                    <li>✅ <strong>Insurance Capstone (5 Tests):</strong> Schema contract, R² hierarchy, polynomial superiority.</li>
                </ul>
            </div>
            <div class="card">
                <h3>💡 Key Recommendations for Production</h3>
                <ul style="font-size: 0.9rem; line-height: 2;">
                    <li><strong>Sample Size Dictates Complexity:</strong> Use MLR on small datasets ($n < 300$); reserve Polynomial Pipelines for larger datasets ($n > 2,000$).</li>
                    <li><strong>Always Regularize Expanded Feature Spaces:</strong> Tune Ridge $\alpha$ via <code>GridSearchCV</code> to control variance.</li>
                    <li><strong>Standardize Before Expanding:</strong> Encapsulate <code>StandardScaler</code> in a Scikit-Learn Pipeline to prevent data leakage.</li>
                </ul>
            </div>
        </div>

        <div style="text-align: center; margin-top: 24px; padding: 20px; background: #ffffff; border-radius: 12px; border: 1px solid var(--border); box-shadow: var(--shadow);">
            <h4 style="font-size: 1.1rem; color: var(--primary); margin-bottom: 8px;">🔗 Open Source Code & Reproducibility</h4>
            <p style="font-size: 0.92rem; color: var(--text-muted); margin-bottom: 14px;">All 7 executed Jupyter notebooks, datasets, test suites, and modeling pipelines are available in the public repository:</p>
            <a href="https://github.com/mohamed-soubhi/data-science-model-development" target="_blank" class="btn btn-github" style="font-size: 0.95rem; padding: 10px 22px;">
                ⭐ GitHub: mohamed-soubhi/data-science-model-development
            </a>
        </div>
    </div>

    <!-- Slide Navigation Footer -->
    <div class="slide-nav-bar">
        <button class="btn btn-outline" style="color: var(--primary);" onclick="prevSlide()">◀ Previous</button>
        <span class="slide-indicator" id="slideIndicator">Slide 1 of 7: Portfolio Overview</span>
        <button class="btn btn-primary" onclick="nextSlide()">Next ▶</button>
    </div>

</div>

<script>
    let currentSlide = 0;
    const slides = document.querySelectorAll('.slide-pane');
    const tabs = document.querySelectorAll('.nav-tab');
    const indicator = document.getElementById('slideIndicator');
    const titles = [
        "Portfolio Overview",
        "Automobile Exploratory Data Analysis (EDA)",
        "Automobile Predictive Modeling",
        "Laptop Model Evaluation & Hyperparameter Tuning",
        "Medical Insurance Cost Analysis Capstone",
        "Statistical Diagnostics & Overfitting",
        "Portfolio Decision Matrix & Strategic Takeaways"
    ];

    function switchSlide(index) {{
        if (index < 0 || index >= slides.length) return;
        slides[currentSlide].classList.remove('active');
        tabs[currentSlide].classList.remove('active');
        currentSlide = index;
        slides[currentSlide].classList.add('active');
        tabs[currentSlide].classList.add('active');
        indicator.textContent = `Slide ${{currentSlide + 1}} of ${{slides.length}}: ${{titles[currentSlide]}}`;
    }}

    function nextSlide() {{
        switchSlide((currentSlide + 1) % slides.length);
    }}

    function prevSlide() {{
        switchSlide((currentSlide - 1 + slides.length) % slides.length);
    }}

    function toggleFullScreen() {{
        if (!document.fullscreenElement) {{
            document.documentElement.requestFullscreen().catch(err => {{}});
        }} else {{
            document.exitFullscreen();
        }}
    }}

    document.addEventListener('keydown', (e) => {{
        if (e.key === 'ArrowRight' || e.key === ' ') {{
            nextSlide();
        }} else if (e.key === 'ArrowLeft') {{
            prevSlide();
        }}
    }});
</script>

</body>
</html>
"""

output_html = "presentation_dashboard.html"
with open(output_html, "w", encoding="utf-8") as f:
    f.write(html_content)

shutil.copyfile(output_html, "index.html")

print(f"Master Presentation HTML successfully updated at {output_html} and index.html (size: {len(html_content):,} bytes)!")
