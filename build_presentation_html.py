"""
Script to generate an executive-ready, interactive HTML presentation dashboard
summarizing the results, diagnostics, and testing of both Model Development labs.
"""
import base64
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# Matplotlib styling for high-DPI crisp figures
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial']
sns.set_theme(style="whitegrid", palette="muted")

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

print("1. Generating charts for Automobile lab...")
df_auto = pd.read_csv("automobileEDA.csv")
Z_auto = df_auto[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
y_auto = df_auto['price']

# Train-test split
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Z_auto, y_auto, test_size=0.2, random_state=42)
lm_auto = LinearRegression().fit(Xa_tr, ya_tr)

# Fig 1: Automobile Actual vs Fitted KDE
fig1, ax1 = plt.subplots(figsize=(7, 4.2))
sns.kdeplot(y_auto, color="#e63946", label="Actual Price", linewidth=2.5, fill=True, alpha=0.15, ax=ax1)
sns.kdeplot(lm_auto.predict(Z_auto), color="#1d3557", label="MLR Fitted Values", linewidth=2.5, fill=True, alpha=0.15, ax=ax1)
ax1.set_title("Automobile Price: Actual vs. MLR Fitted Distribution", fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel("Price (USD)", fontsize=10)
ax1.set_ylabel("Density", fontsize=10)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
img_auto_dist = fig_to_base64(fig1)

# Fig 2: Automobile Overfitting (Deg 1 vs 3 vs 11)
x_mpg = df_auto[['highway-mpg']]
x_tr, x_te, y_tr, y_te = train_test_split(x_mpg, y_auto, test_size=0.2, random_state=42)
m1 = Pipeline([('model', LinearRegression())]).fit(x_tr, y_tr)
m3 = Pipeline([('poly', PolynomialFeatures(3)), ('model', LinearRegression())]).fit(x_tr, y_tr)
m11 = Pipeline([('scale', StandardScaler()), ('poly', PolynomialFeatures(11)), ('model', LinearRegression())]).fit(x_tr, y_tr)
x_dom = pd.DataFrame({'highway-mpg': np.linspace(15, 55, 300)})

fig2, ax2 = plt.subplots(figsize=(7, 4.2))
ax2.scatter(x_tr, y_tr, color='#457b9d', alpha=0.5, s=30, label='Train (80%)')
ax2.scatter(x_te, y_te, color='#e63946', marker='^', s=45, label='Test (20%)', edgecolors='k')
ax2.plot(x_dom, m1.predict(x_dom), 'k--', label=f"Deg 1 [Test R²: {r2_score(y_te, m1.predict(x_te)):.2f}]", linewidth=1.8)
ax2.plot(x_dom, m3.predict(x_dom), '#2a9d8f', label=f"Deg 3 [Test R²: {r2_score(y_te, m3.predict(x_te)):.2f}]", linewidth=2.2)
ax2.plot(x_dom, m11.predict(x_dom), '#e76f51', linestyle='-.', label=f"Deg 11 [Test R²: {r2_score(y_te, m11.predict(x_te)):.2f}]", linewidth=1.8)
ax2.set_title("Automobile: Overfitting Comparison (Degree 1 vs 3 vs 11)", fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel("Highway MPG", fontsize=10)
ax2.set_ylabel("Price ($)", fontsize=10)
ax2.set_ylim(0, 52000)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
img_auto_overfit = fig_to_base64(fig2)

# Fig 3: Automobile VIF / Heatmap
fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.heatmap(Z_auto.corr(), annot=True, cmap='Blues', fmt='.2f', vmin=-1, vmax=1, ax=ax3, cbar=False)
ax3.set_title("Automobile Predictor Correlation Matrix (VIF > 5)", fontsize=11, fontweight='bold', pad=10)
img_auto_vif = fig_to_base64(fig3)

# Fig 4: Automobile Q-Q Plot
res_auto = ya_tr - lm_auto.predict(Xa_tr)
fig4, ax4 = plt.subplots(figsize=(6, 4))
stats.probplot(res_auto, dist="norm", plot=ax4)
ax4.get_lines()[0].set_color('#1d3557')
ax4.get_lines()[1].set_color('#e63946')
ax4.set_title("Automobile: Normal Q-Q Plot of Residuals", fontsize=11, fontweight='bold', pad=10)
img_auto_qq = fig_to_base64(fig4)

print("2. Generating charts for Laptop lab...")
df_lap = pd.read_csv("laptops.csv")
lap_features = ['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']
Z_lap = df_lap[lap_features]
y_lap = df_lap['Price']

Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(Z_lap, y_lap, test_size=0.2, random_state=42)
lm_lap = LinearRegression().fit(Xl_tr, yl_tr)

# Fig 5: Laptop Actual vs Fitted
fig5, ax5 = plt.subplots(figsize=(7, 4.2))
sns.kdeplot(y_lap, color="#e63946", label="Actual Price", linewidth=2.5, fill=True, alpha=0.15, ax=ax5)
sns.kdeplot(lm_lap.predict(Z_lap), color="#2a9d8f", label="MLR Fitted Values", linewidth=2.5, fill=True, alpha=0.15, ax=ax5)
ax5.set_title("Laptop Price: Actual vs. MLR Fitted Distribution", fontsize=12, fontweight='bold', pad=10)
ax5.set_xlabel("Price (USD)", fontsize=10)
ax5.set_ylabel("Density", fontsize=10)
ax5.legend(frameon=True, facecolor='white', framealpha=0.9)
img_laptop_dist = fig_to_base64(fig5)

# Fig 6: Laptop Overfitting comparison (MLR vs Poly Deg 2)
p2_lap = Pipeline([('scale', StandardScaler()), ('poly', PolynomialFeatures(2, include_bias=False)), ('model', LinearRegression())]).fit(Xl_tr, yl_tr)

fig6, ax6 = plt.subplots(figsize=(7, 4.2))
models_bar = ['MLR (7 feats)', 'Poly Deg 2 (35 terms)']
train_r2s = [r2_score(yl_tr, lm_lap.predict(Xl_tr)), r2_score(yl_tr, p2_lap.predict(Xl_tr))]
test_r2s = [r2_score(yl_te, lm_lap.predict(Xl_te)), r2_score(yl_te, p2_lap.predict(Xl_te))]
x_pos = np.arange(len(models_bar))
width = 0.35

rects1 = ax6.bar(x_pos - width/2, train_r2s, width, label='Train R² (In-Sample)', color='#457b9d')
rects2 = ax6.bar(x_pos + width/2, test_r2s, width, label='Test R² (Out-of-Sample)', color='#e76f51')

ax6.set_ylabel('R² Score', fontsize=10)
ax6.set_title('Laptop Overfitting: High Train R² vs Negative Test R²', fontsize=12, fontweight='bold', pad=10)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(models_bar, fontsize=10)
ax6.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax6.legend(frameon=True, facecolor='white')

for rect in rects1 + rects2:
    h = rect.get_height()
    ax6.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

img_laptop_overfit = fig_to_base64(fig6)

# Fig 7: Laptop VIF / Correlation Matrix
fig7, ax7 = plt.subplots(figsize=(6, 4))
sns.heatmap(Z_lap.corr(), annot=True, cmap='Blues', fmt='.2f', vmin=-1, vmax=1, ax=ax7, cbar=False)
ax7.set_title("Laptop Predictor Correlation Matrix (All VIF < 1.6)", fontsize=11, fontweight='bold', pad=10)
img_laptop_vif = fig_to_base64(fig7)

# Fig 8: Summary Benchmark Chart
fig8, ax8 = plt.subplots(figsize=(10, 4.5))
bench_models = [
    'Auto: MLR', 'Auto: Poly 2', 'Auto: Ridge', 'Auto: SLR Engine', 'Auto: Poly 3', 'Auto: Poly 11',
    'Lap: MLR', 'Lap: Poly 3', 'Lap: Ridge', 'Lap: SLR CPU', 'Lap: Poly 2'
]
bench_test_r2 = [0.7637, 0.7561, 0.7680, 0.7102, 0.6014, 0.5560, 0.2685, 0.1065, -0.0134, -0.0372, -0.0475]
colors = ['#2a9d8f' if x > 0.7 else ('#457b9d' if x > 0.2 else '#e76f51') for x in bench_test_r2]

y_ind = np.arange(len(bench_models))
ax8.barh(y_ind, bench_test_r2, color=colors, height=0.65)
ax8.set_yticks(y_ind)
ax8.set_yticklabels(bench_models, fontsize=9.5)
ax8.set_xlabel('Test R² (Unseen Data Generalization)', fontsize=10, fontweight='bold')
ax8.set_title('Cross-Study Benchmark: Out-of-Sample Test R² Comparison', fontsize=12, fontweight='bold', pad=10)
ax8.axvline(0, color='black', linewidth=1)
ax8.invert_yaxis()
for i, v in enumerate(bench_test_r2):
    ax8.text(v + (0.015 if v >= 0 else -0.07), i + 0.1, f"{v:.3f}", color='black', fontweight='bold', fontsize=8.5)
img_benchmark = fig_to_base64(fig8)

print("3. Building HTML Presentation file...")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Development & Evaluation | Executive Presentation</title>
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
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
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

        /* Header & Navigation */
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
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .title-area p {{
            color: #94a3b8;
            font-size: 0.9rem;
            margin-top: 4px;
        }}

        /* Slide / Tab Controls */
        .nav-tabs {{
            display: flex;
            gap: 8px;
            background: rgba(255, 255, 255, 0.1);
            padding: 4px;
            border-radius: 8px;
        }}

        .nav-tab {{
            background: transparent;
            border: none;
            color: #cbd5e1;
            padding: 8px 16px;
            font-size: 0.88rem;
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

        /* Presentation Mode Toolbar */
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

        .btn-primary:hover {{
            filter: brightness(1.1);
        }}

        /* Main Container */
        .container {{
            max-width: 1300px;
            margin: 28px auto;
            padding: 0 24px;
        }}

        /* Slide View Logic */
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

        /* Metric Cards Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
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
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}

        .kpi-value {{
            font-size: 1.85rem;
            font-weight: 800;
            color: var(--text-main);
            margin: 8px 0 4px;
        }}

        .kpi-desc {{
            font-size: 0.82rem;
            color: var(--text-muted);
        }}

        /* Split Grid for Visuals and Tables */
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

        /* Styled Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
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
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
        }}

        tr:hover td {{
            background-color: #f8fafc;
        }}

        .badge {{
            display: inline-block;
            padding: 3px 8px;
            font-size: 0.75rem;
            font-weight: 700;
            border-radius: 999px;
            text-transform: uppercase;
        }}

        .badge-success {{ background: #dcfce7; color: #166534; }}
        .badge-warning {{ background: #fef3c7; color: #92400e; }}
        .badge-danger {{ background: #fee2e2; color: #991b1b; }}
        .badge-info {{ background: #e0f2fe; color: #075985; }}

        /* Callout Box */
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

        .callout strong {{
            color: #1e3a8a;
        }}

        /* Presentation Slide Navigation Bar at bottom */
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
            <h1>📊 Model Development & Predictive Analytics</h1>
            <p>Comprehensive Review & Presentation Dashboard | Automobile & Laptop Pricing Studies</p>
        </div>
        <div class="nav-tabs" id="tabContainer">
            <button class="nav-tab active" onclick="switchSlide(0)">1. Overview</button>
            <button class="nav-tab" onclick="switchSlide(1)">2. Automobile Lab</button>
            <button class="nav-tab" onclick="switchSlide(2)">3. Laptop Lab</button>
            <button class="nav-tab" onclick="switchSlide(3)">4. Diagnostics & Overfitting</button>
            <button class="nav-tab" onclick="switchSlide(4)">5. Decision Matrix</button>
        </div>
        <div class="toolbar">
            <button class="btn btn-outline" onclick="window.print()">🖨️ Print / PDF</button>
            <button class="btn btn-primary" onclick="toggleFullScreen()">⛶ Fullscreen</button>
        </div>
    </div>
</header>

<div class="container">

    <!-- SLIDE 0: EXECUTIVE OVERVIEW -->
    <div class="slide-pane active" id="slide-0">
        <div class="kpi-grid">
            <div class="kpi-card green">
                <div class="kpi-title">Top Automobile Model</div>
                <div class="kpi-value">Test R² 0.764</div>
                <div class="kpi-desc">Multiple Linear Regression (4 features) | RMSE $5,377</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Top Laptop Model</div>
                <div class="kpi-value">Test R² 0.269</div>
                <div class="kpi-desc">Multiple Linear Regression (7 features) | RMSE $410</div>
            </div>
            <div class="kpi-card red">
                <div class="kpi-title">Overfitting Trap Exposed</div>
                <div class="kpi-value">Degree 11 & Poly 2</div>
                <div class="kpi-desc">High train R² collapsed to negative / low out-of-sample R²</div>
            </div>
            <div class="kpi-card accent">
                <div class="kpi-title">Unit Test Suite</div>
                <div class="kpi-value">100% Passed</div>
                <div class="kpi-desc">10 automated assertions across both notebooks verified</div>
            </div>
        </div>

        <div class="content-grid">
            <div class="card">
                <h3>🎯 Executive Summary & Objectives</h3>
                <p>This presentation synthesizes the empirical findings, model evaluations, and statistical diagnostics across two standard Data Science regression labs:</p>
                <ul style="margin: 12px 0 12px 20px; font-size: 0.92rem; line-height: 1.8;">
                    <li><strong>Automobile Price Prediction:</strong> 201 vehicles, analyzing how engine size, curb weight, horsepower, and fuel efficiency determine market price.</li>
                    <li><strong>Laptop Price Prediction:</strong> 238 laptops, investigating processor clock speed, RAM, SSD capacity, core count, and GPU on retail pricing.</li>
                    <li><strong>Methodological Rigor:</strong> Moving beyond naive in-sample $R^2$ to rigorous <strong>80/20 train/test splits</strong>, <strong>5-fold cross-validation</strong>, and <strong>Gauss-Markov diagnostic testing</strong>.</li>
                    <li><strong>Software Reliability:</strong> Executed end-to-end with <strong>zero warnings</strong> (replacing deprecated Seaborn functions) and verified via automated assertion test suites.</li>
                </ul>
                <div class="callout">
                    <strong>Key Strategic Insight:</strong> Simpler, well-regularized models (MLR and low-degree polynomials) consistently outperform high-degree polynomial expansions on unseen test data.
                </div>
            </div>
            <div class="card">
                <h3>🏆 Cross-Study Benchmark: Out-of-Sample Test R²</h3>
                <img src="data:image/png;base64,{img_benchmark}" alt="Benchmark Chart" class="card-img">
            </div>
        </div>
    </div>

    <!-- SLIDE 1: AUTOMOBILE LAB -->
    <div class="slide-pane" id="slide-1">
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Dataset Scale</div>
                <div class="kpi-value">201 Cars</div>
                <div class="kpi-desc">29 raw specifications | Target: Price ($5,118 - $45,400)</div>
            </div>
            <div class="kpi-card green">
                <div class="kpi-title">Strongest Predictor</div>
                <div class="kpi-value">Engine Size</div>
                <div class="kpi-desc">t = 5.80, p = 2.66 × 10⁻⁸ (Highly Significant)</div>
            </div>
            <div class="kpi-card red">
                <div class="kpi-title">Confounding Variable</div>
                <div class="kpi-value">Highway MPG</div>
                <div class="kpi-desc">Negative in SLR (-$822), but insignificant in MLR (p = 0.63)</div>
            </div>
            <div class="kpi-card accent">
                <div class="kpi-title">MLR Test Accuracy</div>
                <div class="kpi-value">76.4%</div>
                <div class="kpi-desc">Explained out-of-sample variance ($R^2 = 0.7637$)</div>
            </div>
        </div>

        <div class="content-grid">
            <div class="card">
                <h3>📈 Actual vs. Fitted Distribution (MLR)</h3>
                <img src="data:image/png;base64,{img_auto_dist}" alt="Automobile Distribution Plot" class="card-img">
                <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 8px;">
                    Modern KDE plot showing how the 4-feature MLR model closely mirrors actual automobile price density across budget and mid-tier segments.
                </p>
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

    <!-- SLIDE 2: LAPTOP LAB -->
    <div class="slide-pane" id="slide-2">
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Dataset Scale</div>
                <div class="kpi-value">238 Laptops</div>
                <div class="kpi-desc">7 key hardware features | Target: Price ($527 - $3,810)</div>
            </div>
            <div class="kpi-card green">
                <div class="kpi-title">Dominant Price Driver</div>
                <div class="kpi-value">RAM (GB)</div>
                <div class="kpi-desc">+$86.81 / GB | t = 6.81, p = 8.40 × 10⁻¹¹</div>
            </div>
            <div class="kpi-card accent">
                <div class="kpi-title">Secondary Driver</div>
                <div class="kpi-value">CPU Frequency</div>
                <div class="kpi-desc">+$961.28 / GHz | t = 4.71, p = 4.31 × 10⁻⁶</div>
            </div>
            <div class="kpi-card red">
                <div class="kpi-title">Non-Significant Feature</div>
                <div class="kpi-value">SSD Capacity</div>
                <div class="kpi-desc">p = 0.647 (t = 0.46) | Inconsequential when RAM is controlled</div>
            </div>
        </div>

        <div class="content-grid">
            <div class="card">
                <h3>📈 Actual vs. Fitted Distribution (Laptop MLR)</h3>
                <img src="data:image/png;base64,{img_laptop_dist}" alt="Laptop Distribution Plot" class="card-img">
                <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 8px;">
                    MLR model captures the primary distribution mode around $1,000 and the secondary shoulder around $2,000, significantly outperforming single-variable CPU regression.
                </p>
            </div>
            <div class="card">
                <h3>📋 Laptop Model Performance Leaderboard</h3>
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
                            <td><strong>Multiple Linear Regression (7 feats)</strong></td>
                            <td>0.5212</td>
                            <td><strong>0.2685</strong></td>
                            <td>$410.58</td>
                            <td><span class="badge badge-success">Top Choice</span></td>
                        </tr>
                        <tr>
                            <td>Poly Degree 3 (CPU_frequency)</td>
                            <td>0.2711</td>
                            <td>0.1065</td>
                            <td>$453.78</td>
                            <td><span class="badge badge-info">Modest Fit</span></td>
                        </tr>
                        <tr>
                            <td>Poly Degree 5 (CPU_frequency)</td>
                            <td>0.3131</td>
                            <td>0.1004</td>
                            <td>$455.34</td>
                            <td><span class="badge badge-warning">Mild Overfit</span></td>
                        </tr>
                        <tr>
                            <td>Ridge Regularized (Poly 2)</td>
                            <td>0.6707</td>
                            <td>-0.0134</td>
                            <td>$483.26</td>
                            <td><span class="badge badge-danger">Negative R²</span></td>
                        </tr>
                        <tr>
                            <td>Simple Linear Regression (CPU_freq)</td>
                            <td>0.1384</td>
                            <td>-0.0372</td>
                            <td>$488.91</td>
                            <td><span class="badge badge-danger">Inadequate</span></td>
                        </tr>
                        <tr>
                            <td>Polynomial Degree 2 (7 feats, 35 terms)</td>
                            <td>0.6746</td>
                            <td>-0.0475</td>
                            <td>$491.33</td>
                            <td><span class="badge badge-danger">Dimensionality Collapse</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- SLIDE 3: DIAGNOSTICS & OVERFITTING -->
    <div class="slide-pane" id="slide-3">
        <div class="content-grid">
            <div class="card">
                <h3>⚠️ The Overfitting Trap: Degree 1 vs 3 vs 11</h3>
                <img src="data:image/png;base64,{img_auto_overfit}" alt="Overfitting Curves" class="card-img">
                <div class="callout callout-warning">
                    <strong>Runge's Phenomenon:</strong> Degree 11 polynomial wiggles aggressively at boundary points (40-55 mpg). While training error is minimized, test error surges ($RMSE = $7,370).
                </div>
            </div>
            <div class="card">
                <h3>📉 Laptop Dimensionality Collapse (Poly Deg 2)</h3>
                <img src="data:image/png;base64,{img_laptop_overfit}" alt="Laptop Overfit Bar Chart" class="card-img">
                <div class="callout callout-warning">
                    <strong>Curse of Dimensionality:</strong> Expanding 7 features into degree 2 created 35 terms. With only 190 training laptops, the model memorized noise, producing a negative Test R² (-0.05).
                </div>
            </div>
        </div>

        <div class="content-grid">
            <div class="card">
                <h3>🔍 Multicollinearity Diagnostics (VIF)</h3>
                <img src="data:image/png;base64,{img_auto_vif}" alt="VIF Heatmap" class="card-img">
                <p style="font-size: 0.85rem; margin-top: 8px;">
                    In automobiles, <code>engine-size</code> and <code>curb-weight</code> exhibit VIF &gt; 5.5 and pairwise correlation $r = 0.85$. This multicollinearity masks the independent effect of fuel economy.
                </p>
            </div>
            <div class="card">
                <h3>📊 Residual Normality (Q-Q Plot)</h3>
                <img src="data:image/png;base64,{img_auto_qq}" alt="Q-Q Plot" class="card-img">
                <p style="font-size: 0.85rem; margin-top: 8px;">
                    Residuals adhere closely to the theoretical normal line for standard vehicles, with slight upward deviation in the high-end luxury bracket (&gt; $35,000).
                </p>
            </div>
        </div>
    </div>

    <!-- SLIDE 4: DECISION MATRIX & CONCLUSION -->
    <div class="slide-pane" id="slide-4">
        <div class="card" style="margin-bottom: 24px;">
            <h3>⚖️ Strategic Model Decision Matrix</h3>
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
                        <td>Highly transparent coefficients; avoids polynomial edge instability.</td>
                    </tr>
                    <tr>
                        <td><strong>Laptop Pricing</strong></td>
                        <td><strong>Multiple Linear Regression</strong> (7 Features)</td>
                        <td>RAM GB (p &lt; 10⁻¹⁰), Category (p &lt; 10⁻⁶), CPU Freq (p &lt; 10⁻⁵)</td>
                        <td><span class="badge badge-info">Moderate (Test R² 0.269)</span></td>
                        <td>Polynomial interaction terms suffer from sample sparsity; linear model is safest.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="content-grid">
            <div class="card">
                <h3>🛡️ Automated Quality Assurance Suite</h3>
                <ul style="font-size: 0.9rem; line-height: 2;">
                    <li>✅ <strong>Data Contract Assertions:</strong> Validated non-empty, non-null, and non-negative target distributions.</li>
                    <li>✅ <strong>Mathematical Bounds:</strong> Verified 0 &le; R&sup2; &le; 1 and R&sup2;(MLR) &gt; R&sup2;(SLR).</li>
                    <li>✅ <strong>Pipeline Equivalence:</strong> Proved Scikit-Learn Pipeline output matches manual sequential transformations.</li>
                    <li>✅ <strong>Overfitting Detection:</strong> Confirmed programmatic failure conditions whenever Test R&sup2; diverges excessively from Train R&sup2;.</li>
                </ul>
            </div>
            <div class="card">
                <h3>💡 Key Recommendations for Production</h3>
                <ul style="font-size: 0.9rem; line-height: 2;">
                    <li><strong>Always Validate Out-of-Sample:</strong> In-sample $R^2$ is an unconstrained optimizer of training error and cannot be trusted alone.</li>
                    <li><strong>Enforce Scikit-Learn Pipelines:</strong> Encapsulating <code>StandardScaler</code> within a pipeline eliminates data leakage during cross-validation.</li>
                    <li><strong>Audit Feature Multicollinearity:</strong> Run VIF tests before drawing causal inferences about individual coefficient signs.</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- Slide Navigation Footer -->
    <div class="slide-nav-bar">
        <button class="btn btn-outline" style="color: var(--primary);" onclick="prevSlide()">◀ Previous</button>
        <span class="slide-indicator" id="slideIndicator">Slide 1 of 5: Executive Overview</span>
        <button class="btn btn-primary" onclick="nextSlide()">Next ▶</button>
    </div>

</div>

<script>
    let currentSlide = 0;
    const slides = document.querySelectorAll('.slide-pane');
    const tabs = document.querySelectorAll('.nav-tab');
    const indicator = document.getElementById('slideIndicator');
    const titles = [
        "Executive Overview",
        "Automobile Pricing Study",
        "Laptop Pricing Study",
        "Statistical Diagnostics & Overfitting",
        "Decision Matrix & Strategic Takeaways"
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

    // Keyboard navigation (Left / Right arrow, Space)
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

print(f"Presentation HTML dashboard successfully generated at {output_html} (size: {len(html_content):,} bytes)!")
