import os
import time
import json
import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

os.makedirs('output_plots', exist_ok=True)

# -------------------------------------------------------------
# TASK 1: DATA EXPLORATION AND STATISTICAL ANALYSIS
# -------------------------------------------------------------
print("="*70)
print("TASK 1: DATA EXPLORATION AND STATISTICAL ANALYSIS")
print("="*70)

wb = openpyxl.load_workbook('FBG_biaxial_tilt_sensor.xlsx', data_only=True)
col_names = ['tilt_angle', 'fbg1', 'fbg2', 'd_lambda1', 'd_lambda2', 'd_lambda_diff', 'd_lambda_avg']

def load_sheet(name):
    sheet = wb[name]
    rows = []
    for r in range(1, 38):
        rows.append([sheet.cell(r, c).value for c in range(1, 8)])
    df = pd.DataFrame(rows, columns=col_names).astype(float)
    return df

df_axial = load_sheet('Experiment axial ')
df_trans = load_sheet('Experiment transverse ')

print(f"Axial dataset shape: {df_axial.shape}, Missing values: {df_axial.isnull().sum().sum()}")
print(f"Transverse dataset shape: {df_trans.shape}, Missing values: {df_trans.isnull().sum().sum()}")

# Descriptive statistics
stats_axial = df_axial.describe().T[['mean', '50%', 'std', 'min', 'max']].rename(columns={'50%': 'median'})
stats_axial['variance'] = df_axial.var()
stats_trans = df_trans.describe().T[['mean', '50%', 'std', 'min', 'max']].rename(columns={'50%': 'median'})
stats_trans['variance'] = df_trans.var()

print("\n--- Descriptive Statistics: Axial ---")
print(stats_axial)
print("\n--- Descriptive Statistics: Transverse ---")
print(stats_trans)

# Feature distribution plots
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sns.histplot(df_axial['tilt_angle'], kde=True, ax=axes[0, 0], color='#2b5c8f', bins=19)
axes[0, 0].set_title('Axial Tilt Angle Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Tilt Angle (°)')

sns.histplot(df_trans['tilt_angle'], kde=True, ax=axes[0, 1], color='#d95f02', bins=19)
axes[0, 1].set_title('Transverse Tilt Angle Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Tilt Angle (°)')

sns.kdeplot(data=df_axial[['d_lambda1', 'd_lambda2', 'd_lambda_diff']], ax=axes[1, 0], fill=True, palette='tab10')
axes[1, 0].set_title('Axial Wavelength Change Densities', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Wavelength Change (pm)')

sns.kdeplot(data=df_trans[['d_lambda1', 'd_lambda2', 'd_lambda_avg']], ax=axes[1, 1], fill=True, palette='Dark2')
axes[1, 1].set_title('Transverse Wavelength Change Densities', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Wavelength Change (pm)')

plt.tight_layout()
plt.savefig('output_plots/task1_distributions.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# TASK 2: DATA VISUALIZATION AND PREPROCESSING
# -------------------------------------------------------------
print("\n" + "="*70)
print("TASK 2: DATA VISUALIZATION AND PREPROCESSING")
print("="*70)

# 1 & 2: Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(df_axial['tilt_angle'], df_axial['fbg1'], 'o-', color='#1f77b4', label='FBG1 (Axial)')
axes[0].plot(df_axial['tilt_angle'], df_axial['fbg2'], 's--', color='#ff7f0e', label='FBG2 (Axial)')
axes[0].set_title('Axial Tilt: FBG Responses', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Tilt Angle (°)')
axes[0].set_ylabel('Peak Wavelength (nm)')
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.6)

axes[1].plot(df_trans['tilt_angle'], df_trans['fbg1'], 'o-', color='#2ca02c', label='FBG1 (Transverse)')
axes[1].plot(df_trans['tilt_angle'], df_trans['fbg2'], 's--', color='#d62728', label='FBG2 (Transverse)')
axes[1].set_title('Transverse Tilt: FBG Responses', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Tilt Angle (°)')
axes[1].set_ylabel('Spectral Measurement')
axes[1].legend()
axes[1].grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('output_plots/task2_scatter_responses.png', dpi=300)
plt.close()

# 3: Correlation heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
sns.heatmap(df_axial.corr(), annot=True, fmt='.3f', cmap='coolwarm', ax=axes[0], cbar=True)
axes[0].set_title('Correlation Heatmap - Axial Dataset', fontsize=12, fontweight='bold')

sns.heatmap(df_trans.corr(), annot=True, fmt='.3f', cmap='coolwarm', ax=axes[1], cbar=True)
axes[1].set_title('Correlation Heatmap - Transverse Dataset', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('output_plots/task2_correlation_heatmaps.png', dpi=300)
plt.close()

# 4: Box plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df_axial[['d_lambda1', 'd_lambda2', 'd_lambda_diff', 'd_lambda_avg']], ax=axes[0], palette='Blues')
axes[0].set_title('Outlier Analysis (Boxplot) - Axial Dataset', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Value (pm)')

sns.boxplot(data=df_trans[['d_lambda1', 'd_lambda2', 'd_lambda_diff', 'd_lambda_avg']], ax=axes[1], palette='Oranges')
axes[1].set_title('Outlier Analysis (Boxplot) - Transverse Dataset', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Value')
plt.tight_layout()
plt.savefig('output_plots/task2_boxplots.png', dpi=300)
plt.close()

# 5 & 6: Standardization vs Normalization
feature_cols = ['d_lambda1', 'd_lambda2', 'd_lambda_diff', 'd_lambda_avg']
scaler_std = StandardScaler()
scaler_minmax = MinMaxScaler()

axial_std = pd.DataFrame(scaler_std.fit_transform(df_axial[feature_cols]), columns=feature_cols)
axial_minmax = pd.DataFrame(scaler_minmax.fit_transform(df_axial[feature_cols]), columns=feature_cols)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
sns.kdeplot(data=df_axial[feature_cols], ax=axes[0], palette='tab10')
axes[0].set_title('Raw Features (Axial)', fontsize=11, fontweight='bold')

sns.kdeplot(data=axial_std, ax=axes[1], palette='tab10')
axes[1].set_title('StandardScaler Normalization', fontsize=11, fontweight='bold')

sns.kdeplot(data=axial_minmax, ax=axes[2], palette='tab10')
axes[2].set_title('MinMaxScaler Normalization', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('output_plots/task2_normalization_comparison.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# TASK 3: FEATURE ENGINEERING AND FEATURE SELECTION
# -------------------------------------------------------------
print("\n" + "="*70)
print("TASK 3: FEATURE ENGINEERING AND SELECTION")
print("="*70)

# Feature configurations
# F1: Individual FBG wavelength changes [d_lambda1, d_lambda2]
# F2: Differential wavelength change [d_lambda_diff]
# F3: Average wavelength change [d_lambda_avg]
# F4: Differential and average wavelength changes [d_lambda_diff, d_lambda_avg]

feat_configs = {
    'F1': ['d_lambda1', 'd_lambda2'],
    'F2': ['d_lambda_diff'],
    'F3': ['d_lambda_avg'],
    'F4': ['d_lambda_diff', 'd_lambda_avg']
}

corr_axial = {
    'd_lambda1': df_axial['tilt_angle'].corr(df_axial['d_lambda1']),
    'd_lambda2': df_axial['tilt_angle'].corr(df_axial['d_lambda2']),
    'd_lambda_diff': df_axial['tilt_angle'].corr(df_axial['d_lambda_diff']),
    'd_lambda_avg': df_axial['tilt_angle'].corr(df_axial['d_lambda_avg'])
}

corr_trans_signed = {
    'd_lambda1': df_trans['tilt_angle'].corr(df_trans['d_lambda1']),
    'd_lambda2': df_trans['tilt_angle'].corr(df_trans['d_lambda2']),
    'd_lambda_diff': df_trans['tilt_angle'].corr(df_trans['d_lambda_diff']),
    'd_lambda_avg': df_trans['tilt_angle'].corr(df_trans['d_lambda_avg'])
}

corr_trans_abs = {
    'd_lambda1': df_trans['tilt_angle'].abs().corr(df_trans['d_lambda1']),
    'd_lambda2': df_trans['tilt_angle'].abs().corr(df_trans['d_lambda2']),
    'd_lambda_diff': df_trans['tilt_angle'].abs().corr(df_trans['d_lambda_diff']),
    'd_lambda_avg': df_trans['tilt_angle'].abs().corr(df_trans['d_lambda_avg'])
}

df_corr_analysis = pd.DataFrame({
    'Axial (r with theta)': corr_axial,
    'Transverse Signed (r with theta)': corr_trans_signed,
    'Transverse Magnitude (r with |theta|)': corr_trans_abs
})
print("\n--- Pearson Correlation Coefficients with Tilt Angle ---")
print(df_corr_analysis)

# -------------------------------------------------------------
# TASK 4, 5, 6: MODEL IMPLEMENTATION, EVALUATION, AND COMPARISON
# -------------------------------------------------------------
print("\n" + "="*70)
print("TASKS 4, 5, 6: CROSS-VALIDATION & MODEL EVALUATION")
print("="*70)

# Evaluation Methodology:
# 5-fold GroupKFold based on tilt_angle (keeps identical angles in the same fold)
# Fitting scaler on training folds only
# For models with hyperparameters: inner CV or sweep on training fold to select optimal hyperparameter

def evaluate_models(df, target_col='tilt_angle', feat_cols=['d_lambda1', 'd_lambda2']):
    X = df[feat_cols].values
    y = df[target_col].values
    groups = df['tilt_angle'].values
    
    gkf = GroupKFold(n_splits=5)
    
    model_defs = {
        'Linear Regression': {
            'type': 'pipeline',
            'pipe': lambda: Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])
        },
        'Polynomial Regression': {
            'type': 'poly', # Degree 2 or 3 tuned in-fold
            'degrees': [2, 3]
        },
        'Ridge Regression': {
            'type': 'ridge', # alphas: [0.01, 0.1, 1.0, 10.0]
            'alphas': [0.01, 0.1, 1.0, 10.0]
        },
        'Lasso Regression': {
            'type': 'lasso', # alphas: [0.01, 0.1, 1.0]
            'alphas': [0.01, 0.1, 1.0]
        },
        'Elastic Net': {
            'type': 'elasticnet',
            'params': [(0.01, 0.5), (0.1, 0.5), (1.0, 0.5)]
        },
        'SVR': {
            'type': 'svr', # linear vs rbf
            'kernels': ['linear', 'rbf']
        },
        'Decision Tree': {
            'type': 'dtr', # max_depth: 2, 3, 5
            'depths': [2, 3, 5]
        },
        'Random Forest': {
            'type': 'rfr', # n_estimators: 50, 100
            'estimators': [50, 100]
        },
        'KNN': {
            'type': 'knn', # n_neighbors: 3, 5, 7
            'neighbors': [3, 5, 7]
        },
        'Gradient Boosting': {
            'type': 'gbr', # lr: 0.01, 0.1
            'lrs': [0.01, 0.1]
        },
        'GPR': {
            'type': 'gpr', # RBF vs Matern
            'kernels': ['rbf', 'matern']
        }
    }
    
    results = {}
    oof_predictions = {name: np.zeros(len(y)) for name in model_defs}
    training_times = {name: 0.0 for name in model_defs}
    selected_hyperparams = {name: [] for name in model_defs}
    
    # Run 5-fold CV
    for model_name, config in model_defs.items():
        t0 = time.time()
        preds = np.zeros(len(y))
        
        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            train_groups = groups[train_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Select best hyperparameter within training fold using inner GroupKFold
            inner_gkf = GroupKFold(n_splits=3)
            
            if config['type'] == 'pipeline':
                reg = LinearRegression()
                reg.fit(X_train_scaled, y_train)
                preds[test_idx] = reg.predict(X_test_scaled)
                best_param = 'None'
                
            elif config['type'] == 'poly':
                best_deg = 2
                best_score = float('inf')
                for d in config['degrees']:
                    fold_scores = []
                    for in_tr, in_val in inner_gkf.split(X_train_scaled, y_train, groups=train_groups):
                        poly = PolynomialFeatures(degree=d, include_bias=False)
                        X_in_tr = poly.fit_transform(X_train_scaled[in_tr])
                        X_in_val = poly.transform(X_train_scaled[in_val])
                        m = LinearRegression().fit(X_in_tr, y_train[in_tr])
                        val_pred = m.predict(X_in_val)
                        fold_scores.append(mean_squared_error(y_train[in_val], val_pred))
                    if np.mean(fold_scores) < best_score:
                        best_score = np.mean(fold_scores)
                        best_deg = d
                poly = PolynomialFeatures(degree=best_deg, include_bias=False)
                X_tr_poly = poly.fit_transform(X_train_scaled)
                X_te_poly = poly.transform(X_test_scaled)
                reg = LinearRegression().fit(X_tr_poly, y_train)
                preds[test_idx] = reg.predict(X_te_poly)
                best_param = f"degree={best_deg}"
                
            elif config['type'] == 'ridge':
                best_a = 1.0
                best_score = float('inf')
                for a in config['alphas']:
                    scores = [mean_squared_error(y_train[iv], Ridge(alpha=a).fit(X_train_scaled[it], y_train[it]).predict(X_train_scaled[iv]))
                              for it, iv in inner_gkf.split(X_train_scaled, y_train, groups=train_groups)]
                    if np.mean(scores) < best_score:
                        best_score = np.mean(scores)
                        best_a = a
                reg = Ridge(alpha=best_a).fit(X_train_scaled, y_train)
                preds[test_idx] = reg.predict(X_test_scaled)
                best_param = f"alpha={best_a}"
                
            elif config['type'] == 'lasso':
                best_a = 0.01
                best_score = float('inf')
                for a in config['alphas']:
                    scores = [mean_squared_error(y_train[iv], Lasso(alpha=a, max_iter=5000).fit(X_train_scaled[it], y_train[it]).predict(X_train_scaled[iv]))
                              for it, iv in inner_gkf.split(X_train_scaled, y_train, groups=train_groups)]
                    if np.mean(scores) < best_score:
                        best_score = np.mean(scores)
                        best_a = a
                reg = Lasso(alpha=best_a, max_iter=5000).fit(X_train_scaled, y_train)
                preds[test_idx] = reg.predict(X_test_scaled)
                best_param = f"alpha={best_a}"
                
            elif config['type'] == 'elasticnet':
                best_p = (0.01, 0.5)
                best_score = float('inf')
                for a, l1 in config['params']:
                    scores = [mean_squared_error(y_train[iv], ElasticNet(alpha=a, l1_ratio=l1, max_iter=5000).fit(X_train_scaled[it], y_train[it]).predict(X_train_scaled[iv]))
                              for it, iv in inner_gkf.split(X_train_scaled, y_train, groups=train_groups)]
                    if np.mean(scores) < best_score:
                        best_score = np.mean(scores)
                        best_p = (a, l1)
                reg = ElasticNet(alpha=best_p[0], l1_ratio=best_p[1], max_iter=5000).fit(X_train_scaled, y_train)
                preds[test_idx] = reg.predict(X_test_scaled)
                best_param = f"alpha={best_p[0]}, l1={best_p[1]}"
                
            elif config['type'] == 'svr':
                best_k = 'rbf'
                best_score = float('inf')
                for k in config['kernels']:
                    scores = [mean_squared_error(y_train[iv], SVR(kernel=k).fit(X_train_scaled[it], y_train[it]).predict(X_train_scaled[iv]))
                              for it, iv in inner_gkf.split(X_train_scaled, y_train, groups=train_groups)]
                    if np.mean(scores) < best_score:
                        best_score = np.mean(scores)
                        best_k = k
                reg = SVR(kernel=best_k).fit(X_train_scaled, y_train)
                preds[test_idx] = reg.predict(X_test_scaled)
                best_param = f"kernel={best_k}"
                
            elif config['type'] == 'dtr':
                best_d = 3
                best_score = float('inf')
                for d in config['depths']:
                    scores = [mean_squared_error(y_train[iv], DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train_scaled[it], y_train[it]).predict(X_train_scaled[iv]))
                              for it, iv in inner_gkf.split(X_train_scaled, y_train, groups=train_groups)]
                    if np.mean(scores) < best_score:
                        best_score = np.mean(scores)
                        best_d = d
                reg = DecisionTreeRegressor(max_depth=best_d, random_state=42).fit(X_train_scaled, y_train)
                preds[test_idx] = reg.predict(X_test_scaled)
                best_param = f"max_depth={best_d}"
                
            elif config['type'] == 'rfr':
                best_n = 100
                best_score = float('inf')
                for n in config['estimators']:
                    scores = [mean_squared_error(y_train[iv], RandomForestRegressor(n_estimators=n, random_state=42).fit(X_train_scaled[it], y_train[it]).predict(X_train_scaled[iv]))
                              for it, iv in inner_gkf.split(X_train_scaled, y_train, groups=train_groups)]
                    if np.mean(scores) < best_score:
                        best_score = np.mean(scores)
                        best_n = n
                reg = RandomForestRegressor(n_estimators=best_n, random_state=42).fit(X_train_scaled, y_train)
                preds[test_idx] = reg.predict(X_test_scaled)
                best_param = f"n_estimators={best_n}"
                
            elif config['type'] == 'knn':
                best_k = 3
                best_score = float('inf')
                for k in config['neighbors']:
                    scores = [mean_squared_error(y_train[iv], KNeighborsRegressor(n_neighbors=k).fit(X_train_scaled[it], y_train[it]).predict(X_train_scaled[iv]))
                              for it, iv in inner_gkf.split(X_train_scaled, y_train, groups=train_groups)]
                    if np.mean(scores) < best_score:
                        best_score = np.mean(scores)
                        best_k = k
                reg = KNeighborsRegressor(n_neighbors=best_k).fit(X_train_scaled, y_train)
                preds[test_idx] = reg.predict(X_test_scaled)
                best_param = f"n_neighbors={best_k}"
                
            elif config['type'] == 'gbr':
                best_lr = 0.1
                best_score = float('inf')
                for lr in config['lrs']:
                    scores = [mean_squared_error(y_train[iv], GradientBoostingRegressor(learning_rate=lr, random_state=42).fit(X_train_scaled[it], y_train[it]).predict(X_train_scaled[iv]))
                              for it, iv in inner_gkf.split(X_train_scaled, y_train, groups=train_groups)]
                    if np.mean(scores) < best_score:
                        best_score = np.mean(scores)
                        best_lr = lr
                reg = GradientBoostingRegressor(learning_rate=best_lr, random_state=42).fit(X_train_scaled, y_train)
                preds[test_idx] = reg.predict(X_test_scaled)
                best_param = f"lr={best_lr}"
                
            elif config['type'] == 'gpr':
                best_kern_name = 'rbf'
                best_score = float('inf')
                for kname in config['kernels']:
                    kern = RBF() if kname == 'rbf' else Matern()
                    scores = [mean_squared_error(y_train[iv], GaussianProcessRegressor(kernel=kern, random_state=42).fit(X_train_scaled[it], y_train[it]).predict(X_train_scaled[iv]))
                              for it, iv in inner_gkf.split(X_train_scaled, y_train, groups=train_groups)]
                    if np.mean(scores) < best_score:
                        best_score = np.mean(scores)
                        best_kern_name = kname
                kern = RBF() if best_kern_name == 'rbf' else Matern()
                reg = GaussianProcessRegressor(kernel=kern, random_state=42).fit(X_train_scaled, y_train)
                preds[test_idx] = reg.predict(X_test_scaled)
                best_param = f"kernel={best_kern_name}"
            
            selected_hyperparams[model_name].append(best_param)
            
        t_elapsed = time.time() - t0
        training_times[model_name] = t_elapsed
        oof_predictions[model_name] = preds
        
        mae = mean_absolute_error(y, preds)
        mse = mean_squared_error(y, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, preds)
        
        results[model_name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'Training Time (s)': t_elapsed,
            'Hyperparams': max(set(selected_hyperparams[model_name]), key=selected_hyperparams[model_name].count)
        }
        
    df_res = pd.DataFrame(results).T
    return df_res, oof_predictions, y

print("Running 5-fold GroupKFold Cross-Validation on AXIAL dataset...")
df_res_axial, oof_axial, y_axial = evaluate_models(df_axial, 'tilt_angle', ['d_lambda1', 'd_lambda2'])

print("\n--- AXIAL REGRESSION PERFORMANCE TABLE ---")
print(df_res_axial.to_string())

print("\nRunning 5-fold GroupKFold Cross-Validation on TRANSVERSE dataset (signed angle)...")
df_res_trans_signed, oof_trans_signed, y_trans_signed = evaluate_models(df_trans, 'tilt_angle', ['d_lambda1', 'd_lambda2'])
print("\n--- TRANSVERSE REGRESSION PERFORMANCE TABLE (SIGNED) ---")
print(df_res_trans_signed.to_string())

print("\nRunning 5-fold GroupKFold Cross-Validation on TRANSVERSE dataset (absolute magnitude |theta|)...")
df_trans_abs = df_trans.copy()
df_trans_abs['abs_angle'] = df_trans['tilt_angle'].abs()
df_res_trans_abs, oof_trans_abs, y_trans_abs = evaluate_models(df_trans_abs, 'abs_angle', ['d_lambda1', 'd_lambda2'])
print("\n--- TRANSVERSE REGRESSION PERFORMANCE TABLE (ABSOLUTE MAGNITUDE |theta|) ---")
print(df_res_trans_abs.to_string())

# -------------------------------------------------------------
# GENERATING TASK 6 REQUIRED GRAPHS (1 to 7)
# -------------------------------------------------------------
print("\n" + "="*70)
print("GENERATING TASK 6 REQUIRED VISUALIZATIONS")
print("="*70)

models_list = list(df_res_axial.index)

# 1. Model-wise MAE comparison
plt.figure(figsize=(12, 6))
x = np.arange(len(models_list))
width = 0.35
plt.bar(x - width/2, df_res_axial['MAE'], width, label='Axial Tilt', color='#1f77b4', edgecolor='black', alpha=0.85)
plt.bar(x + width/2, df_res_trans_abs['MAE'], width, label='Transverse (|θ|)', color='#ff7f0e', edgecolor='black', alpha=0.85)
plt.xticks(x, models_list, rotation=35, ha='right', fontsize=10, fontweight='bold')
plt.ylabel('MAE (°)', fontsize=11, fontweight='bold')
plt.title('Graph 1: Model-wise MAE Comparison (5-Fold Grouped CV)', fontsize=13, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('output_plots/graph1_mae_comparison.png', dpi=300)
plt.close()

# 2. Model-wise RMSE comparison
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, df_res_axial['RMSE'], width, label='Axial Tilt', color='#2ca02c', edgecolor='black', alpha=0.85)
plt.bar(x + width/2, df_res_trans_abs['RMSE'], width, label='Transverse (|θ|)', color='#d62728', edgecolor='black', alpha=0.85)
plt.xticks(x, models_list, rotation=35, ha='right', fontsize=10, fontweight='bold')
plt.ylabel('RMSE (°)', fontsize=11, fontweight='bold')
plt.title('Graph 2: Model-wise RMSE Comparison (5-Fold Grouped CV)', fontsize=13, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('output_plots/graph2_rmse_comparison.png', dpi=300)
plt.close()

# 3. Model-wise R2 comparison
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, df_res_axial['R2'], width, label='Axial Tilt', color='#9467bd', edgecolor='black', alpha=0.85)
plt.bar(x + width/2, df_res_trans_abs['R2'], width, label='Transverse (|θ|)', color='#8c564b', edgecolor='black', alpha=0.85)
plt.xticks(x, models_list, rotation=35, ha='right', fontsize=10, fontweight='bold')
plt.ylabel('R² Score', fontsize=11, fontweight='bold')
plt.title('Graph 3: Model-wise R² Comparison (5-Fold Grouped CV)', fontsize=13, fontweight='bold')
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig('output_plots/graph3_r2_comparison.png', dpi=300)
plt.close()

# Identify best statistical model and best ML model for Axial
stat_models = ['Linear Regression', 'Polynomial Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net']
ml_models = ['SVR', 'Decision Tree', 'Random Forest', 'KNN', 'Gradient Boosting', 'GPR']

best_stat_model = df_res_axial.loc[stat_models]['RMSE'].idxmin()
best_ml_model = df_res_axial.loc[ml_models]['RMSE'].idxmin()
print(f"Best Statistical Model: {best_stat_model} (RMSE: {df_res_axial.loc[best_stat_model, 'RMSE']:.3f})")
print(f"Best Conventional ML Model: {best_ml_model} (RMSE: {df_res_axial.loc[best_ml_model, 'RMSE']:.3f})")

# 4. Actual vs predicted angles for best statistical model
plt.figure(figsize=(7, 6))
plt.scatter(y_axial, oof_axial[best_stat_model], color='#1f77b4', edgecolors='black', s=55, alpha=0.9, label='Predictions')
plt.plot([-100, 100], [-100, 100], 'r--', lw=2, label='Ideal 1:1 Line')
plt.title(f'Graph 4: Actual vs Predicted Angles\nBest Statistical Model: {best_stat_model}', fontsize=12, fontweight='bold')
plt.xlabel('Actual Tilt Angle (°)', fontsize=11, fontweight='bold')
plt.ylabel('Predicted Tilt Angle (°)', fontsize=11, fontweight='bold')
plt.xlim(-100, 100)
plt.ylim(-100, 100)
plt.legend()
plt.tight_layout()
plt.savefig('output_plots/graph4_actual_vs_predicted_stat.png', dpi=300)
plt.close()

# 5. Actual vs predicted angles for best ML model
plt.figure(figsize=(7, 6))
plt.scatter(y_axial, oof_axial[best_ml_model], color='#2ca02c', edgecolors='black', s=55, alpha=0.9, label='Predictions')
plt.plot([-100, 100], [-100, 100], 'r--', lw=2, label='Ideal 1:1 Line')
plt.title(f'Graph 5: Actual vs Predicted Angles\nBest Conventional ML Model: {best_ml_model}', fontsize=12, fontweight='bold')
plt.xlabel('Actual Tilt Angle (°)', fontsize=11, fontweight='bold')
plt.ylabel('Predicted Tilt Angle (°)', fontsize=11, fontweight='bold')
plt.xlim(-100, 100)
plt.ylim(-100, 100)
plt.legend()
plt.tight_layout()
plt.savefig('output_plots/graph5_actual_vs_predicted_ml.png', dpi=300)
plt.close()

# 6. Residual plots for both selected models
res_stat = y_axial - oof_axial[best_stat_model]
res_ml = y_axial - oof_axial[best_ml_model]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
axes[0].scatter(oof_axial[best_stat_model], res_stat, color='#1f77b4', edgecolors='black', s=50, alpha=0.85)
axes[0].axhline(0, color='red', linestyle='--', lw=2)
axes[0].set_title(f'Residuals: {best_stat_model}', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted Angle (°)', fontsize=11)
axes[0].set_ylabel('Residual Error (Actual - Pred) (°)', fontsize=11)
axes[0].grid(True, linestyle=':', alpha=0.6)

axes[1].scatter(oof_axial[best_ml_model], res_ml, color='#2ca02c', edgecolors='black', s=50, alpha=0.85)
axes[1].axhline(0, color='red', linestyle='--', lw=2)
axes[1].set_title(f'Residuals: {best_ml_model}', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted Angle (°)', fontsize=11)
axes[1].set_ylabel('Residual Error (Actual - Pred) (°)', fontsize=11)
axes[1].grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('output_plots/graph6_residual_plots.png', dpi=300)
plt.close()

# 7. Training time comparison across regression models
plt.figure(figsize=(11, 5.5))
times = [df_res_axial.loc[m, 'Training Time (s)'] for m in models_list]
bars = plt.bar(models_list, times, color='#e377c2', edgecolor='black', alpha=0.85)
plt.xticks(rotation=35, ha='right', fontsize=10, fontweight='bold')
plt.ylabel('Total 5-Fold Training Time (s)', fontsize=11, fontweight='bold')
plt.title('Graph 7: Training Time Comparison Across Models', fontsize=13, fontweight='bold')
for b in bars:
    yval = b.get_height()
    plt.text(b.get_x() + b.get_width()/2.0, yval + 0.005, f'{yval:.3f}s', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('output_plots/graph7_training_time_comparison.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# ADVANCED TASKS: A, B, and C
# -------------------------------------------------------------
print("\n" + "="*70)
print("ADVANCED TASKS: A (NOISE), B (HYSTERESIS), C (SIGNED VS ABS)")
print("="*70)

# ADVANCED TASK A: Noise Robustness
noise_levels = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0] # in pm
noise_results = {model: [] for model in ['Linear Regression', 'Ridge Regression', 'SVR', 'Random Forest', 'GPR']}

gkf = GroupKFold(n_splits=5)
for noise_std in noise_levels:
    np.random.seed(42)
    for model_name in noise_results:
        preds = np.zeros(len(y_axial))
        for train_idx, test_idx in gkf.split(df_axial[['d_lambda1', 'd_lambda2']].values, y_axial, groups=df_axial['tilt_angle'].values):
            X_tr = df_axial.loc[train_idx, ['d_lambda1', 'd_lambda2']].values
            y_tr = y_axial[train_idx]
            X_te = df_axial.loc[test_idx, ['d_lambda1', 'd_lambda2']].values.copy()
            # Add Gaussian noise ONLY to evaluation inputs
            if noise_std > 0:
                X_te += np.random.normal(0, noise_std, X_te.shape)
            
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
            
            if model_name == 'Linear Regression':
                m = LinearRegression().fit(X_tr_s, y_tr)
            elif model_name == 'Ridge Regression':
                m = Ridge(alpha=1.0).fit(X_tr_s, y_tr)
            elif model_name == 'SVR':
                m = SVR(kernel='rbf').fit(X_tr_s, y_tr)
            elif model_name == 'Random Forest':
                m = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr_s, y_tr)
            elif model_name == 'GPR':
                m = GaussianProcessRegressor(kernel=RBF(), random_state=42).fit(X_tr_s, y_tr)
            preds[test_idx] = m.predict(X_te_s)
            
        rmse_val = np.sqrt(mean_squared_error(y_axial, preds))
        noise_results[model_name].append(rmse_val)

plt.figure(figsize=(9, 5.5))
for m_name, rmses in noise_results.items():
    plt.plot(noise_levels, rmses, 'o-', lw=2, label=m_name)
plt.title('Advanced Task A: Noise Robustness (Gaussian Noise Injection)', fontsize=12, fontweight='bold')
plt.xlabel('Added Noise Std Dev σ (pm)', fontsize=11, fontweight='bold')
plt.ylabel('Evaluation RMSE (°)', fontsize=11, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('output_plots/advanced_taskA_noise_robustness.png', dpi=300)
plt.close()

# ADVANCED TASK B: Sensor Hysteresis
# Cycle analysis: Forward 0 -> 90 vs Return 90 -> 0; Forward 0 -> -90 vs Return -90 -> 0
fwd_pos = df_axial.iloc[0:10].set_index('tilt_angle')['d_lambda_diff']
ret_pos = df_axial.iloc[10:19].set_index('tilt_angle')['d_lambda_diff']
common_pos_angles = sorted(list(set(fwd_pos.index).intersection(set(ret_pos.index))))
hyst_pos = pd.DataFrame({
    'Forward (0->90)': fwd_pos.loc[common_pos_angles],
    'Return (90->0)': ret_pos.loc[common_pos_angles]
})
hyst_pos['Difference (Hysteresis)'] = hyst_pos['Return (90->0)'] - hyst_pos['Forward (0->90)']

print("\n--- Advanced Task B: Positive Angle Hysteresis Table (Axial d_lambda_diff) ---")
print(hyst_pos)

plt.figure(figsize=(8, 5.5))
plt.plot(df_axial.iloc[0:10]['tilt_angle'], df_axial.iloc[0:10]['d_lambda_diff'], 'o-', color='#1f77b4', lw=2, label='Forward Run (0° -> +90°)')
plt.plot(df_axial.iloc[9:19]['tilt_angle'], df_axial.iloc[9:19]['d_lambda_diff'], 's--', color='#d62728', lw=2, label='Return Run (+90° -> 0°)')
plt.title('Advanced Task B: Sensor Hysteresis Loop in Axial Tilt', fontsize=12, fontweight='bold')
plt.xlabel('Tilt Angle (°)', fontsize=11, fontweight='bold')
plt.ylabel('Differential Wavelength Change Δλ_diff (pm)', fontsize=11, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('output_plots/advanced_taskB_hysteresis.png', dpi=300)
plt.close()

# ADVANCED TASK C: Signed vs Absolute Tilt Prediction
trans_compare = pd.DataFrame({
    'Model': models_list,
    'Signed_R2': df_res_trans_signed['R2'].values,
    'Signed_RMSE': df_res_trans_signed['RMSE'].values,
    'Abs_R2': df_res_trans_abs['R2'].values,
    'Abs_RMSE': df_res_trans_abs['RMSE'].values,
})

plt.figure(figsize=(12, 5.5))
x = np.arange(len(models_list))
plt.bar(x - width/2, trans_compare['Signed_R2'], width, label='Signed θ (R²)', color='#e377c2', edgecolor='black', alpha=0.85)
plt.bar(x + width/2, trans_compare['Abs_R2'], width, label='Absolute |θ| (R²)', color='#17becf', edgecolor='black', alpha=0.85)
plt.xticks(x, models_list, rotation=35, ha='right', fontsize=10, fontweight='bold')
plt.ylabel('R² Score', fontsize=11, fontweight='bold')
plt.title('Advanced Task C: Transverse Tilt Prediction - Signed θ vs Absolute Magnitude |θ|', fontsize=13, fontweight='bold')
plt.axhline(0, color='black', lw=0.8)
plt.legend()
plt.tight_layout()
plt.savefig('output_plots/advanced_taskC_signed_vs_absolute.png', dpi=300)
plt.close()

# Save all quantitative tables to CSV and JSON for inclusion in report
df_res_axial.to_csv('output_plots/axial_results.csv')
df_res_trans_signed.to_csv('output_plots/transverse_signed_results.csv')
df_res_trans_abs.to_csv('output_plots/transverse_abs_results.csv')
df_corr_analysis.to_csv('output_plots/correlation_analysis.csv')
hyst_pos.to_csv('output_plots/hysteresis_table.csv')

print("\nALL EXPERIMENTS COMPLETED SUCCESSFULLY! All figures and data saved in output_plots/")
