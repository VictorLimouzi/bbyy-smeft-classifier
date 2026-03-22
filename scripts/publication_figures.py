#!/usr/bin/env python3
"""
Generate publication-ready figures for the write-up (outputs under figures/).

Prefer running from the repo root: ``python reproduce.py figures``
(omit stratified panels with ``reproduce.py figures --no-stratified``).
"""

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_ROOT)

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix

from plot_style import STEP_HIST_KW, apply_publication_style

apply_publication_style()

# GNN imports (optional - only needed for combined ROC figure)
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as GeoDataLoader
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, BatchNorm
    import torch.nn.functional as F
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

os.makedirs('figures', exist_ok=True)

# -----------------------------------------------------------------------------
# Load data and models
# -----------------------------------------------------------------------------
def load_dataset(fp):
    with h5py.File(fp, 'r') as f:
        return pd.DataFrame.from_records(f['ForAnalysis/1d'][:])

exclude_columns = ['EventNumber', 'is_HHEvent', 'is_SingleHiggsEvent', 'is_SingleZEvent', 'is_ZHEvent', 'Lumi_weight', 'nBTaggedJets', 'NJets']
bsm_names = ['cbbim', 'cbgim', 'cbhim', 'chbtil', 'chgtil', 'ctbim']

sm_df = load_dataset('datasets/new_Input_bbyy_SMEFT_SM_4thMarch_2026.h5')
bsm_dfs = {n: load_dataset(f'datasets/new_Input_bbyy_SMEFT_{n}_4thMarch_2026.h5') for n in bsm_names}
feature_columns = [c for c in sm_df.columns if c not in exclude_columns]

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cpu')


class SMvsBSMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hd), nn.BatchNorm1d(hd), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hd
        layers.extend([nn.Linear(prev_dim, 1), nn.Sigmoid()])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


def _get_event_type(row):
    if row.get('is_ZHEvent', False):
        return 'ZH'
    if row.get('is_HHEvent', False):
        return 'HH'
    if row.get('is_SingleHiggsEvent', False):
        return 'Single H'
    return 'Other'


def _stratified_sample_sm_bsm(sm_df, bsm_df, random_state=42):
    """Stratify by HH, Single H, Other (ZH when present). Cap each stratum at min(n_SM, n_BSM); skip strata with zero on either side."""
    sm_copy = sm_df.copy()
    bsm_copy = bsm_df.copy()
    sm_copy['_event_type'] = sm_copy.apply(_get_event_type, axis=1)
    bsm_copy['_event_type'] = bsm_copy.apply(_get_event_type, axis=1)
    sm_samples, bsm_samples = [], []
    for etype in ['ZH', 'HH', 'Single H', 'Other']:
        sm_sub = sm_copy[sm_copy['_event_type'] == etype]
        bsm_sub = bsm_copy[bsm_copy['_event_type'] == etype]
        n_min = min(len(sm_sub), len(bsm_sub))
        if n_min > 0:
            sm_samples.append(sm_sub.sample(n=n_min, random_state=random_state))
            bsm_samples.append(bsm_sub.sample(n=n_min, random_state=random_state))
    if sm_samples and bsm_samples:
        return pd.concat(sm_samples, ignore_index=True), pd.concat(bsm_samples, ignore_index=True)
    n_min = min(len(sm_df), len(bsm_df))
    return sm_df.sample(n=n_min, random_state=random_state), bsm_df.sample(n=n_min, random_state=random_state)


def prepare_binary_data(sm_df, bsm_df, feature_columns, balance=True):
    if balance:
        sm_sample, bsm_sample = _stratified_sample_sm_bsm(sm_df, bsm_df)
        X_sm = sm_sample[feature_columns].values
        X_bsm = bsm_sample[feature_columns].values
    else:
        X_sm = sm_df[feature_columns].values
        X_bsm = bsm_df[feature_columns].values
    y_sm = np.zeros(len(X_sm))
    y_bsm = np.ones(len(X_bsm))
    return np.vstack([X_sm, X_bsm]), np.concatenate([y_sm, y_bsm])


def load_model(bsm_name):
    """Load DNN classifier for given BSM operator."""
    ckpt = torch.load(f'models/sm_vs_{bsm_name}_classifier.pt', map_location='cpu', weights_only=False)
    model = SMvsBSMClassifier(ckpt['input_dim'], ckpt['hidden_dims'], dropout=0.3)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def bootstrap_auc(y_true, probs, n_bootstrap=1000, confidence=0.95, random_state=42):
    """Bootstrap AUC: resample test set with replacement, compute AUC per sample."""
    np.random.seed(random_state)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        p_b = probs[idx]
        if len(np.unique(y_b)) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_b, p_b)
        aucs.append(auc(fpr_b, tpr_b))
    aucs = np.array(aucs)
    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(aucs, 100 * alpha)
    ci_upper = np.percentile(aucs, 100 * (1 - alpha))
    return np.mean(aucs), np.std(aucs), ci_lower, ci_upper


def bootstrap_roc_band(y_true, probs, n_bootstrap=200, fpr_grid=None, confidence=0.95, random_state=42):
    """Bootstrap ROC curve: return (fpr_grid, tpr_lower, tpr_upper) for confidence band."""
    np.random.seed(random_state)
    n = len(y_true)
    if fpr_grid is None:
        fpr_grid = np.linspace(0, 1, 101)
    tprs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        p_b = probs[idx]
        if len(np.unique(y_b)) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_b, p_b)
        tpr_interp = np.interp(fpr_grid, fpr_b, tpr_b)
        tprs.append(tpr_interp)
    if not tprs:
        return fpr_grid, np.zeros_like(fpr_grid), np.ones_like(fpr_grid)
    tprs = np.array(tprs)
    alpha = (1 - confidence) / 2
    tpr_lower = np.percentile(tprs, 100 * alpha, axis=0)
    tpr_upper = np.percentile(tprs, 100 * (1 - alpha), axis=0)
    return fpr_grid, tpr_lower, tpr_upper


def load_model_and_eval(bsm_name):
    ckpt = torch.load(f'models/sm_vs_{bsm_name}_classifier.pt', map_location='cpu', weights_only=False)
    model = SMvsBSMClassifier(ckpt['input_dim'], ckpt['hidden_dims'], dropout=0.3)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Use checkpoint's feature columns if available (for compatibility)
    feat_cols = ckpt.get('feature_columns', feature_columns)
    X, y = prepare_binary_data(sm_df, bsm_dfs[bsm_name], feat_cols)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Use checkpoint's scaler for consistency with training
    scaler = StandardScaler()
    scaler.mean_ = ckpt['scaler_mean']
    scaler.scale_ = ckpt['scaler_scale']
    X_test_scaled = scaler.transform(X_test)
    X_t = torch.FloatTensor(X_test_scaled)
    with torch.no_grad():
        probs = model(X_t).numpy()
    preds = (probs > 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    auc_mean, auc_std, _, _ = bootstrap_auc(y_test, probs, n_bootstrap=1000)
    return fpr, tpr, roc_auc, auc_mean, auc_std, probs, preds, y_test


# -----------------------------------------------------------------------------
# 1. ROC + confusion matrix (combined on same figure per operator)
# -----------------------------------------------------------------------------
print("Generating ROC + confusion matrix figures...")
for bsm_name in bsm_names:
    fpr, tpr, roc_auc, auc_mean, auc_std, probs, preds, labels = load_model_and_eval(bsm_name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Left: ROC with 95% confidence band
    fpr_g, tpr_lo, tpr_hi = bootstrap_roc_band(labels, probs, n_bootstrap=200)
    axes[0].fill_between(fpr_g, tpr_lo, tpr_hi, color='#2E86AB', alpha=0.2)
    axes[0].plot(fpr, tpr, color='#2E86AB', lw=2, label=f'DNN (AUC = {auc_mean:.4f} ± {auc_std:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title(f'ROC Curve: SM vs {bsm_name}', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    # Right: Confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_frac = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    im = axes[1].imshow(cm_frac, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[1], format='%.2f')
    axes[1].set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=['SM', bsm_name], yticklabels=['SM', bsm_name],
               xlabel='Predicted', ylabel='True',
               title=f'Confusion Matrix: SM vs {bsm_name}')
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f'{cm_frac[i, j]:.3f}\n({cm[i, j]:,})',
                         ha='center', va='center',
                         color='white' if cm_frac[i, j] > 0.5 else 'black', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'figures/nn_roc_confusion_{bsm_name}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/nn_roc_confusion_{bsm_name}.png")
    # Confusion matrix only — nn_confusion_{operator}.png
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_frac, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, format='%.2f')
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=['SM', bsm_name], yticklabels=['SM', bsm_name],
           xlabel='Predicted', ylabel='True',
           title=f'NN Confusion Matrix: SM vs {bsm_name}')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm_frac[i, j]:.3f}\n({cm[i, j]:,})',
                    ha='center', va='center',
                    color='white' if cm_frac[i, j] > 0.5 else 'black', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'figures/nn_confusion_{bsm_name}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/nn_confusion_{bsm_name}.png")

# -----------------------------------------------------------------------------
# 1b. NN learned distribution (P(BSM)) per operator — nn_score_{operator}.png
# -----------------------------------------------------------------------------
print("Generating NN learned distribution (P(BSM)) per operator...")
for bsm_name in bsm_names:
    _, _, _, _, _, probs, _, labels = load_model_and_eval(bsm_name)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(probs[labels == 0], bins=50, density=True, label='SM', color='blue', **STEP_HIST_KW)
    ax.hist(probs[labels == 1], bins=50, density=True, label=bsm_name, color='red', **STEP_HIST_KW)
    ax.set_xlabel('P(BSM)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'NN learned distribution: SM vs {bsm_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(f'figures/nn_score_{bsm_name}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/nn_score_{bsm_name}.png")

# -----------------------------------------------------------------------------
# 1c. DNN top 10 feature importance (one PNG per operator)
# -----------------------------------------------------------------------------
if os.path.exists('metrics/nn_feature_importance.csv'):
    print("Generating DNN feature importance (one per dataset)...")
    fi_df = pd.read_csv('metrics/nn_feature_importance.csv')
    # Global x-axis limits from all top-10 values (for consistent scale)
    all_imp = []
    for op in bsm_names:
        sub = fi_df[fi_df['operator'] == op].nlargest(10, 'importance')
        all_imp.extend(sub['importance'].tolist())
    x_min, x_max = min(all_imp), max(all_imp)
    x_margin = (x_max - x_min) * 0.05 or 0.01
    x_lim = (x_min - x_margin, x_max + x_margin)
    for op in bsm_names:
        sub = fi_df[fi_df['operator'] == op].nlargest(10, 'importance')
        features = sub['feature'].tolist()[::-1]
        imp = sub['importance'].tolist()[::-1]
        colors = ['#E94F37' if f == 'Higgs_Mass' else '#2E86AB' for f in features]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(features, imp, color=colors, alpha=0.85)
        ax.set_xlim(x_lim)
        ax.set_xlabel('Permutation importance', fontsize=12)
        ax.set_title(f'DNN top 10 features: {op}', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'figures/nn_feature_importance_top10_{op}.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved figures/nn_feature_importance_top10_{op}.png")

# -----------------------------------------------------------------------------
# 2b. NN training curves for cbgim
# -----------------------------------------------------------------------------
print("Generating NN training curves (cbgim)...")
X, y = prepare_binary_data(sm_df, bsm_dfs['cbgim'], feature_columns)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.125, random_state=42, stratify=y_trainval)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
train_ds = TensorDataset(torch.FloatTensor(X_train_s), torch.FloatTensor(y_train))
val_ds = TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor(y_val))
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)

model = SMvsBSMClassifier(len(feature_columns), [128, 64, 32], dropout=0.3)
opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.BCELoss()

train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val, patience, max_epochs = float('inf'), 0, 100
for epoch in range(max_epochs):
    model.train()
    tl, tc, tt = 0, 0, 0
    for xb, yb in train_loader:
        opt.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        opt.step()
        tl += loss.item() * len(yb)
        tc += ((out > 0.5).float() == yb).sum().item()
        tt += len(yb)
    train_losses.append(tl / tt)
    train_accs.append(tc / tt)
    model.eval()
    vl, vc, vt = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb)
            vl += criterion(out, yb).item() * len(yb)
            vc += ((out > 0.5).float() == yb).sum().item()
            vt += len(yb)
    val_losses.append(vl / vt)
    val_accs.append(vc / vt)
    if val_losses[-1] < best_val:
        best_val = val_losses[-1]
        patience = 0
    else:
        patience += 1
    if patience >= 15:
        break

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, 'b-', lw=2, label='Train')
axes[0].plot(val_losses, 'r-', lw=2, label='Validation')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (BCE)', fontsize=12)
axes[0].set_title('Loss: SM vs cbgim', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].plot(train_accs, 'b-', lw=2, label='Train')
axes[1].plot(val_accs, 'r-', lw=2, label='Validation')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Accuracy: SM vs cbgim', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.suptitle('DNN Training Curves', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/nn_training_curves_cbgim.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved figures/nn_training_curves_cbgim.png")

# -----------------------------------------------------------------------------
# 3. Model comparison (DNN vs GCN vs GAT across operators) — AUC plot
# -----------------------------------------------------------------------------
# Build combined dataframe from nn_comparison and gnn_comparison if all_models_comparison doesn't exist
if os.path.exists('metrics/all_models_comparison.csv'):
    df = pd.read_csv('metrics/all_models_comparison.csv')
elif os.path.exists('metrics/nn_comparison.csv') and os.path.exists('metrics/gnn_comparison.csv'):
    nn_df = pd.read_csv('metrics/nn_comparison.csv')
    gnn_df = pd.read_csv('metrics/gnn_comparison.csv')
    # nn_comparison: BSM_Operator, Architecture (DNN), AUC, AUC_mean, AUC_std, ...
    # gnn_comparison: BSM_Operator, Architecture (GCN/GAT), AUC, AUC_mean, AUC_std, ...
    base_cols = ['BSM_Operator', 'Architecture', 'AUC', 'Training_Time_s']
    extra_cols = ['AUC_mean', 'AUC_std'] if ('AUC_std' in nn_df.columns or 'AUC_std' in gnn_df.columns) else []
    cols = base_cols + extra_cols
    nn_cols = [c for c in cols if c in nn_df.columns]
    gnn_cols = [c for c in cols if c in gnn_df.columns]
    df = pd.concat([nn_df[nn_cols], gnn_df[gnn_cols]], ignore_index=True)
else:
    df = None

if df is not None:
    print("Generating AUC comparison (DNN vs GCN vs GAT)...")
    x = np.arange(len(bsm_names))
    width = 0.25
    colors = ['#2E86AB', '#E94F37', '#44AF69']  # DNN, GCN, GAT
    # AUC-only plot (single figure)
    fig, ax = plt.subplots(figsize=(10, 5))
    has_std = 'AUC_std' in df.columns
    for i, arch in enumerate(['DNN', 'GCN', 'GAT']):
        aucs = []
        errs = []
        for b in bsm_names:
            row = df[(df['BSM_Operator'] == b) & (df['Architecture'] == arch)]
            if len(row) > 0:
                auc_val = (row['AUC_mean'].values[0] if has_std and 'AUC_mean' in row.columns else row['AUC'].values[0])
                if np.isnan(auc_val):
                    auc_val = row['AUC'].values[0]
                aucs.append(auc_val)
                e = row['AUC_std'].values[0] if has_std else 0
                errs.append(0 if np.isnan(e) else e)
            else:
                aucs.append(np.nan)
                errs.append(0)
        ax.bar(x + (i - 1) * width, aucs, width, yerr=errs if has_std else None, label=arch, color=colors[i], alpha=0.85, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(bsm_names, rotation=25, ha='right')
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('AUC by BSM Operator: DNN vs GCN vs GAT', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/auc_dnn_gcn_gat.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved figures/auc_dnn_gcn_gat.png")
    # Full comparison (AUC + training time)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, arch in enumerate(['DNN', 'GCN', 'GAT']):
        aucs = []
        errs = []
        for b in bsm_names:
            row = df[(df['BSM_Operator'] == b) & (df['Architecture'] == arch)]
            if len(row) > 0:
                auc_val = (row['AUC_mean'].values[0] if has_std and 'AUC_mean' in row.columns else row['AUC'].values[0])
                if np.isnan(auc_val):
                    auc_val = row['AUC'].values[0]
                aucs.append(auc_val)
                e = row['AUC_std'].values[0] if has_std else 0
                errs.append(0 if np.isnan(e) else e)
            else:
                aucs.append(np.nan)
                errs.append(0)
        axes[0].bar(x + (i - 1) * width, aucs, width, yerr=errs if has_std else None, label=arch, color=colors[i], alpha=0.85, capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bsm_names, rotation=25, ha='right')
    axes[0].set_ylabel('AUC', fontsize=12)
    axes[0].set_title('AUC by BSM Operator', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0].grid(axis='y', alpha=0.3)
    for i, arch in enumerate(['DNN', 'GCN', 'GAT']):
        times = [df[(df['BSM_Operator'] == b) & (df['Architecture'] == arch)]['Training_Time_s'].values[0] for b in bsm_names]
        axes[1].bar(x + (i - 1) * width, times, width, label=arch, color=colors[i], alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bsm_names, rotation=25, ha='right')
    axes[1].set_ylabel('Training Time (s)', fontsize=12)
    axes[1].set_title('Training Time by BSM Operator', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved figures/model_comparison.png")
else:
    print("Skipping model comparison (no metrics found)")

# -----------------------------------------------------------------------------
# 4. ROC for DNN, GCN, GAT on same axes — one figure per dataset
# -----------------------------------------------------------------------------
print("Generating ROC (DNN vs GCN vs GAT) for each dataset...")

if HAS_TORCH_GEOMETRIC:
    def event_to_graph(row, label):
        node_features = torch.tensor([
            [row['Higgs_pT'], row['Higgs_Eta'], row['Higgs_Phi'], row['Higgs_Mass']],
            [row['LeadJet_pT'], row['LeadJet_Eta'], row['LeadJet_Phi'], row['LeadJet_M']],
            [row['SubLeadJet_pT'], row['SubLeadJet_Eta'], row['SubLeadJet_Phi'], row['SubLeadJet_M']],
            [row['LeadPhoton_pT'], row['LeadPhoton_Eta'], row['LeadPhoton_Phi'], 0.0],
            [row['SubLeadPhoton_pT'], row['SubLeadPhoton_Eta'], row['SubLeadPhoton_Phi'], 0.0],
        ], dtype=torch.float)
        edge_index = torch.tensor([[i, j] for i in range(5) for j in range(5) if i != j], dtype=torch.long).t().contiguous()
        global_features = torch.tensor([
            row['m_bbyy'], row['pT_jj'], row['Eta_jj'], row['DPhi_bb'],
            row['signed_DeltaPhi_jj'], row['cosThetaStar'], row['costheta1'], row['costheta2'],
            row['DPhi_yybb'], row['Eta_yybb'],
        ], dtype=torch.float)
        return Data(x=node_features, edge_index=edge_index, u=global_features, y=torch.tensor([label], dtype=torch.float))

    class GCN_Classifier(nn.Module):
        def __init__(self, node_features=4, global_features=10, hidden_dim=64, num_layers=3, dropout=0.3):
            super().__init__()
            self.num_layers = num_layers
            self.n_global = global_features
            self.convs = nn.ModuleList([GCNConv(node_features, hidden_dim)] + [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
            self.bns = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers)])
            combined_dim = hidden_dim * 2 + global_features
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())
            self.dropout = dropout
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            gf = data.u.view(-1, self.n_global)
            return self.classifier(torch.cat([x_mean, x_max, gf], dim=1)).squeeze()

    class GAT_Classifier(nn.Module):
        def __init__(self, node_features=4, global_features=10, hidden_dim=32, num_layers=3, heads=4, dropout=0.3):
            super().__init__()
            self.num_layers = num_layers
            self.n_global = global_features
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.convs.append(GATConv(node_features, hidden_dim, heads=heads, dropout=dropout))
            self.bns.append(BatchNorm(hidden_dim * heads))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
                self.bns.append(BatchNorm(hidden_dim * heads))
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout))
            self.bns.append(BatchNorm(hidden_dim))
            combined_dim = hidden_dim * 2 + global_features
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())
            self.dropout = dropout
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                x = self.bns[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            gf = data.u.view(-1, self.n_global)
            return self.classifier(torch.cat([x_mean, x_max, gf], dim=1)).squeeze()

all_roc_data = {}  # bsm_name -> list of (name, fpr, tpr, auc_val, color)

for bsm_name in bsm_names:
    n_min = min(len(sm_df), len(bsm_dfs[bsm_name]))
    sm_sample = sm_df.sample(n=n_min, random_state=42)
    bsm_sample = bsm_dfs[bsm_name].sample(n=n_min, random_state=42)
    y_all = np.array([0] * n_min + [1] * n_min)
    n_total = len(y_all)
    np.random.seed(42)
    idx = np.random.permutation(n_total)
    n_train, n_val = int(0.7 * n_total), int(0.1 * n_total)

    # DNN: same pipeline as nn_roc_confusion_* (checkpoint scaler, stratified balance, 80/20 test).
    # The previous code refit StandardScaler on a 70/10/20 split, so AUC did not match training or CSV.
    fpr_dnn, tpr_dnn, _, auc_dnn_mean, auc_dnn_std, probs_dnn, _, y_test_dnn = load_model_and_eval(bsm_name)
    roc_curves = [('DNN', fpr_dnn, tpr_dnn, auc_dnn_mean, auc_dnn_std, '#2E86AB', y_test_dnn, probs_dnn)]

    # GCN and GAT
    if HAS_TORCH_GEOMETRIC:
        all_rows = pd.concat([sm_sample, bsm_sample], ignore_index=True)
        graphs = [event_to_graph(all_rows.iloc[i], y_all[i]) for i in range(len(all_rows))]
        graphs_shuffled = [graphs[i] for i in idx]
        test_graphs = graphs_shuffled[n_train + n_val:]
        for arch, ckpt_name, color in [('GCN', f'gcn_{bsm_name}_classifier', '#E94F37'), ('GAT', f'gat_{bsm_name}_classifier', '#44AF69')]:
            try:
                ckpt = torch.load(f'models/{ckpt_name}.pt', map_location='cpu', weights_only=False)
                model = GCN_Classifier() if arch == 'GCN' else GAT_Classifier()
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()
                all_probs, all_labels = [], []
                test_loader = GeoDataLoader(test_graphs, batch_size=64, shuffle=False)
                with torch.no_grad():
                    for batch in test_loader:
                        probs = model(batch).cpu().numpy()
                        all_probs.extend(np.atleast_1d(probs).ravel().tolist())
                        all_labels.extend(np.atleast_1d(batch.y.numpy()).ravel().tolist())
                fpr, tpr, _ = roc_curve(all_labels, all_probs)
                auc_m, auc_s, _, _ = bootstrap_auc(np.array(all_labels), np.array(all_probs), n_bootstrap=1000)
                roc_curves.append((arch, fpr, tpr, auc_m, auc_s, color, np.array(all_labels), np.array(all_probs)))
            except FileNotFoundError:
                pass

    all_roc_data[bsm_name] = roc_curves

    # Per-dataset figure
    fig, ax = plt.subplots(figsize=(10, 8))
    for item in roc_curves:
        name, fpr, tpr, auc_mean, auc_std, color = item[:6]
        if len(item) >= 8:
            fpr_g, tpr_lo, tpr_hi = bootstrap_roc_band(item[6], item[7], n_bootstrap=200)
            ax.fill_between(fpr_g, tpr_lo, tpr_hi, color=color, alpha=0.2)
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{name} (AUC = {auc_mean:.4f} ± {auc_std:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves: SM vs {bsm_name} — DNN vs GCN vs GAT', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'figures/roc_dnn_gcn_gat_{bsm_name}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/roc_dnn_gcn_gat_{bsm_name}.png")

# Combined figure: all datasets in one plot (6 subplots)
print("Generating combined ROC figure (all datasets)...")
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
for idx, bsm_name in enumerate(bsm_names):
    ax = axes[idx]
    roc_curves = all_roc_data.get(bsm_name, [])
    for item in roc_curves:
        name, fpr, tpr, auc_mean, auc_std, color = item[:6]
        if len(item) >= 8:
            fpr_g, tpr_lo, tpr_hi = bootstrap_roc_band(item[6], item[7], n_bootstrap=200)
            ax.fill_between(fpr_g, tpr_lo, tpr_hi, color=color, alpha=0.15)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc_mean:.3f}±{auc_std:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title(f'SM vs {bsm_name}', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
plt.suptitle('ROC Curves: DNN vs GCN vs GAT per BSM Operator', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/roc_all_datasets_combined.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved figures/roc_all_datasets_combined.png")

if not HAS_TORCH_GEOMETRIC:
    print("  (GCN/GAT skipped — install torch-geometric for full comparison)")

print("\nAll figures generated in figures/")
