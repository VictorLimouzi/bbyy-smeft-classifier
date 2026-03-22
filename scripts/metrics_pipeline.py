#!/usr/bin/env python3
"""
Unified metrics exports (subcommands):

  comparison   — metrics/all_models_comparison.csv (DNN, GCN, GAT + bootstrap)
  nn-importance — metrics/nn_feature_importance.csv (permutation on test set)
  top-overlap  — metrics/top_observables_overlap_separation.csv (EDA overlap metric)
  all          — run the three steps in order
"""

import argparse
import os
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as GeoDataLoader
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, BatchNorm
    import torch.nn.functional as F
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

os.makedirs('metrics', exist_ok=True)

# -----------------------------------------------------------------------------
# Load data
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


def _bootstrap_stats(values, confidence=0.95):
    """Return mean, std, ci_lower, ci_upper for a list of bootstrap values."""
    vals = np.array(values)
    alpha = (1 - confidence) / 2
    return np.mean(vals), np.std(vals), np.percentile(vals, 100 * alpha), np.percentile(vals, 100 * (1 - alpha))


def bootstrap_metrics(y_true, probs, n_bootstrap=1000, confidence=0.95, random_state=42):
    """
    Bootstrap all metrics: resample test set with replacement, compute each metric per sample.
    Returns dict with mean, std, ci_lower, ci_upper for AUC, Accuracy, Precision, Recall, F1.
    """
    np.random.seed(random_state)
    n = len(y_true)
    preds = (probs > 0.5).astype(int)
    aucs, accs, precs, recs, f1s = [], [], [], [], []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        p_b = probs[idx]
        preds_b = (p_b > 0.5).astype(int)
        # AUC
        if len(np.unique(y_b)) >= 2:
            fpr_b, tpr_b, _ = roc_curve(y_b, p_b)
            aucs.append(auc(fpr_b, tpr_b))
        # Classification metrics (use zero_division=0 for edge cases)
        accs.append(accuracy_score(y_b, preds_b))
        precs.append(precision_score(y_b, preds_b, zero_division=0))
        recs.append(recall_score(y_b, preds_b, zero_division=0))
        f1s.append(f1_score(y_b, preds_b, zero_division=0))
    result = {}
    if aucs:
        result['AUC'] = _bootstrap_stats(aucs, confidence)
    result['Accuracy'] = _bootstrap_stats(accs, confidence)
    result['Precision'] = _bootstrap_stats(precs, confidence)
    result['Recall'] = _bootstrap_stats(recs, confidence)
    result['F1'] = _bootstrap_stats(f1s, confidence)
    return result


# -----------------------------------------------------------------------------
# DNN model
# -----------------------------------------------------------------------------
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


def prepare_binary_data_nn(sm_df, bsm_df, feature_columns, balance=True):
    """Match 03_neural_network_classification.ipynb: simple random undersampling."""
    X_sm = sm_df[feature_columns].values
    X_bsm = bsm_df[feature_columns].values
    y_sm = np.zeros(len(X_sm))
    y_bsm = np.ones(len(X_bsm))
    if balance:
        n_min = min(len(X_sm), len(X_bsm))
        idx_sm = np.random.choice(len(X_sm), n_min, replace=False)
        idx_bsm = np.random.choice(len(X_bsm), n_min, replace=False)
        X_sm, y_sm = X_sm[idx_sm], y_sm[idx_sm]
        X_bsm, y_bsm = X_bsm[idx_bsm], y_bsm[idx_bsm]
    return np.vstack([X_sm, X_bsm]), np.concatenate([y_sm, y_bsm])


# -----------------------------------------------------------------------------
# GNN data prep (match 04_graph_neural_network_classification.ipynb create_graph_dataset)
# -----------------------------------------------------------------------------
def _get_event_type(row):
    if row.get('is_ZHEvent', False):
        return 'ZH'
    if row.get('is_HHEvent', False):
        return 'HH'
    if row.get('is_SingleHiggsEvent', False):
        return 'Single H'
    return 'Other'


def create_graph_dataset(sm_df, bsm_df, balance=True):
    """Match 04_graph_neural_network_classification.ipynb: stratified by event type."""
    graphs = []
    if balance:
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
                sm_samples.append(sm_sub.sample(n=n_min, random_state=42))
                bsm_samples.append(bsm_sub.sample(n=n_min, random_state=42))
        if sm_samples and bsm_samples:
            sm_sample = pd.concat(sm_samples, ignore_index=True)
            bsm_sample = pd.concat(bsm_samples, ignore_index=True)
        else:
            n_min = min(len(sm_df), len(bsm_df))
            sm_sample = sm_df.sample(n=n_min, random_state=42)
            bsm_sample = bsm_df.sample(n=n_min, random_state=42)
    else:
        n_min = min(len(sm_df), len(bsm_df))
        sm_sample = sm_df.sample(n=n_min, random_state=42)
        bsm_sample = bsm_df.sample(n=n_min, random_state=42)
    all_rows = pd.concat([sm_sample, bsm_sample], ignore_index=True)
    y_all = np.array([0] * len(sm_sample) + [1] * len(bsm_sample))
    for i in range(len(all_rows)):
        graphs.append((all_rows.iloc[i], y_all[i]))
    return graphs


# -----------------------------------------------------------------------------
# GNN models (from generate_figures)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Main: evaluate all models and save CSV
# -----------------------------------------------------------------------------
def run_model_comparison():
    rows = []
    n_bootstrap = 1000

    # Load existing metrics for training time (optional)
    nn_metrics = pd.read_csv('metrics/nn_comparison.csv') if os.path.exists('metrics/nn_comparison.csv') else None
    gnn_metrics = pd.read_csv('metrics/gnn_comparison.csv') if os.path.exists('metrics/gnn_comparison.csv') else None

    def get_training_time(bsm, arch):
        df = nn_metrics if arch == 'DNN' else gnn_metrics
        if df is None:
            return np.nan
        r = df[(df['BSM_Operator'] == bsm) & (df['Architecture'] == arch)]
        return r['Training_Time_s'].values[0] if len(r) > 0 else np.nan

    # --- DNN ---
    print("Evaluating DNN models...")
    for bsm_name in bsm_names:
        try:
            ckpt = torch.load(f'models/sm_vs_{bsm_name}_classifier.pt', map_location='cpu', weights_only=False)
            model = SMvsBSMClassifier(ckpt['input_dim'], ckpt['hidden_dims'], dropout=0.3)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()

            feat_cols = ckpt.get('feature_columns', feature_columns)
            X, y = prepare_binary_data_nn(sm_df, bsm_dfs[bsm_name], feat_cols)
            X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            scaler = StandardScaler()
            scaler.mean_ = ckpt['scaler_mean']
            scaler.scale_ = ckpt['scaler_scale']
            X_test_scaled = scaler.transform(X_test)

            t0 = time.time()
            with torch.no_grad():
                probs = model(torch.FloatTensor(X_test_scaled)).numpy().ravel()
            inf_time = time.time() - t0
            preds = (probs > 0.5).astype(int)

            boot = bootstrap_metrics(y_test, probs, n_bootstrap=n_bootstrap)
            row = {
                'BSM_Operator': bsm_name,
                'Architecture': 'DNN',
                'AUC': auc(*roc_curve(y_test, probs)[:2]),
                'Accuracy': accuracy_score(y_test, preds),
                'Precision': precision_score(y_test, preds, zero_division=0),
                'Recall': recall_score(y_test, preds, zero_division=0),
                'F1': f1_score(y_test, preds, zero_division=0),
                'Training_Time_s': get_training_time(bsm_name, 'DNN'),
                'Inference_Time_s': inf_time,
                'Events_per_Second': len(y_test) / inf_time,
            }
            for metric in ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']:
                if metric in boot:
                    m, s, lo, hi = boot[metric]
                    row[f'{metric}_mean'] = m
                    row[f'{metric}_std'] = s
                    row[f'{metric}_ci_lower'] = lo
                    row[f'{metric}_ci_upper'] = hi
            rows.append(row)
        except FileNotFoundError:
            print(f"  Skipping DNN {bsm_name} (model not found)")

    # --- GCN and GAT ---
    if HAS_TORCH_GEOMETRIC:
        print("Evaluating GCN and GAT models...")
        for bsm_name in bsm_names:
            graph_dataset = create_graph_dataset(sm_df, bsm_dfs[bsm_name])
            np.random.seed(42)
            np.random.shuffle(graph_dataset)
            n_total = len(graph_dataset)
            n_train = int(0.7 * n_total)
            n_val = int(0.1 * n_total)
            test_dataset = graph_dataset[n_train + n_val:]
            test_graphs = [event_to_graph(row, label) for row, label in test_dataset]
            y_test = np.array([label for _, label in test_dataset])

            for arch, ckpt_name in [('GCN', f'gcn_{bsm_name}_classifier'), ('GAT', f'gat_{bsm_name}_classifier')]:
                try:
                    ckpt = torch.load(f'models/{ckpt_name}.pt', map_location='cpu', weights_only=False)
                    model = GCN_Classifier() if arch == 'GCN' else GAT_Classifier()
                    model.load_state_dict(ckpt['model_state_dict'])
                    model.eval()

                    test_loader = GeoDataLoader(test_graphs, batch_size=64, shuffle=False)
                    all_probs, all_labels = [], []
                    t0 = time.time()
                    with torch.no_grad():
                        for batch in test_loader:
                            probs = model(batch).cpu().numpy()
                            all_probs.extend(np.atleast_1d(probs).ravel().tolist())
                            all_labels.extend(np.atleast_1d(batch.y.numpy()).ravel().tolist())
                    inf_time = time.time() - t0
                    probs = np.array(all_probs)
                    labels = np.array(all_labels)
                    preds = (probs > 0.5).astype(int)

                    boot = bootstrap_metrics(labels, probs, n_bootstrap=n_bootstrap)
                    row = {
                        'BSM_Operator': bsm_name,
                        'Architecture': arch,
                        'AUC': auc(*roc_curve(labels, probs)[:2]),
                        'Accuracy': accuracy_score(labels, preds),
                        'Precision': precision_score(labels, preds, zero_division=0),
                        'Recall': recall_score(labels, preds, zero_division=0),
                        'F1': f1_score(labels, preds, zero_division=0),
                        'Training_Time_s': get_training_time(bsm_name, arch),
                        'Inference_Time_s': inf_time,
                        'Events_per_Second': len(labels) / inf_time,
                    }
                    for metric in ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']:
                        if metric in boot:
                            m, s, lo, hi = boot[metric]
                            row[f'{metric}_mean'] = m
                            row[f'{metric}_std'] = s
                            row[f'{metric}_ci_lower'] = lo
                            row[f'{metric}_ci_upper'] = hi
                    rows.append(row)
                except FileNotFoundError:
                    print(f"  Skipping {arch} {bsm_name} (model not found)")

    df = pd.DataFrame(rows)
    out_path = 'metrics/all_models_comparison.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")
    print(df.to_string(index=False))


# -----------------------------------------------------------------------------
# NN permutation importance (was compute_nn_feature_importance_all.py)
# -----------------------------------------------------------------------------
BSM_OPERATORS_NN = ["cbbim", "cbgim", "cbhim", "chbtil", "chgtil", "ctbim"]
EXCLUDE_NN = [
    "EventNumber",
    "is_HHEvent",
    "is_SingleHiggsEvent",
    "is_SingleZEvent",
    "is_ZHEvent",
    "Lumi_weight",
    "nBTaggedJets",
    "NJets",
]


def run_nn_importance():
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sm_path = _ROOT / "datasets" / "new_Input_bbyy_SMEFT_SM_4thMarch_2026.h5"
    if not sm_path.is_file():
        raise SystemExit(f"Missing {sm_path}")

    sm_df = load_dataset(str(sm_path))
    feature_columns_nn = [c for c in sm_df.columns if c not in EXCLUDE_NN]

    os.makedirs(_ROOT / "metrics", exist_ok=True)
    all_rows = []

    for bsm in BSM_OPERATORS_NN:
        ckpt_path = _ROOT / "models" / f"sm_vs_{bsm}_classifier.pt"
        bsm_path = _ROOT / "datasets" / f"new_Input_bbyy_SMEFT_{bsm}_4thMarch_2026.h5"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing model: {ckpt_path}")
        if not bsm_path.is_file():
            raise FileNotFoundError(f"Missing dataset: {bsm_path}")

        bsm_df = load_dataset(str(bsm_path))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        feat_cols = list(ckpt.get("feature_columns", feature_columns_nn))
        X, y = prepare_binary_data_nn(sm_df, bsm_df, feat_cols)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        scaler.mean_ = ckpt["scaler_mean"]
        scaler.scale_ = ckpt["scaler_scale"]
        X_test_s = scaler.transform(X_test)

        model = SMvsBSMClassifier(ckpt["input_dim"], ckpt.get("hidden_dims", [128, 64, 32]), dropout=0.3)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        print(f"Permutation importance: {bsm} ({len(feat_cols)} features, n_test={len(y_test)})...")
        imp = _permutation_importance(model, X_test_s, y_test, feat_cols, device, n_repeats=5)
        for feat, val in imp.items():
            all_rows.append({"operator": bsm, "feature": feat, "importance": val})

    out_df = pd.DataFrame(all_rows)
    out_path = _ROOT / "metrics" / "nn_feature_importance.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


def _permutation_importance(model, X_test, y_test, feature_names, device, n_repeats=5):
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        baseline_probs = model(X_test_t).cpu().numpy()
    baseline_preds = (baseline_probs > 0.5).astype(int)
    baseline_acc = (baseline_preds == y_test).mean()

    out = {}
    for idx, feature in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            np.random.shuffle(Xp[:, idx])
            with torch.no_grad():
                pp = model(torch.FloatTensor(Xp).to(device)).cpu().numpy()
            pr = (pp > 0.5).astype(int)
            drops.append(baseline_acc - (pr == y_test).mean())
        out[feature] = float(np.mean(drops))
    return out


# -----------------------------------------------------------------------------
# Top observables by histogram overlap (was export_top_overlap_observables.py)
# -----------------------------------------------------------------------------
BSM_NAMES_OVERLAP = ["cbbim", "cbgim", "cbhim", "chbtil", "chgtil", "ctbim"]
EXCLUDE_OVERLAP = [
    "EventNumber",
    "is_HHEvent",
    "is_SingleHiggsEvent",
    "is_SingleZEvent",
    "is_ZHEvent",
    "Lumi_weight",
    "nBTaggedJets",
    "NJets",
]


def _compute_histogram_overlap(vals1, vals2, bins=50):
    combined = np.concatenate([vals1, vals2])
    bins_edges = np.linspace(np.percentile(combined, 0.5), np.percentile(combined, 99.5), bins + 1)
    h1, _ = np.histogram(vals1, bins=bins_edges)
    h2, _ = np.histogram(vals2, bins=bins_edges)
    p1 = h1 / (h1.sum() + 1e-10)
    p2 = h2 / (h2.sum() + 1e-10)
    return float(np.minimum(p1, p2).sum())


def run_top_overlap(top_k=3, bins=50, out_path=None):
    if out_path is None:
        out_path = _ROOT / "metrics" / "top_observables_overlap_separation.csv"

    sm_path = _ROOT / "datasets" / "new_Input_bbyy_SMEFT_SM_4thMarch_2026.h5"
    if not sm_path.is_file():
        raise SystemExit(f"Missing {sm_path}")

    sm_df = load_dataset(str(sm_path))
    observables = [c for c in sm_df.columns if c not in EXCLUDE_OVERLAP]

    rows = []
    for bsm in BSM_NAMES_OVERLAP:
        bsm_path = _ROOT / "datasets" / f"new_Input_bbyy_SMEFT_{bsm}_4thMarch_2026.h5"
        if not bsm_path.is_file():
            raise FileNotFoundError(bsm_path)
        bsm_df = load_dataset(str(bsm_path))

        scored = []
        for obs in observables:
            v_sm = sm_df[obs].dropna().values
            v_bsm = bsm_df[obs].dropna().values
            if len(v_sm) < 10 or len(v_bsm) < 10:
                continue
            ov = _compute_histogram_overlap(v_sm, v_bsm, bins=bins)
            sep = 1.0 - ov
            scored.append((obs, ov, sep))

        scored.sort(key=lambda x: x[2], reverse=True)
        for rank, (obs, ov, sep) in enumerate(scored[:top_k], start=1):
            rows.append(
                {
                    "BSM_Operator": bsm,
                    "rank": rank,
                    "observable": obs,
                    "histogram_overlap": ov,
                    "overlap_deviation_1_minus_overlap": sep,
                    "event_scope": "all_events",
                    "n_bins": bins,
                }
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")
    print(out_df.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="Export metrics CSVs (unified pipeline).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("comparison", help="DNN/GCN/GAT comparison → metrics/all_models_comparison.csv")
    sub.add_parser("nn-importance", help="DNN permutation importance → metrics/nn_feature_importance.csv")
    p_top = sub.add_parser("top-overlap", help="Top observables by SM–BSM separation → metrics/top_observables_overlap_separation.csv")
    p_top.add_argument("--top-k", type=int, default=3)
    p_top.add_argument("--bins", type=int, default=50)
    p_top.add_argument("--out", type=Path, default=None)
    sub.add_parser("all", help="Run comparison, nn-importance, then top-overlap")

    args = p.parse_args()
    if args.cmd == "comparison":
        run_model_comparison()
    elif args.cmd == "nn-importance":
        run_nn_importance()
    elif args.cmd == "top-overlap":
        run_top_overlap(top_k=args.top_k, bins=args.bins, out_path=args.out)
    elif args.cmd == "all":
        run_model_comparison()
        run_nn_importance()
        run_top_overlap()


if __name__ == "__main__":
    main()
