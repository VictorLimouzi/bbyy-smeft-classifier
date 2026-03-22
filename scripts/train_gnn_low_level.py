#!/usr/bin/env python3
"""
Train GCN / GAT on **low-level** inputs only (no high-level global observables).

Two representations (see README `ForAnalysis/1d` vs `ForAnalysis/2d`):

1. **1d-objects (default, works with March 2026 files)**  
   Graph: 5 nodes (Higgs, lead jet, sublead jet, two photons), 4 features each
   (pT, η, φ, M). Fully connected. **No** `m_bbyy`, angular CP observables, or
   other event-level summaries — only object kinematics.

2. **2d-objects** (`--use-2d`)  
   Graph: 16 nodes from `ForAnalysis/2d` (8 fields per object).  
   **Note:** In the current `new_Input_*_4thMarch_2026.h5` files, kinematic
   fields in `2d` are entirely NaN (branch not filled in the ntuple). The script
   will exit with an error unless you pass `--allow-empty-2d` (not recommended).

Usage:
  python scripts/train_gnn_low_level.py --bsm cbgim
  python scripts/train_gnn_low_level.py --quick --bsm cbbim
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, global_max_pool, global_mean_pool

# -----------------------------------------------------------------------------
# 1d: object-level graph (low-level)
# -----------------------------------------------------------------------------

N_NODES_1D = 5


def load_1d_df(filepath: str) -> pd.DataFrame:
    with h5py.File(filepath, "r") as f:
        return pd.DataFrame.from_records(f["ForAnalysis/1d"][:])


def event_to_graph_1d_objects(row, label: int) -> Data:
    """5 objects × (pT, η, φ, M); photon mass set to 0."""
    x = torch.tensor(
        [
            [row["Higgs_pT"], row["Higgs_Eta"], row["Higgs_Phi"], row["Higgs_Mass"]],
            [row["LeadJet_pT"], row["LeadJet_Eta"], row["LeadJet_Phi"], row["LeadJet_M"]],
            [
                row["SubLeadJet_pT"],
                row["SubLeadJet_Eta"],
                row["SubLeadJet_Phi"],
                row["SubLeadJet_M"],
            ],
            [row["LeadPhoton_pT"], row["LeadPhoton_Eta"], row["LeadPhoton_Phi"], 0.0],
            [
                row["SubLeadPhoton_pT"],
                row["SubLeadPhoton_Eta"],
                row["SubLeadPhoton_Phi"],
                0.0,
            ],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [[i, j] for i in range(N_NODES_1D) for j in range(N_NODES_1D) if i != j],
        dtype=torch.long,
    ).t().contiguous()
    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([float(label)], dtype=torch.float),
    )


# -----------------------------------------------------------------------------
# 2d: optional (16 × 8 structured)
# -----------------------------------------------------------------------------

OBJECT_FEATURE_FIELDS = [
    "object_pt",
    "object_eta",
    "object_phi",
    "object_mass",
    "object_isJet",
    "object_isLep",
    "object_isMET",
    "object_isbTagged",
]
N_NODES_2D = 16
N_FEAT_2D = len(OBJECT_FEATURE_FIELDS)


def load_2d_array(filepath: str) -> np.ndarray:
    with h5py.File(filepath, "r") as f:
        arr = f["ForAnalysis/2d"][:]
    if arr.ndim != 2 or arr.shape[1] != N_NODES_2D:
        raise ValueError(f"Expected 2d array (N, {N_NODES_2D}), got {arr.shape}")
    return arr


def finite_fraction_kinematics(arr: np.ndarray) -> float:
    pt = arr["object_pt"].astype(np.float64)
    return float(np.isfinite(pt).mean())


def event_row_2d_to_tensor(event_objects: np.ndarray) -> torch.Tensor:
    out = np.zeros((N_NODES_2D, N_FEAT_2D), dtype=np.float32)
    for j, name in enumerate(OBJECT_FEATURE_FIELDS):
        col = event_objects[name].astype(np.float32)
        out[:, j] = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(out)


def event_to_graph_2d(row_2d, label: int) -> Data:
    x = event_row_2d_to_tensor(row_2d)
    edge_index = torch.tensor(
        [[i, j] for i in range(N_NODES_2D) for j in range(N_NODES_2D) if i != j],
        dtype=torch.long,
    ).t().contiguous()
    return Data(x=x, edge_index=edge_index, y=torch.tensor([float(label)], dtype=torch.float))


def _get_event_type(row) -> str:
    if row.get("is_ZHEvent", False):
        return "ZH"
    if row.get("is_HHEvent", False):
        return "HH"
    if row.get("is_SingleHiggsEvent", False):
        return "Single H"
    return "Other"


def stratified_indices(sm_df: pd.DataFrame, bsm_df: pd.DataFrame, balance: bool):
    if not balance:
        n = min(len(sm_df), len(bsm_df))
        rng = np.random.default_rng(42)
        return rng.choice(len(sm_df), size=n, replace=False), rng.choice(
            len(bsm_df), size=n, replace=False
        )
    sm_copy = sm_df.copy()
    bsm_copy = bsm_df.copy()
    sm_copy["_event_type"] = sm_copy.apply(_get_event_type, axis=1)
    bsm_copy["_event_type"] = bsm_copy.apply(_get_event_type, axis=1)
    sm_parts, bsm_parts = [], []
    for etype in ["ZH", "HH", "Single H", "Other"]:
        sm_sub = sm_copy[sm_copy["_event_type"] == etype]
        bsm_sub = bsm_copy[bsm_copy["_event_type"] == etype]
        n_min = min(len(sm_sub), len(bsm_sub))
        if n_min > 0:
            sm_parts.append(sm_sub.sample(n=n_min, random_state=42))
            bsm_parts.append(bsm_sub.sample(n=n_min, random_state=42))
    if sm_parts and bsm_parts:
        sm_s = pd.concat(sm_parts, ignore_index=False)
        bsm_s = pd.concat(bsm_parts, ignore_index=False)
    else:
        n = min(len(sm_df), len(bsm_df))
        sm_s = sm_df.sample(n=n, random_state=42)
        bsm_s = bsm_df.sample(n=n, random_state=42)
    return sm_s.index.to_numpy(), bsm_s.index.to_numpy()


def mean_std_from_graphs(graphs: list[Data]) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.cat([g.x for g in graphs], dim=0)
    mean = xs.mean(dim=0)
    std = torch.clamp(xs.std(dim=0), min=1e-4)
    return mean, std


def standardize_graph_features(graphs: list[Data], mean: torch.Tensor, std: torch.Tensor) -> None:
    for g in graphs:
        g.x = (g.x - mean) / std


# -----------------------------------------------------------------------------
# Models: no global event vector
# -----------------------------------------------------------------------------


class GCN_Classifier_LL(nn.Module):
    def __init__(self, node_features: int, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GCNConv(node_features, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        combined_dim = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        return self.classifier(torch.cat([x_mean, x_max], dim=1)).squeeze(-1)


class GAT_Classifier_LL(nn.Module):
    def __init__(
        self,
        node_features: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GATConv(node_features, hidden_dim, heads=heads, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_dim * heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim * heads))
        self.convs.append(
            GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        )
        self.norms.append(nn.LayerNorm(hidden_dim))
        combined_dim = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        return self.classifier(torch.cat([x_mean, x_max], dim=1)).squeeze(-1)


def train_gnn(model, train_loader, val_loader, device, epochs=100, lr=0.001, patience=15):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    best_val = float("inf")
    best_state = None
    bad = 0
    for epoch in range(epochs):
        model.train()
        tr_loss = tr_ok = tr_n = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += loss.item() * batch.num_graphs
            probs = torch.sigmoid(out)
            tr_ok += ((probs > 0.5).float() == batch.y).sum().item()
            tr_n += batch.num_graphs
        tr_loss /= tr_n
        model.eval()
        va_loss = va_ok = va_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y.float())
                va_loss += loss.item() * batch.num_graphs
                probs = torch.sigmoid(out)
                va_ok += ((probs > 0.5).float() == batch.y).sum().item()
                va_n += batch.num_graphs
        va_loss /= va_n
        scheduler.step(va_loss)
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs}  train_loss={tr_loss:.4f}  "
                f"val_loss={va_loss:.4f}  val_acc={va_ok/va_n:.4f}"
            )
        if bad >= patience:
            print(f"  Early stop at epoch {epoch+1}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)


def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            p = torch.sigmoid(model(batch)).cpu().numpy()
            probs.extend(np.atleast_1d(p).tolist())
            labels.extend(batch.y.cpu().numpy().tolist())
    y = np.array(labels)
    p = np.array(probs)
    pred = (p > 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": auc(fpr, tpr),
    }


def build_graphs_1d(sm_df, bsm_df, sm_idx, bsm_idx) -> list[Data]:
    graphs = []
    for i in sm_idx:
        graphs.append(event_to_graph_1d_objects(sm_df.loc[i], 0))
    for i in bsm_idx:
        graphs.append(event_to_graph_1d_objects(bsm_df.loc[i], 1))
    return graphs


def build_graphs_2d(sm_2d, bsm_2d, sm_idx, bsm_idx) -> list[Data]:
    graphs = []
    for i in sm_idx:
        graphs.append(event_to_graph_2d(sm_2d[i], 0))
    for i in bsm_idx:
        graphs.append(event_to_graph_2d(bsm_2d[i], 1))
    return graphs


def main():
    parser = argparse.ArgumentParser(description="GNN: low-level inputs only (no global observables)")
    parser.add_argument(
        "--bsm",
        nargs="*",
        default=["cbbim", "cbgim", "cbhim", "chbtil", "chgtil", "ctbim"],
        help="BSM operator names",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--use-2d",
        action="store_true",
        help="Use ForAnalysis/2d (16 objects). Fails if kinematics are empty unless --allow-empty-2d",
    )
    parser.add_argument(
        "--allow-empty-2d",
        action="store_true",
        help="Allow training even when 2d kinematics are all NaN (not meaningful)",
    )
    args = parser.parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 25)
        args.patience = min(args.patience, 8)

    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    sm_path = "datasets/new_Input_bbyy_SMEFT_SM_4thMarch_2026.h5"
    sm_df = load_1d_df(sm_path)
    sm_2d = load_2d_array(sm_path) if args.use_2d else None
    if args.use_2d:
        frac = finite_fraction_kinematics(sm_2d)
        print(f"ForAnalysis/2d finite kinematics fraction (SM): {frac:.4f}")
        if frac < 0.01 and not args.allow_empty_2d:
            raise SystemExit(
                "2d kinematic fields are empty (NaN) in this file. Use default 1d object graphs "
                "(omit --use-2d), or regenerate ntuples with 2d filled. "
                "To force run anyway: --allow-empty-2d"
            )

    node_features = N_FEAT_2D if args.use_2d else 4
    repr_name = "2d_16x8" if args.use_2d else "1d_5x4_no_global"

    rows = []
    for bsm_name in args.bsm:
        bsm_path = f"datasets/new_Input_bbyy_SMEFT_{bsm_name}_4thMarch_2026.h5"
        if not os.path.isfile(bsm_path):
            print(f"Skip {bsm_name}: missing {bsm_path}")
            continue
        print(f"\n{'='*60}\n SM vs {bsm_name}  [{repr_name}]\n{'='*60}")
        bsm_df = load_1d_df(bsm_path)
        bsm_2d = load_2d_array(bsm_path) if args.use_2d else None

        sm_idx, bsm_idx = stratified_indices(sm_df, bsm_df, balance=not args.no_balance)
        if args.use_2d:
            graphs = build_graphs_2d(sm_2d, bsm_2d, sm_idx, bsm_idx)
        else:
            graphs = build_graphs_1d(sm_df, bsm_df, sm_idx, bsm_idx)

        np.random.default_rng(42).shuffle(graphs)
        n = len(graphs)
        n_train = int(0.7 * n)
        n_val = int(0.1 * n)
        train_ds = graphs[:n_train]
        val_ds = graphs[n_train : n_train + n_val]
        test_ds = graphs[n_train + n_val :]
        x_mean, x_std = mean_std_from_graphs(train_ds)
        standardize_graph_features(train_ds, x_mean, x_std)
        standardize_graph_features(val_ds, x_mean, x_std)
        standardize_graph_features(test_ds, x_mean, x_std)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        print(f"Graphs: {n} | nodes per graph: {N_NODES_2D if args.use_2d else N_NODES_1D} | feats: {node_features}")

        suffix = "_ll" if not args.use_2d else "_ll2d"
        t0 = time.time()
        gcn = GCN_Classifier_LL(node_features=node_features).to(device)
        print("Training GCN...")
        train_gnn(gcn, train_loader, val_loader, device, epochs=args.epochs, patience=args.patience)
        gcn_eval = evaluate(gcn, test_loader, device)
        gcn_time = time.time() - t0
        print(f"  GCN test AUC={gcn_eval['auc']:.4f} acc={gcn_eval['accuracy']:.4f} ({gcn_time:.1f}s)")

        t0 = time.time()
        gat = GAT_Classifier_LL(node_features=node_features).to(device)
        print("Training GAT...")
        train_gnn(gat, train_loader, val_loader, device, epochs=args.epochs, patience=args.patience)
        gat_eval = evaluate(gat, test_loader, device)
        gat_time = time.time() - t0
        print(f"  GAT test AUC={gat_eval['auc']:.4f} acc={gat_eval['accuracy']:.4f} ({gat_time:.1f}s)")

        torch.save(
            {
                "model_state_dict": gcn.state_dict(),
                "model_type": "GCN_LL",
                "representation": repr_name,
                "bsm_operator": bsm_name,
                "node_features": node_features,
                "auc": float(gcn_eval["auc"]),
            },
            f"models/gcn{suffix}_{bsm_name}_classifier.pt",
        )
        torch.save(
            {
                "model_state_dict": gat.state_dict(),
                "model_type": "GAT_LL",
                "representation": repr_name,
                "bsm_operator": bsm_name,
                "node_features": node_features,
                "auc": float(gat_eval["auc"]),
            },
            f"models/gat{suffix}_{bsm_name}_classifier.pt",
        )

        rows.append(
            {
                "BSM_Operator": bsm_name,
                "Representation": repr_name,
                "GCN_AUC": gcn_eval["auc"],
                "GAT_AUC": gat_eval["auc"],
                "GCN_Acc": gcn_eval["accuracy"],
                "GAT_Acc": gat_eval["accuracy"],
            }
        )

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv("metrics/gnn_low_level_comparison.csv", index=False)
        print("\nSaved metrics/gnn_low_level_comparison.csv")
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
