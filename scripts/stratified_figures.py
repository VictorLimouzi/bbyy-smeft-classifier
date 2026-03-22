#!/usr/bin/env python3
"""
Thesis §3.3.2 style figures: compare the most important NN inputs under
(1) random undersampling vs (2) stratified undersampling that matches SM/BSM
event-type fractions (ZH, HH, Single H, Other).

Default: all six DNN BSM datasets (cbbim, cbgim, cbhim, chbtil, chgtil, ctbim). Per operator:
  - figures/stratified_sampling_event_fractions_{bsm}.png
  - figures/stratified_sampling_top_features_{bsm}.png
Use --bsm <name> to generate only one.

Requires: datasets/*.h5 and metrics/nn_feature_importance.csv with ≥ --n-features rows per operator
(no fallbacks). If the CSV only has some operators, run:
  python reproduce.py metrics nn-importance
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import STEP_HIST_KW, apply_publication_style

ROOT = Path(__file__).resolve().parent.parent
# Same six operators as DNN / nn_classification multi-dataset setup
BSM_OPERATORS = ["cbbim", "cbgim", "cbhim", "chbtil", "chgtil", "ctbim"]
EVENT_ORDER = ["ZH", "HH", "Single H", "Other"]
EVENT_COLORS = ["#3357c2", "#e94f37", "#44af69", "#7c3aed"]


def load_dataset(fp: Path) -> pd.DataFrame:
    with h5py.File(fp, "r") as f:
        return pd.DataFrame.from_records(f["ForAnalysis/1d"][:])


def _get_event_type(row) -> str:
    if row.get("is_ZHEvent", False):
        return "ZH"
    if row.get("is_HHEvent", False):
        return "HH"
    if row.get("is_SingleHiggsEvent", False):
        return "Single H"
    return "Other"


def random_undersample_sm_bsm(sm_df: pd.DataFrame, bsm_df: pd.DataFrame, random_state: int = 42):
    n_min = min(len(sm_df), len(bsm_df))
    sm_s = sm_df.sample(n=n_min, random_state=random_state)
    bsm_s = bsm_df.sample(n=n_min, random_state=random_state)
    return sm_s, bsm_s


def stratified_undersample_sm_bsm(sm_df: pd.DataFrame, bsm_df: pd.DataFrame, random_state: int = 42):
    sm_copy = sm_df.copy()
    bsm_copy = bsm_df.copy()
    sm_copy["_event_type"] = sm_copy.apply(_get_event_type, axis=1)
    bsm_copy["_event_type"] = bsm_copy.apply(_get_event_type, axis=1)
    sm_samples, bsm_samples = [], []
    for etype in EVENT_ORDER:
        sm_sub = sm_copy[sm_copy["_event_type"] == etype]
        bsm_sub = bsm_copy[bsm_copy["_event_type"] == etype]
        n_min = min(len(sm_sub), len(bsm_sub))
        if n_min > 0:
            sm_samples.append(sm_sub.sample(n=n_min, random_state=random_state))
            bsm_samples.append(bsm_sub.sample(n=n_min, random_state=random_state))
    if not sm_samples or not bsm_samples:
        raise RuntimeError(
            "Stratified undersampling failed: no SM/BSM overlap in any event-type stratum (ZH/HH/Single H/Other)."
        )
    sm_out = pd.concat(sm_samples, ignore_index=True)
    bsm_out = pd.concat(bsm_samples, ignore_index=True)
    return sm_out.drop(columns=["_event_type"]), bsm_out.drop(columns=["_event_type"])


def event_fractions(df: pd.DataFrame) -> pd.Series:
    et = df.apply(_get_event_type, axis=1)
    counts = et.value_counts()
    fr = pd.Series({e: 0.0 for e in EVENT_ORDER})
    for e in EVENT_ORDER:
        if e in counts.index:
            fr[e] = counts[e] / len(df)
    return fr


def top_features_from_csv(bsm: str, n: int, csv_path: Path) -> list[str]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Required file missing: {csv_path}")
    fi = pd.read_csv(csv_path)
    for col in ("operator", "feature", "importance"):
        if col not in fi.columns:
            raise ValueError(f"{csv_path} must contain column {col!r}")
    op_rows = fi[fi["operator"] == bsm]
    if len(op_rows) < n:
        raise ValueError(
            f"{csv_path}: operator {bsm!r} has {len(op_rows)} row(s); need at least {n} for --n-features"
        )
    sub = op_rows.nlargest(n, "importance")
    if len(sub) < n:
        raise ValueError(f"{csv_path}: could not select {n} features for operator {bsm!r}")
    return sub["feature"].tolist()


def plot_event_fractions(
    sm_full: pd.DataFrame,
    bsm_full: pd.DataFrame,
    sm_rand: pd.DataFrame,
    bsm_rand: pd.DataFrame,
    sm_strat: pd.DataFrame,
    bsm_strat: pd.DataFrame,
    bsm_name: str,
    out_path: Path,
):
    fig, (ax_sm, ax_bsm) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(3)
    width = 0.18
    labels = ["Full\nsample", "Random\nbalance", "Stratified\nbalance"]

    def stack_bars(ax, fr_list, title, use_labels: bool):
        bottom = np.zeros(3)
        for e, c in zip(EVENT_ORDER, EVENT_COLORS):
            vals = np.array([fr[e] for fr in fr_list])
            lbl = e if use_labels else "_nolegend_"
            ax.bar(x, vals, width=0.65, bottom=bottom, label=lbl, color=c, edgecolor="white", linewidth=0.5)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Fraction of events", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.3)

    sm_fr = [
        event_fractions(sm_full),
        event_fractions(sm_rand),
        event_fractions(sm_strat),
    ]
    bsm_fr = [
        event_fractions(bsm_full),
        event_fractions(bsm_rand),
        event_fractions(bsm_strat),
    ]
    stack_bars(ax_sm, sm_fr, f"SM — event composition ({bsm_name} study)", use_labels=False)
    stack_bars(ax_bsm, bsm_fr, f"BSM ({bsm_name}) — event composition", use_labels=True)
    handles, labels_leg = ax_bsm.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08), fontsize=10)
    fig.suptitle(
        "Event-type fractions: before/after undersampling correction",
        fontsize=13,
        fontweight="bold",
        y=1.18,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_feature_grid(
    sm_rand: pd.DataFrame,
    bsm_rand: pd.DataFrame,
    sm_strat: pd.DataFrame,
    bsm_strat: pd.DataFrame,
    features: list[str],
    bsm_name: str,
    out_path: Path,
):
    n = len(features)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.8 * n))
    if n == 1:
        axes = np.atleast_2d(axes)
    col_titles = [
        "Random undersampling\n(no event-type matching)",
        "Stratified undersampling\n(matched SM/BSM event fractions)",
    ]

    for row, feat in enumerate(features):
        for col, (sm_s, bsm_s, title) in enumerate(
            zip(
                [sm_rand, sm_strat],
                [bsm_rand, bsm_strat],
                col_titles,
            )
        ):
            ax = axes[row, col]
            d_sm = sm_s[feat].dropna()
            d_bsm = bsm_s[feat].dropna()
            if len(d_sm) < 5 or len(d_bsm) < 5:
                raise ValueError(
                    f"Too few non-NaN values for feature {feat!r} ({bsm_name} plot): "
                    f"SM={len(d_sm)}, BSM={len(d_bsm)} (need ≥5 each)"
                )
            combined = np.concatenate([d_sm.values, d_bsm.values])
            lo, hi = np.percentile(combined, [1, 99])
            if lo >= hi:
                lo, hi = combined.min(), combined.max()
            bins = np.linspace(lo, hi, 45)
            ax.hist(d_sm, bins=bins, density=True, label="SM", color="C0", **STEP_HIST_KW)
            ax.hist(d_bsm, bins=bins, density=True, label=bsm_name, color="C3", **STEP_HIST_KW)
            ax.set_xlabel(feat, fontsize=10)
            ax.set_ylabel("Normalized density", fontsize=9)
            if row == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")
            if col == 1:
                ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.25)

    plt.suptitle(
        f"Top input features: SM vs {bsm_name} (importance-ranked)",
        fontsize=13,
        fontweight="bold",
        y=1.002,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def process_one_operator(
    bsm: str,
    sm_df: pd.DataFrame,
    n_features: int,
    importance_csv: Path,
    fig_dir: Path,
) -> None:
    """Load BSM HDF5, write both figures. Raises if dataset or features are invalid."""
    bsm_path = ROOT / "datasets" / f"new_Input_bbyy_SMEFT_{bsm}_4thMarch_2026.h5"
    if not bsm_path.is_file():
        raise FileNotFoundError(f"Missing BSM dataset: {bsm_path}")

    bsm_df = load_dataset(bsm_path)
    sm_rand, bsm_rand = random_undersample_sm_bsm(sm_df, bsm_df)
    sm_strat, bsm_strat = stratified_undersample_sm_bsm(sm_df, bsm_df)

    plot_event_fractions(
        sm_df,
        bsm_df,
        sm_rand,
        bsm_rand,
        sm_strat,
        bsm_strat,
        bsm,
        fig_dir / f"stratified_sampling_event_fractions_{bsm}.png",
    )

    features = top_features_from_csv(bsm, n_features, importance_csv)
    exclude = {"EventNumber", "is_HHEvent", "is_SingleHiggsEvent", "is_SingleZEvent", "is_ZHEvent"}
    bad = [f for f in features if f not in sm_df.columns or f not in bsm_df.columns or f in exclude]
    if bad:
        raise ValueError(
            f"Operator {bsm!r}: features from CSV missing in SM/BSM tables or are metadata flags: {bad}"
        )

    plot_feature_grid(
        sm_rand,
        bsm_rand,
        sm_strat,
        bsm_strat,
        features,
        bsm,
        fig_dir / f"stratified_sampling_top_features_{bsm}.png",
    )


def main():
    parser = argparse.ArgumentParser(description="Stratified vs random undersampling feature plots (§3.3.2).")
    parser.add_argument(
        "--bsm",
        default=None,
        metavar="NAME",
        help="Single BSM operator (e.g. chbtil). If omitted, process all six DNN datasets.",
    )
    parser.add_argument("--n-features", type=int, default=6, help="Number of top features to plot per operator")
    parser.add_argument(
        "--importance-csv",
        type=Path,
        default=ROOT / "metrics" / "nn_feature_importance.csv",
        help="Permutation importance CSV (operator, feature, importance)",
    )
    args = parser.parse_args()
    apply_publication_style()

    imp = args.importance_csv
    if not imp.is_file():
        raise SystemExit(f"Missing required importance CSV: {imp}")

    sm_path = ROOT / "datasets" / "new_Input_bbyy_SMEFT_SM_4thMarch_2026.h5"
    if not sm_path.is_file():
        raise SystemExit(f"Missing {sm_path}")

    operators = [args.bsm] if args.bsm else list(BSM_OPERATORS)
    if args.bsm and args.bsm not in BSM_OPERATORS:
        print(f"Note: {args.bsm!r} is not in the standard six; still requiring HDF5 and CSV rows.")

    sm_df = load_dataset(sm_path)
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for bsm in operators:
        print(f"Processing {bsm}...")
        process_one_operator(bsm, sm_df, args.n_features, args.importance_csv, fig_dir)

    print(f"Done: {len(operators)} operator(s) — figures under {fig_dir}/")


if __name__ == "__main__":
    main()
