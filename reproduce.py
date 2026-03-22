#!/usr/bin/env python3
"""
Single entry point for regenerating plots, figures, and metrics.

Run from the project root:

  python reproduce.py eda              # exploratory figures → figures/exploratory/
  python reproduce.py figures          # publication figures → figures/publication/ (+ stratified panels)
  python reproduce.py figures --no-stratified
  python reproduce.py metrics comparison
  python reproduce.py metrics nn-importance
  python reproduce.py metrics top-overlap
  python reproduce.py metrics all
  python reproduce.py architecture     # DNN/GCN/GAT diagrams (torchviz + graphviz)
  python reproduce.py graph          # GNN input graph schematic
  python reproduce.py clean-notebooks
  python reproduce.py all              # eda, figures, metrics all (slow)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(script: str, extra: list[str] | None = None) -> int:
    path = ROOT / "scripts" / script
    cmd = [sys.executable, str(path)]
    if extra:
        cmd.extend(extra)
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate project outputs (run from repo root).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("eda", help="EDA figures → figures/exploratory/")

    p_fig = sub.add_parser("figures", help="Publication figures → figures/publication/")
    p_fig.add_argument(
        "--no-stratified",
        action="store_true",
        help="Skip stratified vs random undersampling figures (stratified_figures.py).",
    )

    p_met = sub.add_parser("metrics", help="Export metrics CSVs (see subcommands).")
    p_met.add_argument(
        "metrics_cmd",
        nargs="?",
        default="all",
        choices=("comparison", "nn-importance", "top-overlap", "all"),
        help="metrics_pipeline subcommand (default: all).",
    )
    p_met.add_argument("--top-k", type=int, default=3, help="For top-overlap only.")
    p_met.add_argument("--bins", type=int, default=50, help="For top-overlap only.")

    sub.add_parser("architecture", help="Architecture diagrams → figures/publication/")
    sub.add_parser("graph", help="GNN graph schematic → figures/publication/")
    sub.add_parser("clean-notebooks", help="Strip notebook outputs for submission.")

    p_all = sub.add_parser("all", help="Run eda, figures, and metrics all (heavy).")
    p_all.add_argument(
        "--no-stratified",
        action="store_true",
        help="Pass through to figures step.",
    )

    args = p.parse_args()

    if args.cmd == "eda":
        raise SystemExit(_run("eda_plots.py"))
    if args.cmd == "figures":
        code = _run("publication_figures.py")
        if code != 0:
            raise SystemExit(code)
        if not args.no_stratified:
            raise SystemExit(_run("stratified_figures.py"))
        raise SystemExit(0)
    if args.cmd == "metrics":
        extra = [args.metrics_cmd]
        if args.metrics_cmd == "top-overlap":
            extra.extend(["--top-k", str(args.top_k), "--bins", str(args.bins)])
        raise SystemExit(_run("metrics_pipeline.py", extra))
    if args.cmd == "architecture":
        raise SystemExit(_run("architecture.py"))
    if args.cmd == "graph":
        raise SystemExit(_run("visualize_gnn_graph.py"))
    if args.cmd == "clean-notebooks":
        raise SystemExit(_run("clean_notebooks.py"))
    if args.cmd == "all":
        for step, script, xtra in [
            ("eda", "eda_plots.py", None),
            ("figures (core)", "publication_figures.py", None),
        ]:
            c = _run(script, xtra)
            if c != 0:
                print(f"Stopped after {step} (exit {c}).", file=sys.stderr)
                raise SystemExit(c)
        if not args.no_stratified:
            c = _run("stratified_figures.py")
            if c != 0:
                print(f"Stopped after stratified figures (exit {c}).", file=sys.stderr)
                raise SystemExit(c)
        c = _run("metrics_pipeline.py", ["all"])
        raise SystemExit(c)


if __name__ == "__main__":
    main()
