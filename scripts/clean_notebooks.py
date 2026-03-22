#!/usr/bin/env python3
"""
Clean notebooks for thesis submission:
- Clear all outputs
- Reset execution counts to null
- Remove pip install cell from GNN (replace with markdown)
"""

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

NOTEBOOKS = [
    "01_eda_hh_bbyy.ipynb",
    "02_event_type_composition.ipynb",
    "03_neural_network_classification.ipynb",
    "04_graph_neural_network_classification.ipynb",
    "05_model_comparison.ipynb",
    "06_gnn_low_level_extended.ipynb",
]


def clean_notebook(path: str) -> None:
    """Clear outputs and reset execution counts."""
    with open(path, "r") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        cell["outputs"] = []
        if "execution_count" in cell:
            cell["execution_count"] = None

    with open(path, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"Cleaned: {path}")


def clean_gnn_notebook(path: str) -> None:
    """Clean GNN notebook: remove pip install cell, replace with markdown."""
    with open(path, "r") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    new_cells = []

    for i, cell in enumerate(cells):
        # Skip the pip install cell
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            if "!pip install torch-geometric" in source or "pip install torch-geometric" in source:
                # Replace with markdown note
                new_cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "**Note**: PyTorch Geometric is required. Install with `pip install torch-geometric` (see `requirements.txt`)."
                    ],
                })
                continue

        # Clear outputs and execution count for other cells
        cell["outputs"] = []
        if "execution_count" in cell:
            cell["execution_count"] = None
        new_cells.append(cell)

    nb["cells"] = new_cells

    with open(path, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"Cleaned (incl. pip cell removal): {path}")


def main():
    os.chdir(_ROOT)
    for name in NOTEBOOKS:
        path = name
        if name == "04_graph_neural_network_classification.ipynb":
            clean_gnn_notebook(path)
        else:
            clean_notebook(path)
    print("Done.")


if __name__ == "__main__":
    main()
