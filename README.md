# Machine Learning Classification of CP-Odd SMEFT Operators in $b\bar{b}\gamma\gamma$ Production at the HL-LHC

Repository for my **Senior Honours** project.

**Victor Limouzi** · March 2026 · Advisor: Dominik Duda

---

## Abstract

Simulated HL-LHC–style events in the $b\bar{b}\gamma\gamma$ final state are used to separate Standard Model (SM) physics from six benchmark **CP-odd SMEFT** scenarios (one dedicated dataset per operator). The pipeline combines exploratory histograms and correlations with three classifiers: a feedforward neural network, a graph convolutional network (GCN), and a graph attention network (GAT). Physics interpretation and operator definitions are left to the thesis; this repo focuses on **code, data layout, and reproduction**.

---

## Physics (short)

The study is cast in **SMEFT** with CP-violating dimension-6 effects turned on **one operator at a time**. Each BSM sample is labelled by a short tag used in filenames and models:

**cbbim, cbgim, cbhim, chbtil, chgtil, ctbim** (six operators) plus **SM** as the reference.

For Wilson coefficients, Lagrangian terms, and motivation, see the written thesis.

---

## Reproducibility

All results can be reproduced using the instructions below. Fixed random seeds (42) are used throughout for train/test splits and model initialisation. **Python 3.9+** is recommended.

### Environment
```bash
pip install -r requirements.txt
# Architecture diagrams (optional): system Graphviz + Python torchviz
#   macOS: brew install graphviz
#   Linux: apt install graphviz
```

### Command-line driver (`reproduce.py`)

From the repository root, `python3 reproduce.py --help` lists all subcommands. Typical workflow:

| Command | What it does |
|---------|----------------|
| `eda` | Exploratory plots → `plots/` |
| `figures` | Publication-style figures → `figures/` (after `figures`, also runs stratified-sampling figures unless you pass `--no-stratified`) |
| `metrics comparison` | Evaluates saved checkpoints; writes `metrics/all_models_comparison.csv` |
| `metrics nn-importance` | Permutation importance → `metrics/nn_feature_importance.csv` |
| `metrics top-overlap` | Top observables by histogram overlap → `metrics/top_observables_overlap_separation.csv` (optional: `--top-k`, `--bins`) |
| `metrics all` | Runs the three metrics steps above in order |
| `architecture` | DNN/GCN/GAT torchviz diagrams → `figures/` |
| `graph` | GNN input graph schematic → `figures/` |
| `clean-notebooks` | Clears outputs and execution counts in all numbered notebooks (optional before submission) |
| `all` | `eda` + `figures` + `metrics all` (slow; needs `models/` and datasets) |

Examples:

```bash
python3 reproduce.py eda
python3 reproduce.py figures
python3 reproduce.py figures --no-stratified   # skip stratified_sampling_* figures
python3 reproduce.py metrics all
python3 reproduce.py metrics top-overlap --top-k 5
python3 reproduce.py clean-notebooks
python3 reproduce.py all                         # full regeneration pipeline
```

Scripts live in `scripts/` and can be run directly; each script changes to the project root first, e.g. `python3 scripts/eda_plots.py`.

### Run Notebooks (in order)
| Order | Notebook | Output |
|-------|----------|--------|
| 1 | `01_eda_hh_bbyy.ipynb` | EDA plots → `plots/` |
| 2 | `02_event_type_composition.ipynb` | Event type plots → `plots/` |
| 3 | `03_neural_network_classification.ipynb` | NN models → `models/`, metrics → `metrics/` |
| 4 | `04_graph_neural_network_classification.ipynb` | GCN/GAT models → `models/`, metrics → `metrics/` |
| 5 | `05_model_comparison.ipynb` | Comparison figures → `figures/` |
| 6 | `06_gnn_low_level_extended.ipynb` | Low-level / extended GNN (optional) |

**Dependencies**: **PyTorch Geometric** is required for notebooks **4–6** (and for `reproduce.py metrics comparison` to evaluate GCN/GAT). Notebook **3** (feedforward NN) uses PyTorch only. Checkpoints under `models/` let you run `reproduce.py figures` and `metrics comparison` without retraining the notebooks.

---

## Datasets

Located in `datasets/`:

| File | Label | Events |
|------|--------|--------|
| `new_Input_bbyy_SMEFT_SM_4thMarch_2026.h5` | SM | 24,435 |
| `new_Input_bbyy_SMEFT_cbbim_4thMarch_2026.h5` | cbbim | 16,288 |
| `new_Input_bbyy_SMEFT_cbgim_4thMarch_2026.h5` | cbgim | 69,460 |
| `new_Input_bbyy_SMEFT_cbhim_4thMarch_2026.h5` | cbhim | 51,702 |
| `new_Input_bbyy_SMEFT_chbtil_4thMarch_2026.h5` | chbtil | 110,166 |
| `new_Input_bbyy_SMEFT_chgtil_4thMarch_2026.h5` | chgtil | 67,434 |
| `new_Input_bbyy_SMEFT_ctbim_4thMarch_2026.h5` | ctbim | 17,893 |

### HDF5 layout
- **`ForAnalysis/1d`**: per-event observables used for the NN and for building graphs (Higgs, jets, photons, angular variables, etc.). This is the main branch for this project.
- **`ForAnalysis/2d`**: object-wise table in the file; **kinematics are empty (NaN)** in the March 2026 inputs used here, so stick to `1d` unless you regenerate ntuples.

Event-type and weight columns include `is_HHEvent`, `is_SingleHiggsEvent`, `is_SingleZEvent`, `is_ZHEvent`, and `Lumi_weight`.

---

## Work Completed

### 1. Exploratory Data Analysis (EDA)
**Notebook**: `01_eda_hh_bbyy.ipynb` | **Script**: `reproduce.py eda` (or `scripts/eda_plots.py`)

Area-normalised histograms, correlation matrices, ratio plots (BSM/SM) with ±10% bands, distribution overlap heatmap, CDFs, angular distributions, invariant mass spectra, pT spectra, profile plots. Outputs → `plots/`.

### 2. Event Type Contribution Analysis
**Notebook**: `02_event_type_composition.ipynb`

Lumi-weighted contribution of ZH, HH, Single H, Other per dataset. Outputs → `plots/event_type_contribution_*.png`.

### 3. Neural Network Classification
**Notebook**: `03_neural_network_classification.ipynb`

4-layer MLP (31→128→64→32→1), StandardScaler, random undersampling, stratified 70/10/20 split. The notebook walks through training for a **representative BSM operator (chbtil)**; the same architecture is used for all six operators, with checkpoints named `models/sm_vs_<operator>_classifier.pt`. Outputs → `models/`, `metrics/`, `plots/`, and (via `reproduce.py figures`) `figures/`.

### 4. Graph Neural Network Classification
**Notebook**: `04_graph_neural_network_classification.ipynb`

GCN and GAT on 5-node fully connected graphs. Event-type stratified undersampling. Outputs → `models/gcn_*_classifier.pt`, `models/gat_*_classifier.pt`, `metrics/`, `plots/`.

**Low-level GNN (no global observables)** — same idea as the full GNN but **no** 10 global event variables. Use either:
- **Notebook**: `06_gnn_low_level_extended.ipynb` — **8-node** graph (5 objects + dijet + γγ summary + angular node), **DeepResidualGCN** + **GraphTransformer**; saves `models/gcn_deep_ll_*`, `models/gt_ll_*`, `metrics/gcn_deep_ll_*` / `gt_ll_*` CSVs, `plots/gnn_ll_*.png`. The CLI script `scripts/train_gnn_low_level.py` is still the lighter **5-node** baseline.
- **Script**: `scripts/train_gnn_low_level.py` for CLI runs.

Checkpoints: `models/gcn_ll_*_classifier.pt`, `models/gat_ll_*_classifier.pt`, summary `metrics/gnn_low_level_comparison.csv`.

```bash
python3 scripts/train_gnn_low_level.py              # all BSM operators
python3 scripts/train_gnn_low_level.py --bsm cbgim  # one operator
python3 scripts/train_gnn_low_level.py --quick      # shorter run for testing
```

Optional `--use-2d` targets `ForAnalysis/2d` (16×8 nodes); the script exits with an error if `2d` kinematics are empty unless you pass `--allow-empty-2d` (not recommended).

### 5. Model Comparison
**Notebook**: `05_model_comparison.ipynb`

Comparison of NN, GCN, GAT across all 6 BSM operators. Outputs → `metrics/`, `figures/`.

---

## File Structure

```
seniour_honours_project/
├── datasets/           # HDF5 files (7 datasets)
├── plots/              # EDA and analysis plots
├── figures/            # Publication figures from scripts
├── models/             # Trained classifiers
├── metrics/            # Feature importance, comparison tables
├── scripts/            # Plotting, metrics, training utilities (see reproduce.py)
│   ├── eda_plots.py
│   ├── publication_figures.py
│   ├── stratified_figures.py
│   ├── metrics_pipeline.py
│   ├── architecture.py
│   ├── visualize_gnn_graph.py
│   ├── train_gnn_low_level.py
│   └── clean_notebooks.py
├── reproduce.py        # CLI: eda | figures | metrics | architecture | graph | clean-notebooks | all
├── plot_style.py       # Shared matplotlib style (notebooks + scripts)
├── 01_eda_hh_bbyy.ipynb
├── 02_event_type_composition.ipynb
├── 03_neural_network_classification.ipynb
├── 04_graph_neural_network_classification.ipynb
├── 05_model_comparison.ipynb
├── 06_gnn_low_level_extended.ipynb
├── requirements.txt
├── SUBMISSION_CHECKLIST.md
└── README.md
```

---

## Figure Reference (for thesis)

| Figure | Path |
|--------|------|
| Distribution overlap heatmap | `plots/distribution_overlap_heatmap.png` |
| Top discriminating observables (distributions) | `plots/top_features_comparison.png` |
| BSM/SM ratio panel (key observables) | `plots/ratio_panel_key_obs.png` |
| Event type contribution | `plots/event_type_contribution_stacked.png`, `plots/event_type_contribution_grouped.png` |
| Stratified vs random undersampling | `figures/stratified_sampling_event_fractions_*.png`, `figures/stratified_sampling_top_features_*.png` |
| ROC curves (combined) | `figures/roc_all_datasets_combined.png` |
| ROC curves (per operator, DNN+GCN+GAT) | `figures/roc_dnn_gcn_gat_*.png` |
| NN ROC + confusion (per operator) | `figures/nn_roc_confusion_*.png`, `figures/nn_confusion_*.png` |
| NN score distributions | `figures/nn_score_*.png` |
| NN feature importance | `plots/nn_feature_importance.png`, `figures/nn_feature_importance_top10_*.png` |
| GNN feature importance | `plots/gnn_feature_importance_gcn.png`, `plots/gnn_feature_importance_gat.png` |
| Learned distributions (notebooks) | `plots/nn_learned_distribution.png`, `plots/gnn_learned_distribution_*.png` |
| Model comparison (bars) | `figures/model_comparison.png`, `figures/auc_dnn_gcn_gat.png` |
| Notebook-only comparison plots | `plots/model_comparison_roc.png`, `plots/model_comparison_auc.png`, `plots/model_comparison_time.png` |

---

## Data Summary (Lumi-weighted)

| Dataset | Events | HH% | Single H% | Other% |
|---------|--------|-----|-----------|--------|
| SM | 24,435 | 0.2 | 4.2 | 95.6 |
| cbbim | 16,288 | 0.1 | 2.0 | 97.9 |
| cbgim | 69,460 | 0.0 | 82.9 | 17.0 |
| cbhim | 51,702 | 11.1 | 60.5 | 28.5 |
| chbtil | 110,166 | 5.4 | 92.5 | 2.2 |
| chgtil | 67,434 | 44.5 | 34.2 | 21.3 |
| ctbim | 17,893 | 0.2 | 4.4 | 95.4 |

*ZH = 0% for all datasets.*

---

## Acknowledgements

Senior Honours project. Thanks to my advisor Dominik for guidance and support.
