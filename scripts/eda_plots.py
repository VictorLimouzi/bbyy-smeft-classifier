#!/usr/bin/env python3
"""
Exploratory data analysis plots → plots/

From repo root: ``python reproduce.py eda`` (or run this file directly; it switches to the project root).
"""

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_ROOT)

import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from plot_style import STEP_HIST_KW, apply_publication_style

apply_publication_style()

os.makedirs('plots', exist_ok=True)

print("Loading datasets...")

with h5py.File('datasets/new_Input_bbyy_SMEFT_SM_4thMarch_2026.h5', 'r') as f:
    sm_df = pd.DataFrame.from_records(f['ForAnalysis/1d'][:])

with h5py.File('datasets/new_Input_bbyy_SMEFT_cbbim_4thMarch_2026.h5', 'r') as f:
    cbbim_df = pd.DataFrame.from_records(f['ForAnalysis/1d'][:])

with h5py.File('datasets/new_Input_bbyy_SMEFT_cbgim_4thMarch_2026.h5', 'r') as f:
    cbgim_df = pd.DataFrame.from_records(f['ForAnalysis/1d'][:])

with h5py.File('datasets/new_Input_bbyy_SMEFT_cbhim_4thMarch_2026.h5', 'r') as f:
    cbhim_df = pd.DataFrame.from_records(f['ForAnalysis/1d'][:])

with h5py.File('datasets/new_Input_bbyy_SMEFT_chbtil_4thMarch_2026.h5', 'r') as f:
    chbtil_df = pd.DataFrame.from_records(f['ForAnalysis/1d'][:])

with h5py.File('datasets/new_Input_bbyy_SMEFT_chgtil_4thMarch_2026.h5', 'r') as f:
    chgtil_df = pd.DataFrame.from_records(f['ForAnalysis/1d'][:])

with h5py.File('datasets/new_Input_bbyy_SMEFT_ctbim_4thMarch_2026.h5', 'r') as f:
    ctbim_df = pd.DataFrame.from_records(f['ForAnalysis/1d'][:])

print(f"Loaded: SM ({len(sm_df):,}), cbbim ({len(cbbim_df):,}), cbgim ({len(cbgim_df):,}), "
      f"cbhim ({len(cbhim_df):,}), chbtil ({len(chbtil_df):,}), chgtil ({len(chgtil_df):,}), "
      f"ctbim ({len(ctbim_df):,}) events")

for df in [sm_df, cbbim_df, cbgim_df, cbhim_df, chbtil_df, chgtil_df, ctbim_df]:
    df['is_HiggsEvent'] = df['is_HHEvent'] | df['is_SingleHiggsEvent'] | df['is_ZHEvent']

COLORS = {
    'SM': 'blue',
    'cbbim': 'orange',
    'cbgim': 'red',
    'cbhim': 'green',
    'chbtil': 'purple',
    'chgtil': 'brown',
    'ctbim': 'magenta',
}

BSM_NAMES = ['cbbim', 'cbgim', 'cbhim', 'chbtil', 'chgtil', 'ctbim']
ALL_NAMES = ['SM'] + BSM_NAMES

all_dfs = {
    'SM': sm_df, 'cbbim': cbbim_df, 'cbgim': cbgim_df, 'cbhim': cbhim_df,
    'chbtil': chbtil_df, 'chgtil': chgtil_df, 'ctbim': ctbim_df,
}

# All 32 physics observables (exclude metadata, event-type flags, and boolean is_HiggsEvent)
exclude_columns = ['EventNumber', 'is_HHEvent', 'is_SingleHiggsEvent', 'is_SingleZEvent', 'is_ZHEvent', 'is_HiggsEvent', 'Lumi_weight', 'nBTaggedJets', 'NJets']
observables = [c for c in sm_df.columns if c not in exclude_columns]

key_observables = ['Higgs_pT', 'm_bbyy', 'pT_jj', 'Eta_jj', 'signed_DeltaPhi_jj', 'cosThetaStar']

all_events = {name: df for name, df in all_dfs.items()}
higgs_events = {name: df[df['is_HiggsEvent'] == True].copy() for name, df in all_dfs.items()}
non_higgs_events = {name: df[df['is_HiggsEvent'] == False].copy() for name, df in all_dfs.items()}


def plot_with_ratio_panel(datasets_dict, observables, title_suffix='', save_name='ratio_panel'):
    """Create plots with main distribution on top and ratio to SM on bottom."""
    n_obs = len(observables)
    n_cols = 3
    n_rows = (n_obs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(6 * n_cols, 4 * n_rows * 2),
                             gridspec_kw={'height_ratios': [3, 1] * n_rows})

    for idx, obs in enumerate(observables):
        row = (idx // n_cols) * 2
        col = idx % n_cols

        ax_main = axes[row, col]
        ax_ratio = axes[row + 1, col]

        all_data_list = []
        skip = False
        for name in ALL_NAMES:
            d = datasets_dict[name][obs].dropna()
            if len(d) < 10:
                skip = True
                break
            all_data_list.append(d)
        if skip:
            ax_main.set_visible(False)
            ax_ratio.set_visible(False)
            continue

        combined = np.concatenate(all_data_list)
        bins = np.linspace(np.percentile(combined, 1), np.percentile(combined, 99), 30)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        hists = {}
        for name in ALL_NAMES:
            h, _ = np.histogram(datasets_dict[name][obs].dropna(), bins=bins, density=True)
            hists[name] = h

        for name in ALL_NAMES:
            ax_main.step(bin_centers, hists[name], where='mid', label=name,
                         color=COLORS[name], linewidth=2)

        ax_main.set_ylabel('Normalized', fontsize=11)
        ax_main.set_title(f'{obs} {title_suffix}', fontsize=12)
        ax_main.legend(fontsize=7, loc='upper right', ncol=2)
        ax_main.tick_params(labelbottom=False)
        ax_main.grid(True, alpha=0.3)

        sm_hist_safe = np.where(hists['SM'] > 0, hists['SM'], 1e-10)
        mask = hists['SM'] > 0
        for bsm in BSM_NAMES:
            ratio = hists[bsm] / sm_hist_safe
            ax_ratio.step(bin_centers[mask], ratio[mask], where='mid',
                          color=COLORS[bsm], linewidth=1.5)
        ax_ratio.axhline(y=1, color='blue', linestyle='-', linewidth=1.5)
        ax_ratio.fill_between(bin_centers, 0.9, 1.1, alpha=0.2, color='gray')

        ax_ratio.set_xlabel(obs, fontsize=11)
        ax_ratio.set_ylabel('Ratio/SM', fontsize=10)
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.grid(True, alpha=0.3)

    total_plots = n_rows * n_cols
    for idx in range(len(observables), total_plots):
        row = (idx // n_cols) * 2
        col = idx % n_cols
        if row < len(axes) and col < len(axes[0]):
            axes[row, col].set_visible(False)
            axes[row + 1, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'plots/{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Plot 1: Dataset comparison (all observables)
# =============================================================================
print("Generating: Dataset comparison plots...")
n_obs = len(observables)
n_cols_cmp = 4
n_rows_cmp = (n_obs + n_cols_cmp - 1) // n_cols_cmp
fig, axes = plt.subplots(n_rows_cmp, n_cols_cmp, figsize=(20, 5 * n_rows_cmp))
axes = axes.flatten()

for idx, obs in enumerate(observables):
    ax = axes[idx]
    data_arrays = [df[obs].dropna() for df in all_dfs.values()]
    combined = np.concatenate(data_arrays)
    bins = np.linspace(np.percentile(combined, 1), np.percentile(combined, 99), 50)

    for name in ALL_NAMES:
        ax.hist(
            all_dfs[name][obs].dropna(),
            bins=bins,
            density=True,
            label=name,
            color=COLORS[name],
            **STEP_HIST_KW,
        )

    ax.set_xlabel(obs)
    ax.set_ylabel('Normalized Density')
    ax.legend(fontsize=6, ncol=2)
    ax.set_title(f'{obs} Distribution')

for idx in range(len(observables), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('plots/dataset_comparison_normalized.png', dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# Plot 2: Correlation matrices
# =============================================================================
print("Generating: Correlation matrices...")
datasets_corr = {name: df[observables] for name, df in all_dfs.items()}

n_datasets = len(ALL_NAMES)
fig, axes = plt.subplots(2, 4, figsize=(32, 16))
axes = axes.flatten()
for i, (name, df) in enumerate(datasets_corr.items()):
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, ax=axes[i], cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                annot=False, square=True, cbar_kws={'shrink': 0.8})
    axes[i].set_title(f'Correlation - {name}', fontsize=14)
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right', fontsize=7)
    axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0, fontsize=7)

axes[-1].set_visible(False)
plt.tight_layout()
plt.savefig('plots/correlation_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

for name, df in datasets_corr.items():
    fig, ax = plt.subplots(figsize=(14, 12))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                annot=True, fmt='.2f', annot_kws={'size': 7}, square=True,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
    ax.set_title(f'Correlation Matrix - {name}', fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(f'plots/correlation_matrix_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Plot 3: signed_DeltaPhi_jj vs Eta_jj (2D)
# =============================================================================
print("Generating: 2D histograms (signed_DeltaPhi_jj vs Eta_jj)...")
n_2d = len(ALL_NAMES)
n_cols_2d = 4
n_rows_2d = (n_2d + n_cols_2d - 1) // n_cols_2d
fig, axes = plt.subplots(n_rows_2d, n_cols_2d, figsize=(6 * n_cols_2d, 5 * n_rows_2d))
axes = axes.flatten()

all_eta = np.concatenate([df['Eta_jj'] for df in all_dfs.values()])
all_dphi = np.concatenate([df['signed_DeltaPhi_jj'] for df in all_dfs.values()])
eta_bins = np.linspace(np.percentile(all_eta, 1), np.percentile(all_eta, 99), 50)
dphi_bins = np.linspace(np.percentile(all_dphi, 1), np.percentile(all_dphi, 99), 50)

for i, name in enumerate(ALL_NAMES):
    df = all_dfs[name]
    h, xedges, yedges = np.histogram2d(df['Eta_jj'], df['signed_DeltaPhi_jj'],
                                        bins=[eta_bins, dphi_bins])
    h_normalized = h / h.sum()
    im = axes[i].pcolormesh(xedges, yedges, h_normalized.T, cmap='viridis', shading='auto')
    axes[i].set_xlabel(r'$\eta_{jj}$', fontsize=12)
    axes[i].set_ylabel(r'signed $\Delta\phi_{jj}$', fontsize=12)
    axes[i].set_title(f'{name}', fontsize=14)
    plt.colorbar(im, ax=axes[i], label='Normalized Density')

for idx in range(len(ALL_NAMES), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('plots/signed_DeltaPhi_jj_vs_Eta_jj.png', dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# Plot 4: Ratio plots (BSM/SM)
# =============================================================================
print("Generating: Ratio plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes_flat = axes.flatten()

for idx, obs in enumerate(key_observables):
    ax = axes_flat[idx]

    data_arrays = [all_dfs[name][obs].dropna() for name in ALL_NAMES]
    combined = np.concatenate(data_arrays)
    bins = np.linspace(np.percentile(combined, 1), np.percentile(combined, 99), 30)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    sm_hist, _ = np.histogram(sm_df[obs].dropna(), bins=bins, density=True)
    sm_hist_safe = np.where(sm_hist > 0, sm_hist, 1e-10)

    for bsm in BSM_NAMES:
        bsm_hist, _ = np.histogram(all_dfs[bsm][obs].dropna(), bins=bins, density=True)
        ratio = bsm_hist / sm_hist_safe
        ax.step(bin_centers, ratio, where='mid', label=f'{bsm}/SM',
                color=COLORS[bsm], linewidth=2)

    ax.axhline(y=1, color='blue', linestyle='--', linewidth=1.5, label='SM (reference)')
    ax.fill_between(bin_centers, 0.9, 1.1, alpha=0.2, color='blue', label='±10% band')

    ax.set_xlabel(obs, fontsize=12)
    ax.set_ylabel('Ratio to SM', fontsize=12)
    ax.set_title(f'{obs} Ratio', fontsize=14)
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/ratio_plots_bsm_sm.png', dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# Plot 5: Signal vs Background
# =============================================================================
print("Generating: Signal vs Background comparison...")
fig, axes = plt.subplots(n_datasets, 6, figsize=(24, 4 * n_datasets))

for row_idx, name in enumerate(ALL_NAMES):
    df = all_dfs[name]
    signal_mask = df['is_HiggsEvent']
    for col_idx, obs in enumerate(key_observables):
        ax = axes[row_idx, col_idx]
        sig_data = df.loc[signal_mask, obs].dropna()
        bkg_data = df.loc[~signal_mask, obs].dropna()

        combined = np.concatenate([sig_data, bkg_data])
        bins = np.linspace(np.percentile(combined, 1), np.percentile(combined, 99), 40)

        ax.hist(sig_data, bins=bins, density=True, label='Signal', color='blue', **STEP_HIST_KW)
        ax.hist(bkg_data, bins=bins, density=True, label='Background', color='red', **STEP_HIST_KW)

        ax.set_xlabel(obs, fontsize=10)
        if col_idx == 0:
            ax.set_ylabel(f'{name}\nNormalized', fontsize=10)
        if row_idx == 0:
            ax.set_title(obs, fontsize=11)
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('plots/signal_vs_background.png', dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# Plot 6: Weighted distributions
# =============================================================================
print("Generating: Weighted distributions...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes_flat = axes.flatten()

for idx, obs in enumerate(key_observables):
    ax = axes_flat[idx]

    data_arrays = [all_dfs[name][obs].dropna() for name in ALL_NAMES]
    combined = np.concatenate(data_arrays)
    bins = np.linspace(np.percentile(combined, 1), np.percentile(combined, 99), 40)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for name in ALL_NAMES:
        df = all_dfs[name]
        data = df[obs].values
        weights = df['Lumi_weight'].values
        hist, _ = np.histogram(data, bins=bins, weights=weights)
        hist = hist / hist.sum()
        ax.step(bin_centers, hist, where='mid', label=name, color=COLORS[name], linewidth=2)

    ax.set_xlabel(obs, fontsize=12)
    ax.set_ylabel('Weighted Normalized', fontsize=12)
    ax.set_title(f'{obs} (Lumi Weighted)', fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/weighted_distributions.png', dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# Plot 7: CDFs
# =============================================================================
print("Generating: Cumulative distribution functions...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes_flat = axes.flatten()

for idx, obs in enumerate(key_observables):
    ax = axes_flat[idx]

    for name in ALL_NAMES:
        sorted_data = np.sort(all_dfs[name][obs].dropna())
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf, label=name, color=COLORS[name], linewidth=2)

    ax.set_xlabel(obs, fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title(f'{obs} CDF', fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('plots/cdf_comparison.png', dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# Plot 8: Angular distributions
# =============================================================================
print("Generating: Angular distributions...")
angular_obs = ['cosThetaStar', 'costheta1', 'costheta2', 'DPhi_bb',
               'DPhi_yybb', 'signed_DeltaPhi_jj']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()

for idx, obs in enumerate(angular_obs):
    ax = axes_flat[idx]

    if 'cos' in obs.lower():
        bins = np.linspace(-1, 1, 40)
    else:
        combined = np.concatenate([all_dfs[name][obs].dropna() for name in ALL_NAMES])
        bins = np.linspace(np.percentile(combined, 0.5), np.percentile(combined, 99.5), 40)

    for name in ALL_NAMES:
        ax.hist(
            all_dfs[name][obs].dropna(),
            bins=bins,
            density=True,
            label=name,
            color=COLORS[name],
            **STEP_HIST_KW,
        )

    ax.set_xlabel(obs, fontsize=12)
    ax.set_ylabel('Normalized Density', fontsize=12)
    ax.set_title(f'{obs} (Angular Observable)', fontsize=14)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/angular_distributions.png', dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# Plot 9: Invariant mass spectra
# =============================================================================
print("Generating: Invariant mass spectra...")
mass_obs = ['Higgs_Mass', 'm_bbyy']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, obs in zip(axes, mass_obs):
    if obs == 'Higgs_Mass':
        bins = np.linspace(100, 150, 50)
    else:
        combined = np.concatenate([all_dfs[name][obs].dropna() for name in ALL_NAMES])
        bins = np.linspace(np.percentile(combined, 1), np.percentile(combined, 99), 50)

    for name in ALL_NAMES:
        ax.hist(
            all_dfs[name][obs].dropna(),
            bins=bins,
            density=True,
            label=name,
            color=COLORS[name],
            **STEP_HIST_KW,
        )

    ax.set_xlabel(f'{obs} [GeV]', fontsize=12)
    ax.set_ylabel('Normalized Density', fontsize=12)
    ax.set_title(f'{obs} Distribution', fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/invariant_mass_spectra.png', dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# Plot 10: pT spectra (log scale)
# =============================================================================
print("Generating: pT spectra...")
pt_obs = ['Higgs_pT', 'pT_jj', 'LeadJet_pT', 'SubLeadJet_pT', 'LeadPhoton_pT', 'SubLeadPhoton_pT']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()

for idx, obs in enumerate(pt_obs):
    ax = axes_flat[idx]

    combined = np.concatenate([all_dfs[name][obs].dropna() for name in ALL_NAMES])
    bins = np.linspace(0, np.percentile(combined, 99), 50)

    for name in ALL_NAMES:
        ax.hist(
            all_dfs[name][obs].dropna(),
            bins=bins,
            density=True,
            label=name,
            color=COLORS[name],
            **STEP_HIST_KW,
        )

    ax.set_xlabel(f'{obs} [GeV]', fontsize=12)
    ax.set_ylabel('Normalized Density', fontsize=12)
    ax.set_title(f'{obs}', fontsize=14)
    ax.legend(fontsize=7, ncol=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/pt_spectra_log.png', dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# Plot 11: Profile plots
# =============================================================================
print("Generating: Profile plots...")


def make_profile(df, x_var, y_var, bins):
    bin_indices = np.digitize(df[x_var], bins)
    means, stds, bin_centers = [], [], []

    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.sum() > 10:
            means.append(df.loc[mask, y_var].mean())
            stds.append(df.loc[mask, y_var].std() / np.sqrt(mask.sum()))
            bin_centers.append((bins[i-1] + bins[i]) / 2)

    return np.array(bin_centers), np.array(means), np.array(stds)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
profile_configs = [
    ('Eta_jj', 'Higgs_pT', r'$\eta_{jj}$', r'Mean $p_T^{H}$ [GeV]'),
    ('m_bbyy', 'cosThetaStar', r'$m_{bb\gamma\gamma}$ [GeV]', r'Mean $\cos\theta^*$'),
    ('Higgs_pT', 'signed_DeltaPhi_jj', r'$p_T^{H}$ [GeV]', r'Mean signed $\Delta\phi_{jj}$'),
]

for ax, (x_var, y_var, xlabel, ylabel) in zip(axes, profile_configs):
    combined_x = np.concatenate([all_dfs[name][x_var] for name in ALL_NAMES])
    bins = np.linspace(np.percentile(combined_x, 2), np.percentile(combined_x, 98), 20)

    for name in ALL_NAMES:
        x, y, yerr = make_profile(all_dfs[name], x_var, y_var, bins)
        ax.errorbar(x, y, yerr=yerr, fmt='o-', label=name, color=COLORS[name],
                    markersize=4, capsize=2, linewidth=1.5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{ylabel} vs {xlabel}', fontsize=13)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/profile_plots.png', dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# Plot 12: Distribution overlap heatmap
# Overlap = sum(min(p1,p2)) over bins; 1 = identical, 0 = no overlap.
# Low overlap indicates large differences between SM and BSM datasets.
# =============================================================================
comparison_pairs = []
for bsm in BSM_NAMES:
    comparison_pairs.append((f'{bsm} vs SM', all_dfs[bsm], sm_df))
for i in range(len(BSM_NAMES)):
    for j in range(i + 1, len(BSM_NAMES)):
        comparison_pairs.append((f'{BSM_NAMES[i]} vs {BSM_NAMES[j]}',
                                 all_dfs[BSM_NAMES[i]], all_dfs[BSM_NAMES[j]]))

def compute_histogram_overlap(vals1, vals2, bins=50):
    """Compute overlap of two distributions: sum(min(p1,p2)) over bins. Returns 0-1."""
    combined = np.concatenate([vals1, vals2])
    bins_edges = np.linspace(np.percentile(combined, 0.5), np.percentile(combined, 99.5), bins + 1)
    h1, _ = np.histogram(vals1, bins=bins_edges)
    h2, _ = np.histogram(vals2, bins=bins_edges)
    p1 = h1 / (h1.sum() + 1e-10)
    p2 = h2 / (h2.sum() + 1e-10)
    overlap = np.minimum(p1, p2).sum()
    return overlap


print("Generating: Distribution overlap heatmap...")

overlap_results = {}
for label, df1, df2 in comparison_pairs:
    overlap_results[label] = {}
    for obs in observables:
        v1 = df1[obs].dropna().values
        v2 = df2[obs].dropna().values
        if len(v1) < 10 or len(v2) < 10:
            overlap_results[label][obs] = np.nan
        else:
            overlap_results[label][obs] = compute_histogram_overlap(v1, v2)

overlap_df = pd.DataFrame(overlap_results)

fig, ax = plt.subplots(figsize=(max(12, len(overlap_df.columns) * 1.2), max(14, len(overlap_df.index) * 0.45)))
sns.heatmap(overlap_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
            annot_kws={'size': 6}, vmin=0.5, vmax=1.0,
            cbar_kws={'label': 'Overlap (1 = identical, lower = larger difference)'})
ax.set_title('Distribution Overlap: SM vs BSM (lower = larger differences)', fontsize=14)
ax.set_xlabel('Comparison', fontsize=12)
ax.set_ylabel('Observable', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

plt.tight_layout()
plt.savefig('plots/distribution_overlap_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# Plot 13-15: Ratio panel plots (All, Higgs, Non-Higgs)
# =============================================================================
ratio_observables = [
    'Higgs_pT', 'm_bbyy', 'pT_jj', 'Eta_jj', 'signed_DeltaPhi_jj',
    'cosThetaStar', 'costheta1', 'costheta2', 'DPhi_bb', 'Phi1', 'M_jj',
]

print("Generating: Ratio panel plots (All Events)...")
plot_with_ratio_panel(all_events, ratio_observables, '(All Events)', 'ratio_panel_all_events')

print("Generating: Ratio panel plots (Higgs Events)...")
plot_with_ratio_panel(higgs_events, ratio_observables, '(Higgs Events)', 'ratio_panel_higgs_events')

print("Generating: Ratio panel plots (Non-Higgs Events)...")
plot_with_ratio_panel(non_higgs_events, ratio_observables, '(Non-Higgs Events)', 'ratio_panel_non_higgs_events')


# =============================================================================
# Plot 16: Side-by-side comparison
# =============================================================================
print("Generating: Side-by-side comparison...")
key_obs_comparison = ['Higgs_pT', 'm_bbyy', 'signed_DeltaPhi_jj', 'cosThetaStar']

fig, axes = plt.subplots(len(key_obs_comparison), 6, figsize=(24, 4 * len(key_obs_comparison)),
                         gridspec_kw={'width_ratios': [3, 1, 3, 1, 3, 1]})

event_types = [('All Events', all_events), ('Higgs Events', higgs_events),
               ('Non-Higgs Events', non_higgs_events)]

for row_idx, obs in enumerate(key_obs_comparison):
    for col_idx, (event_label, datasets) in enumerate(event_types):
        ax_main = axes[row_idx, col_idx * 2]
        ax_ratio = axes[row_idx, col_idx * 2 + 1]

        data_arrays = [datasets[name][obs].dropna() for name in ALL_NAMES]
        if any(len(d) < 10 for d in data_arrays):
            ax_main.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            ax_ratio.set_visible(False)
            continue

        combined = np.concatenate(data_arrays)
        bins = np.linspace(np.percentile(combined, 1), np.percentile(combined, 99), 25)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        hists = {}
        for name in ALL_NAMES:
            h, _ = np.histogram(datasets[name][obs].dropna(), bins=bins, density=True)
            hists[name] = h

        for name in ALL_NAMES:
            ax_main.step(bin_centers, hists[name], where='mid', label=name,
                         color=COLORS[name], linewidth=1.5)
        ax_main.set_ylabel('Normalized', fontsize=9)
        ax_main.set_xlabel(obs, fontsize=9)
        if row_idx == 0:
            ax_main.set_title(event_label, fontsize=11, fontweight='bold')
        ax_main.legend(fontsize=6, loc='upper right', ncol=2)
        ax_main.grid(True, alpha=0.3)

        sm_hist_safe = np.where(hists['SM'] > 0, hists['SM'], 1e-10)
        mask = hists['SM'] > 0
        for bsm in BSM_NAMES:
            ratio = hists[bsm] / sm_hist_safe
            ax_ratio.step(bin_centers[mask], ratio[mask], where='mid',
                          color=COLORS[bsm], linewidth=1.5)
        ax_ratio.axhline(y=1, color='blue', linestyle='-', linewidth=1)
        ax_ratio.fill_between(bin_centers, 0.9, 1.1, alpha=0.2, color='gray')
        ax_ratio.set_ylabel('Ratio', fontsize=8)
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.set_xlabel(obs, fontsize=8)
        ax_ratio.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/side_by_side_comparison.png', dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# Plot 17: BSM deviation summary (overlap: lower = larger difference)
# =============================================================================
print("Generating: BSM deviation summary (overlap)...")
summary_obs = ['Higgs_pT', 'm_bbyy', 'pT_jj', 'Eta_jj', 'signed_DeltaPhi_jj',
               'cosThetaStar', 'costheta1', 'costheta2']

results = []
for obs in summary_obs:
    for bsm_name in BSM_NAMES:
        v_h_sm = higgs_events['SM'][obs].dropna().values
        v_h_bsm = higgs_events[bsm_name][obs].dropna().values
        v_nh_sm = non_higgs_events['SM'][obs].dropna().values
        v_nh_bsm = non_higgs_events[bsm_name][obs].dropna().values
        ov_h = compute_histogram_overlap(v_h_sm, v_h_bsm) if len(v_h_sm) >= 10 and len(v_h_bsm) >= 10 else np.nan
        ov_nh = compute_histogram_overlap(v_nh_sm, v_nh_bsm) if len(v_nh_sm) >= 10 and len(v_nh_bsm) >= 10 else np.nan
        results.append({
            'Observable': obs, 'BSM Model': bsm_name,
            'Overlap (Higgs)': ov_h, 'Overlap (Non-Higgs)': ov_nh,
        })

results_df = pd.DataFrame(results)

n_bsm = len(BSM_NAMES)
n_cols_bsm = 3
n_rows_bsm = (n_bsm + n_cols_bsm - 1) // n_cols_bsm
fig, axes = plt.subplots(n_rows_bsm, n_cols_bsm, figsize=(8 * n_cols_bsm, 6 * n_rows_bsm))
axes_flat = axes.flatten()

for idx, bsm in enumerate(BSM_NAMES):
    ax = axes_flat[idx]
    df_bsm = results_df[results_df['BSM Model'] == bsm]

    x = np.arange(len(summary_obs))
    width = 0.35

    ax.bar(x - width/2, df_bsm['Overlap (Higgs)'].values, width,
           label='Higgs Events', color='blue', alpha=0.7)
    ax.bar(x + width/2, df_bsm['Overlap (Non-Higgs)'].values, width,
           label='Non-Higgs Events', color='red', alpha=0.7)

    ax.set_xlabel('Observable', fontsize=11)
    ax.set_ylabel('Overlap (lower = larger diff)', fontsize=10)
    ax.set_title(f'{bsm} vs SM: Higgs vs Non-Higgs', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_obs, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plots/bsm_deviation_higgs_vs_nonhiggs.png', dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("All plots generated successfully!")
print("=" * 60)

plot_files = sorted([f for f in os.listdir('plots') if f.endswith('.png')])
print(f"\nGenerated {len(plot_files)} plots in 'plots/' folder:")
for f in plot_files:
    print(f"  - {f}")
