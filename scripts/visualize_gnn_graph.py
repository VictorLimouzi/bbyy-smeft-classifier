#!/usr/bin/env python3
"""
Generate publication-quality visualizations of the GNN input graph structure.
Output: figures/gnn_input_graph.png and .pdf

Uses only matplotlib + networkx (no pyvis). Optimized for paper figures.
"""

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch
import networkx as nx

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.2,
    'figure.dpi': 150,
})
os.makedirs('figures', exist_ok=True)


def get_physics_layout():
    """
    Physics-informed layout: reflects HH→bbyy topology.
    Higgs at top (parent), b-jets below left, photons below right.
    """
    pos = {
        0: (0.5, 1.0),   # Higgs (top center)
        1: (-0.6, 0.2),  # LeadJet (bottom left)
        2: (-0.3, 0.0),  # SubLeadJet
        3: (0.3, 0.0),   # LeadPhoton (bottom right)
        4: (0.6, 0.2),   # SubLeadPhoton
    }
    return pos


def get_circular_layout():
    """Circular layout — clean, symmetric."""
    theta = np.linspace(0, 2 * np.pi, 6)[:-1] + np.pi / 2
    r = 1.0
    pos = {i: (r * np.cos(t), r * np.sin(t)) for i, t in enumerate(theta)}
    return pos


def draw_gnn_graph(style='physics', edge_alpha=0.25, save=True):
    """
    Draw the GNN input graph (5 nodes, fully connected).
    
    style: 'physics' (topology-aware) or 'circular'
    edge_alpha: transparency of edges (lower = less clutter)
    """
    G = nx.complete_graph(5)  # Fully connected, no self-loops
    
    if style == 'physics':
        pos = get_physics_layout()
    else:
        pos = get_circular_layout()
    
    # Node metadata
    node_names = ['H', r'$J_1$', r'$J_2$', r'$\gamma_1$', r'$\gamma_2$']
    node_colors = ['#E65100', '#1565C0', '#1565C0', '#C62828', '#C62828']  # Higgs, b-jets, photons
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw edges first (behind nodes)
    nx.draw_networkx_edges(
        G, pos,
        edge_color='#9E9E9E',
        alpha=edge_alpha,
        width=0.8,
        ax=ax,
    )
    
    # Draw nodes (circles, color-coded by type)
    for i in range(5):
        x, y = pos[i]
        color = node_colors[i]
        circle = Circle((x, y), 0.12, facecolor=color, edgecolor='black', linewidth=1.5, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, node_names[i], ha='center', va='center', fontsize=12, color='white', fontweight='bold', zorder=11)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#E65100', edgecolor='black', label='Higgs'),
        mpatches.Patch(facecolor='#1565C0', edgecolor='black', label='b-jets'),
        mpatches.Patch(facecolor='#C62828', edgecolor='black', label='Photons'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)
    
    # Title and caption
    ax.set_title('GNN Input: Event as Graph (HH→bbyy)', fontsize=14, fontweight='bold', pad=15)
    ax.text(0.5, -0.25, '5 nodes, 20 edges (fully connected)\nNode features: $p_T$, $\\eta$, $\\phi$, $M$',
            ha='center', fontsize=10, style='italic', transform=ax.transAxes)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('figures/gnn_input_graph.png', dpi=300, bbox_inches='tight')
        plt.savefig('figures/gnn_input_graph.pdf', bbox_inches='tight')
        print("Saved figures/gnn_input_graph.png and .pdf")
    plt.close()


def draw_gnn_graph_minimal():
    """
    Minimal version: clean schematic for paper.
    No edge clutter — only node topology.
    """
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4)])  # Star-like: Higgs connects all; jets paired, photons paired
    
    pos = {
        0: (0, 0.8),
        1: (-0.5, 0),
        2: (-0.2, -0.1),
        3: (0.2, -0.1),
        4: (0.5, 0),
    }
    
    node_names = ['H', r'$J_1$', r'$J_2$', r'$\gamma_1$', r'$\gamma_2$']
    node_colors = ['#E65100', '#1565C0', '#1565C0', '#C62828', '#C62828']
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_aspect('equal')
    ax.axis('off')
    
    nx.draw_networkx_edges(G, pos, edge_color='#333', width=1.5, ax=ax)
    
    for i in range(5):
        x, y = pos[i]
        circle = Circle((x, y), 0.1, facecolor=node_colors[i], edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y, node_names[i], ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    
    ax.set_title('Event Graph (Simplified Topology)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/gnn_input_graph_minimal.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/gnn_input_graph_minimal.pdf', bbox_inches='tight')
    plt.close()
    print("Saved figures/gnn_input_graph_minimal.png and .pdf")


def draw_gnn_with_feature_table():
    """
    Graph + small feature table — shows both structure and input format.
    """
    G = nx.complete_graph(5)
    pos = get_physics_layout()
    
    node_names = ['H', r'$J_1$', r'$J_2$', r'$\gamma_1$', r'$\gamma_2$']
    node_colors = ['#E65100', '#1565C0', '#1565C0', '#C62828', '#C62828']
    
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    for ax in [ax1, ax2]:
        ax.set_aspect('equal' if ax == ax1 else 'auto')
        ax.axis('off')
    
    # Left: Graph
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='#BDBDBD', alpha=0.4, width=0.6)
    for i in range(5):
        x, y = pos[i]
        circle = Circle((x, y), 0.1, facecolor=node_colors[i], edgecolor='black', linewidth=1.2)
        ax1.add_patch(circle)
        ax1.text(x, y, node_names[i], ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    ax1.set_title('Graph Structure', fontsize=12, fontweight='bold')
    
    # Right: Feature table
    rows = ['Higgs', 'Lead Jet', 'SubLead Jet', 'Lead Photon', 'SubLead Photon']
    cols = [r'$p_T$', r'$\eta$', r'$\phi$', r'$M$']
    data = [['—', '—', '—', '—'] for _ in range(5)]
    table = ax2.table(
        cellText=data,
        rowLabels=rows,
        colLabels=cols,
        loc='center',
        cellLoc='center',
        colColours=['#E3F2FD'] * 4,
        rowColours=[node_colors[i] for i in range(5)],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    ax2.set_title('Node Features (4 per node)', fontsize=12, fontweight='bold')
    
    plt.suptitle('GNN Input Representation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/gnn_input_graph_with_features.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/gnn_input_graph_with_features.pdf', bbox_inches='tight')
    plt.close()
    print("Saved figures/gnn_input_graph_with_features.png and .pdf")


if __name__ == '__main__':
    draw_gnn_graph(style='physics', edge_alpha=0.2)
    draw_gnn_graph(style='circular', edge_alpha=0.2, save=False)  # Optional: uncomment to also save circular
    draw_gnn_graph_minimal()
    draw_gnn_with_feature_table()
    print("\nDone. Use .pdf for LaTeX (vector graphics).")
