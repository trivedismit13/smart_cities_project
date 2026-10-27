"""
Publication-Quality Figure Regeneration Script for Smart Cities FL Trajectory Prediction.

Every plotted value is read from the result files written by run_full_journal_suite.py,
run_dp_sgd_training.py and tune_fedprox_pfedme.py (results/), or from the preprocessing
cache (Fig 1). Nothing is typed in by hand.

- Fig 1: Dataset volume and kinematic distributions in SI units (from the cache).
- Fig 4: Cross-zone ADE heatmap.
- Fig 5: Accuracy vs communication volume.
- Fig 6: Cumulative ADE across horizons.
- Fig 7: Controlled heterogeneity regimes.
- Fig 8: DLG reconstruction vs noise multiplier.
- Fig 9: Spatial fairness and disparity.
- Fig 10: (a) Accuracy vs communication, including FedAvg trained with DP-SGD;
          (b) DLG reconstruction across defences, with the DP-SGD accountant epsilons.
pFedMe is not plotted: no configuration of its validation grid converged (results/tuning_grid_validation.csv).
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_FILE = os.path.join(BASE_DIR, "cache", "federated_trajectory_data.npz")
FEET_TO_METERS = 0.3048

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

ZONES = ['US-101 (Highway)', 'Lankershim (Arterial)', 'Peachtree (Downtown)']
POOLED = 'Global Pooled'

# (result-file model name, display label, colour, marker)
MODELS = [
    ('Constant Velocity', 'Constant Velocity', '#7f7f7f', 's'),
    ('Constant Acceleration', 'Constant Acceleration', '#8c564b', '^'),
    ('Centralized (Pooled)', 'Centralized (Pooled)', '#1f77b4', 'D'),
    ('Centralized (Zone-Aware)', 'Centralized (Zone-Aware)', '#17becf', 'P'),
    ('FedAvg (Global FL)', 'FedAvg (Global FL)', '#d62728', 'o'),
    ('FedProx (mu=0.01)', 'FedProx (mu=0.01)', '#ff7f0e', 's'),
    ('FedPer (Personalized)', 'FedPer (Personalized)', '#2ca02c', '*'),
    ('Isolated (US-101 Silo)', 'Isolated (US-101 Silo)', '#bcbd22', '>'),
    ('Isolated (Lankershim Silo)', 'Isolated (Lankershim Silo)', '#e377c2', '<'),
    ('Isolated (Peachtree Silo)', 'Isolated (Peachtree Silo)', '#393b79', 'd'),
    ('Isolated (Own Zone)', 'Isolated (Own Zone)', '#9467bd', 'h'),
]


def set_style():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 10,
        'axes.labelsize': 10.5,
        'axes.titlesize': 11.5,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 9.0,
        'figure.titlesize': 12.5,
        'figure.dpi': 300,
        'lines.linewidth': 2.0,
        'lines.markersize': 6.5
    })


def sync_file(filename):
    src = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(FIGURES_DIR, filename))


def save(fig_stem, extra_png=None):
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(PLOTS_DIR, f"{fig_stem}.{ext}"), dpi=300, bbox_inches='tight')
        sync_file(f"{fig_stem}.{ext}")
    if extra_png:
        plt.savefig(os.path.join(PLOTS_DIR, extra_png), dpi=300, bbox_inches='tight')
        sync_file(extra_png)
    plt.close()


def load_bench():
    return pd.read_csv(os.path.join(RESULTS_DIR, "journal_benchmark_results.csv")).set_index('Model')


def fedprox_name(bench):
    return [m for m in bench.index if m.startswith('FedProx')][0]


def models_present(bench):
    out = []
    for m, label, col, mk in MODELS:
        if m.startswith('FedProx'):
            m = label = fedprox_name(bench)
        out.append((m, label, col, mk))
    return out


def load_dp_sgd():
    """Per-seed DP-SGD training results, merged from all sigma files."""
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('dp_sgd') and f.endswith('.csv')]
    if not files:
        return None
    df = pd.concat([pd.read_csv(os.path.join(RESULTS_DIR, f)) for f in files], ignore_index=True)
    return df.drop_duplicates(subset=['Seed', 'Noise Multiplier (sigma)', 'Domain'])


def dp_epsilon_text(dp):
    lines = []
    for sigma, grp in dp[dp['Domain'].isin(ZONES)].groupby('Noise Multiplier (sigma)'):
        eps = grp.groupby('Domain')['Epsilon (delta=1e-5)'].first()
        lines.append(rf"• $\sigma={sigma:g}$: worst-case $\epsilon={eps.max():.2f}$ ({eps.min():.2f}–{eps.max():.2f})")
    return lines


# ==============================================================================
# FIG 1: DATASET DISTRIBUTION & PHYSICAL KINEMATIC PROFILES (SI Units: m/s)
# ==============================================================================
def generate_fig1_distribution():
    set_style()
    if not os.path.exists(CACHE_FILE):
        print(f"[Fig 1] Cache file not found: {CACHE_FILE}. Skipping Fig 1 generation.")
        return

    data = np.load(CACHE_FILE, allow_pickle=True)
    x_mean = data['x_mean']
    x_std = data['x_std']
    num_clients = int(data['num_clients'])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    ax = axes[0, 0]
    x = np.arange(num_clients)
    width = 0.25
    train_counts = [len(data[f'client_{cid}_train_x']) for cid in range(num_clients)]
    val_counts = [len(data[f'client_{cid}_val_x']) for cid in range(num_clients)]
    test_counts = [len(data[f'client_{cid}_test_x']) for cid in range(num_clients)]
    ax.bar(x - width, train_counts, width, label='Train (70%)', color='#2b5c8f', edgecolor='black', alpha=0.9)
    ax.bar(x, val_counts, width, label='Val (15%)', color='#4ba3e3', edgecolor='black', alpha=0.9)
    ax.bar(x + width, test_counts, width, label='Test (15%)', color='#9ed2f6', edgecolor='black', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(ZONES, fontweight='bold', fontsize=9.5)
    ax.set_ylabel('Trajectory Sequence Count', fontsize=10.5, fontweight='bold')
    ax.set_title('(a) Non-IID Client Data Volume Partitioning', fontsize=11.5, fontweight='bold')
    ax.legend(frameon=True, fontsize=9.0)
    ax.grid(True, linestyle='--', alpha=0.6)

    ax = axes[0, 1]
    for cid in range(num_clients):
        v_ms = (data[f'client_{cid}_train_x'][:, :, 2].flatten() * x_std[2] + x_mean[2]) * FEET_TO_METERS
        ax.hist(v_ms[v_ms >= 0.0], bins=30, alpha=0.55, color=colors[cid],
                label=f"{ZONES[cid]} (Mean: {np.mean(v_ms):.1f} m/s)", density=True)
    ax.set_xlabel('Instantaneous Vehicle Speed (m/s)', fontsize=10.5, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=10.5, fontweight='bold')
    ax.set_title('(b) Spatial-Temporal Kinematic Heterogeneity (Speed in m/s)', fontsize=11.5, fontweight='bold')
    ax.legend(frameon=True, fontsize=9.0)
    ax.grid(True, linestyle='--', alpha=0.6)

    ax = axes[1, 0]
    for cid in range(num_clients):
        raw_y = data[f'client_{cid}_raw_train_y'][:, -1, 1] * FEET_TO_METERS
        ax.hist(raw_y, bins=30, alpha=0.55, color=colors[cid],
                label=f"{ZONES[cid]} (Max: {np.max(raw_y):.1f} m)", density=True)
    ax.set_xlabel('3.0s Forward Longitudinal Displacement (meters)', fontsize=10.5, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=10.5, fontweight='bold')
    ax.set_title('(c) Longitudinal Motion Span Distribution (meters)', fontsize=11.5, fontweight='bold')
    ax.legend(frameon=True, fontsize=9.0)
    ax.grid(True, linestyle='--', alpha=0.6)

    ax = axes[1, 1]
    for cid in range(num_clients):
        raw_targets = data[f'client_{cid}_raw_train_y'][:15] * FEET_TO_METERS
        for track in raw_targets:
            full_track = np.vstack([[0, 0], track])
            ax.plot(full_track[:, 0], full_track[:, 1], color=colors[cid], alpha=0.4, linewidth=1.2)
        ax.plot([], [], color=colors[cid], label=f"{ZONES[cid]} Paths")
    ax.plot(0, 0, 'ko', markersize=6, label='Observation Origin (t=0)')
    ax.set_xlabel(r'Lateral Displacement $\Delta X$ (meters)', fontsize=10.5, fontweight='bold')
    ax.set_ylabel(r'Longitudinal Displacement $\Delta Y$ (meters)', fontsize=10.5, fontweight='bold')
    ax.set_title('(d) Trajectory Geometry Variations Across Functional Corridors', fontsize=11.5, fontweight='bold')
    ax.legend(frameon=True, loc='upper left', fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig1_dataset_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    sync_file("fig1_dataset_distribution.png")
    print("[Fig 1] Generated dataset distribution plot.")


# ==============================================================================
# FIG 4: CROSS-ZONE ADE HEATMAP
# ==============================================================================
def generate_fig4_heatmap():
    set_style()
    bench = load_bench()
    models = models_present(bench)
    domains = ZONES + [POOLED]
    domain_labels = ['US-101\n(Highway)', 'Lankershim\n(Arterial)', 'Peachtree\n(Downtown)', 'Global Pooled\n(Overall)']
    matrix = np.array([[bench.loc[m, f'{d}_ADE_mean'] for d in domains] for m, _, _, _ in models])

    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    im = ax.imshow(matrix, cmap='YlGnBu_r', aspect='auto')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Average Displacement Error (ADE in meters)', fontsize=10, fontweight='bold', rotation=-90, va='bottom')
    cbar.ax.tick_params(labelsize=9)
    ax.set_xticks(range(len(domain_labels)))
    ax.set_xticklabels(domain_labels, fontsize=9.5, fontweight='bold')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([label for _, label, _, _ in models], fontsize=9.5)
    ax.grid(False)

    names = [m for m, _, _, _ in models]
    own_zone_coords = [(names.index('Isolated (US-101 Silo)'), 0), (names.index('Isolated (Lankershim Silo)'), 1),
                       (names.index('Isolated (Peachtree Silo)'), 2), (names.index('Isolated (Own Zone)'), 3)]
    for r, c in own_zone_coords:
        ax.add_patch(plt.Rectangle((c - 0.48, r - 0.48), 0.96, 0.96, fill=False, edgecolor='crimson', lw=1.8, linestyle='--'))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            rgba = im.cmap(im.norm(val))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, f"{val:.2f}m", ha='center', va='center', fontsize=10.0,
                    fontweight='bold', color="black" if lum > 0.48 else "white")

    ax.set_title('Cross-Zone Average Displacement Error (ADE in meters)\n[Red Dashed Box = Dedicated In-Jurisdiction Native Evaluation]',
                 fontsize=11.5, fontweight='bold', pad=12)
    plt.tight_layout()
    save("fig4_cross_zone_ade", "fig4_cross_domain_generalization.png")
    print("[Fig 4] Generated cross-zone ADE heatmap.")


# ==============================================================================
# FIG 5: WIRELESS COMMUNICATION VOLUME VS ACCURACY
# ==============================================================================
def generate_fig5_communication():
    set_style()
    bench = load_bench()
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for m, label, col, mk in models_present(bench):
        comm, ade = bench.loc[m, 'Comm (MiB)'], bench.loc[m, f'{POOLED}_ADE_mean']
        ax.scatter(comm, ade, color=col, marker=mk, s=110 if mk == '*' else 70,
                   edgecolors='black', linewidth=1.1, zorder=5, label=f"{label} ({ade:.3f}m)")
    ax.set_xlabel('Wireless Communication Volume across 12 Rounds (MiB)', fontsize=10.5, fontweight='bold')
    ax.set_ylabel('Global Pooled Test ADE (meters)', fontsize=10.5, fontweight='bold')
    ax.set_title('Accuracy vs Wireless Communication Volume Trade-off', fontsize=11.5, fontweight='bold')
    ax.set_xlim(-4, 45)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, loc='upper center', fontsize=7.6, ncol=2)
    ax.set_ylim(0.6, bench[f'{POOLED}_ADE_mean'].drop([m for m in bench.index if m.startswith('pFedMe')]).max() * 1.45)
    plt.tight_layout()
    save("fig5_communication", "fig5_communication_privacy_tradeoff.png")
    print("[Fig 5] Generated communication trade-off plot.")


# ==============================================================================
# FIG 6: HORIZON ADE ERROR ACCUMULATION
# ==============================================================================
def generate_fig6_horizon():
    set_style()
    bench = load_bench()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    h_steps = [1.0, 2.0, 3.0]
    styles = {'Constant Velocity': 's--', 'Centralized (Pooled)': 'D-', 'FedAvg (Global FL)': 'o-',
              'FedPer (Personalized)': '*-', 'Isolated (Own Zone)': 'v-.'}
    for m, label, col, _ in models_present(bench):
        if m in styles:
            vals = [bench.loc[m, f'{POOLED}_ADE_1.0s'], bench.loc[m, f'{POOLED}_ADE_2.0s'], bench.loc[m, f'{POOLED}_ADE_mean']]
            ax.plot(h_steps, vals, styles[m], color=col, linewidth=2.2, markersize=7, label=label)
    ax.set_xlabel('Prediction Horizon (seconds ahead)', fontsize=10.5, fontweight='bold')
    ax.set_ylabel('Cumulative ADE (meters)', fontsize=10.5, fontweight='bold')
    ax.set_title('Trajectory Forecast Error Compounding Across Horizons (1.0s to 3.0s)', fontsize=11.5, fontweight='bold')
    ax.set_xticks(h_steps)
    ax.set_xticklabels(['1.0s (Steps 1-2)', '2.0s (Steps 1-4)', '3.0s (Full 6 Steps)'], fontsize=9.5, fontweight='bold')
    ax.legend(frameon=True, loc='upper left', fontsize=9.5)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save("fig6_horizon_ade", "fig6_horizon_error_growth.png")
    print("[Fig 6] Generated horizon ADE plot.")


# ==============================================================================
# FIG 7: CONTROLLED HETEROGENEITY REGIMES
# ==============================================================================
def generate_fig7_heterogeneity():
    set_style()
    het_df = pd.read_csv(os.path.join(RESULTS_DIR, "journal_controlled_heterogeneity_results.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    regimes = list(het_df['Heterogeneity Regime'])
    x_idx = np.arange(len(regimes))
    w = 0.35
    f_ade = list(het_df['FedAvg Mean Zone ADE (m)'])
    p_ade = list(het_df['FedPer Mean Zone ADE (m)'])
    deltas = list(het_df['Delta (m)'])

    axes[0].bar(x_idx - w/2, f_ade, w, label='Canonical FedAvg (Global)', color='#d62728', edgecolor='black', alpha=0.85)
    axes[0].bar(x_idx + w/2, p_ade, w, label='FedPer (Personalized Heads)', color='#2ca02c', edgecolor='black', alpha=0.85)
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels([f"{r}\nRegime" for r in regimes], fontsize=9.5, fontweight='bold')
    axes[0].set_ylabel('Mean Client Test ADE (meters)', fontsize=10.5, fontweight='bold')
    axes[0].set_title('(a) Convergence Degradation Under Geographic Non-IID Shift', fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, max(f_ade + p_ade) * 1.25)
    axes[0].legend(frameon=True, fontsize=9.5)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    bars = axes[1].bar(regimes, deltas, color='#1f77b4', edgecolor='black', width=0.45, alpha=0.85)
    axes[1].set_xlabel('Geographic Heterogeneity Regime', fontsize=10.5, fontweight='bold')
    axes[1].set_ylabel('Performance Difference: FedAvg - FedPer (m)', fontsize=10.5, fontweight='bold')
    axes[1].set_title('(b) Early-Round Local Specialization Response (3 Rounds)', fontsize=11, fontweight='bold')
    axes[1].set_ylim(min(deltas) * 1.4, 0.05)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    for bar, d_val in zip(bars, deltas):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, d_val - 0.01, f"{d_val:.3f}m",
                     ha='center', va='top', fontweight='bold', fontsize=9.5)
    plt.tight_layout()
    save("fig7_heterogeneity_regimes", "fig7_heterogeneity_vs_personalization.png")
    print("[Fig 7] Generated controlled heterogeneity plot.")


# ==============================================================================
# FIG 8: PRIVACY AUDITING (DLG Reconstruction vs DP Noise Multiplier)
# ==============================================================================
def load_dlg():
    dlg = pd.read_csv(os.path.join(RESULTS_DIR, "journal_privacy_dlg_results.csv"))
    fa = dlg[dlg['FL Method'] == 'FedAvg'].sort_values('Noise Multiplier (sigma)')
    fp = dlg[dlg['FL Method'] == 'FedPer'].sort_values('Noise Multiplier (sigma)')
    return fa, fp


def generate_fig8_privacy():
    set_style()
    fa, fp = load_dlg()
    dp = load_dp_sgd()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    axes[0].plot(fa['Noise Multiplier (sigma)'], fa['DLG Position Error (m)'], 'o-', color='#d62728',
                 label='FedAvg (Full Gradient Exchange)', linewidth=2.2, markersize=7)
    axes[0].plot(fp['Noise Multiplier (sigma)'], fp['DLG Position Error (m)'], 's--', color='#2ca02c',
                 label='FedPer (Backbone Shared Gradients)', linewidth=2.2, markersize=7)
    axes[0].axhline(y=3.7, color='crimson', linestyle=':', linewidth=1.6, label='Highway Lane Width Threshold (3.7m)')
    axes[0].axhline(y=3.0, color='darkorange', linestyle='--', linewidth=1.2, label='Arterial Lane Width Threshold (3.0m)')
    for s_val, fa_e in zip(fa['Noise Multiplier (sigma)'], fa['DLG Position Error (m)']):
        axes[0].annotate(f"{fa_e:.2f}m", (s_val, fa_e), xytext=(-28, 10), textcoords='offset points',
                         fontsize=8.2, fontweight='bold')
    if dp is not None:
        info_box = "\n".join(["Separate DP-SGD Training Accounting",
                              r"($|B|=128$, 12 Rounds, $E=1$, $\delta=10^{-5}$):"] + dp_epsilon_text(dp) +
                             ["(RDP accountant of the trained models,", "not of single-window inversion)"])
        axes[0].text(0.40, 0.04, info_box, transform=axes[0].transAxes, fontsize=7.8, fontweight='semibold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#6c757d', alpha=0.95))
    axes[0].set_xlabel(r'Differential Privacy Noise Multiplier ($\sigma$) on Evaluated Gradients', fontsize=10.5, fontweight='bold')
    axes[0].set_ylabel('DLG Reconstruction Position Error (meters)', fontsize=10.5, fontweight='bold')
    axes[0].set_title('(a) Empirical Defense Efficacy Against Gradient Inversion ($|B|=1$)', fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, max(fa['DLG Position Error (m)'].max(), fp['DLG Position Error (m)'].max()) * 1.4)
    axes[0].legend(frameon=True, loc='upper left', fontsize=8.2)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(fa['Noise Multiplier (sigma)'], fa['Input Cosine Similarity'], 'o-', color='#d62728',
                 label='FedAvg Gradient Inversion Fidelity', linewidth=2.2, markersize=7)
    axes[1].plot(fp['Noise Multiplier (sigma)'], fp['Input Cosine Similarity'], 's--', color='#2ca02c',
                 label='FedPer Gradient Inversion Fidelity', linewidth=2.2, markersize=7)
    axes[1].axhline(y=0.0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel(r'Differential Privacy Noise Multiplier ($\sigma$) on Evaluated Gradients', fontsize=10.5, fontweight='bold')
    axes[1].set_ylabel('Trajectory Geometry Cosine Similarity', fontsize=10.5, fontweight='bold')
    axes[1].set_title('(b) Coordinate Geometry Correlation Collapse', fontsize=11, fontweight='bold')
    axes[1].set_ylim(-0.6, 1.0)
    axes[1].legend(frameon=True, loc='upper right', fontsize=8.5)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save("fig8_dlg_reconstruction", "fig8_privacy_reconstruction_vs_dp.png")
    print("[Fig 8] Generated privacy audit plot.")


# ==============================================================================
# FIG 9: SPATIAL FAIRNESS DISPARITY
# ==============================================================================
def generate_fig9_fairness():
    set_style()
    bench = load_bench()
    fair_df = pd.read_csv(os.path.join(RESULTS_DIR, "journal_fairness_results.csv")).set_index('Model')
    models = ['Centralized (Pooled)', 'FedAvg (Global FL)', fedprox_name(bench), 'FedPer (Personalized)', 'Isolated (Own Zone)']
    labels_short = ['Centralized', 'FedAvg', 'FedProx', 'FedPer (PFL)', 'Isolated (Own)']
    f_sub = fair_df.loc[models]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    x_pos = np.arange(len(models))
    w = 0.22

    axes[0].bar(x_pos - w, f_sub['Best-Zone ADE (m)'], w, label='Best-Performing Zone', color='#2ca02c', edgecolor='black', alpha=0.85)
    axes[0].bar(x_pos, f_sub['Mean Zone ADE (m)'], w, label='Corridor Mean', color='#1f77b4', edgecolor='black', alpha=0.85)
    axes[0].bar(x_pos + w, f_sub['Worst-Zone ADE (m)'], w, label='Worst-Performing Zone', color='#d62728', edgecolor='black', alpha=0.85)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels_short, rotation=12, fontsize=9.0, fontweight='bold')
    axes[0].set_ylabel('Average Displacement Error (meters)', fontsize=10.5, fontweight='bold')
    axes[0].set_title('(a) Regional Performance Spread Across Corridor Types', fontsize=11, fontweight='bold')
    axes[0].legend(frameon=True, fontsize=8.8)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    gaps = list(f_sub['Disparity Gap (m)'])
    covs = list(f_sub['Fairness CoV (%)'])
    bars = axes[1].bar(x_pos, gaps, width=0.48, color=['#9467bd'] * 4 + ['#7f7f7f'], edgecolor='black', alpha=0.85)
    for b, g, c in zip(bars, gaps, covs):
        axes[1].text(b.get_x() + b.get_width() / 2.0, g + 0.005, f"{g:.3f}m\n({c:.1f}%)",
                     ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels_short, rotation=12, fontsize=9.0, fontweight='bold')
    axes[1].set_ylabel('Disparity Gap: Worst - Best ADE (meters)', fontsize=10.5, fontweight='bold')
    axes[1].set_title('(b) Inter-Zone Disparity Gap [Lower = More Equitable]', fontsize=11, fontweight='bold')
    axes[1].set_ylim(0, max(gaps) * 1.3)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save("fig9_zone_disparity", "fig9_fairness_disparity_comparison.png")
    print("[Fig 9] Generated spatial fairness plot.")


# ==============================================================================
# FIG 10: TRADE-OFF FRONTIER (measured results only)
# ==============================================================================
def generate_fig10_pareto():
    set_style()
    bench = load_bench()
    dp = load_dp_sgd()
    fa, fp = load_dlg()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))

    ax0 = axes[0]
    for m, label, col, mk in models_present(bench):
        if m.startswith('Isolated (') and m != 'Isolated (Own Zone)':
            continue
        comm, ade = bench.loc[m, 'Comm (MiB)'], bench.loc[m, f'{POOLED}_ADE_mean']
        ax0.scatter(comm, ade, color=col, marker=mk, s=110 if mk == '*' else 70, edgecolors='black',
                    linewidth=1.1, zorder=5, label=f"{label} ({ade:.3f}m)")
    if dp is not None:
        fedavg_comm = bench.loc['FedAvg (Global FL)', 'Comm (MiB)']
        pooled = dp[dp['Domain'] == POOLED].groupby('Noise Multiplier (sigma)')['ADE (m)'].agg(['mean', 'std'])
        dp_cols = {0.5: '#e377c2', 1.0: '#8c564b', 2.0: '#17becf'}
        for sigma, row in pooled.iterrows():
            ax0.errorbar(fedavg_comm, row['mean'], yerr=row['std'], fmt='X', color=dp_cols.get(sigma, 'k'),
                         markersize=9, markeredgecolor='black', capsize=3, zorder=6,
                         label=rf"FedAvg + DP-SGD $\sigma={sigma:g}$ ({row['mean']:.3f}m)")
    ax0.set_xlabel('Wireless Communication Volume across 12 Rounds (MiB)', fontsize=10.5, fontweight='bold')
    ax0.set_ylabel('Global Pooled Test ADE (meters) [Lower = Better]', fontsize=10.5, fontweight='bold')
    ax0.set_title('(a) Measured Utility vs Communication Volume', fontsize=11.5, fontweight='bold')
    ax0.set_xlim(-5, 45)
    ax0.legend(frameon=True, loc='upper left', fontsize=7.4)
    ax0.grid(True, linestyle='--', alpha=0.6)

    ax1 = axes[1]
    cats = [('Unprotected\nFedAvg', float(fa[fa['Noise Multiplier (sigma)'] == 0.0]['DLG Position Error (m)'].iloc[0]), '#d62728'),
            ('FedPer\n(Head Withheld)', float(fp[fp['Noise Multiplier (sigma)'] == 0.0]['DLG Position Error (m)'].iloc[0]), '#2ca02c')]
    for sigma, col in [(0.5, '#e377c2'), (1.0, '#8c564b'), (2.0, '#17becf')]:
        cats.append((f'Gaussian noise\n' + rf'($\sigma={sigma:g}$)',
                     float(fa[fa['Noise Multiplier (sigma)'] == sigma]['DLG Position Error (m)'].iloc[0]), col))
    x_cat = np.arange(len(cats))
    ax1.bar(x_cat, [c[1] for c in cats], width=0.48, color=[c[2] for c in cats], alpha=0.45, edgecolor='black', linewidth=1.0)
    for idx, (_, val, _) in enumerate(cats):
        ax1.annotate(f"{val:.2f}m", (idx, val), xytext=(0, 5), textcoords='offset points', fontsize=8.5, fontweight='bold', ha='center')
    ax1.axhline(y=3.7, color='crimson', linestyle=':', linewidth=1.6, label='Highway Lane Width (3.7m)')
    ax1.axhline(y=3.0, color='darkorange', linestyle='--', linewidth=1.2, label='Arterial Lane Width (3.0m)')
    if dp is not None:
        dp_box = "\n".join(["DP-SGD training accounting",
                            r"($|B|=128$, 12 Rounds, $E=1$, $\delta=10^{-5}$):"] + dp_epsilon_text(dp) +
                           ["(Opacus RDP accountant; distinct from", r"the empirical $|B|=1$ DLG inversion)"])
        ax1.text(0.03, 0.97, dp_box, transform=ax1.transAxes, fontsize=7.6, fontweight='semibold',
                 va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#495057', alpha=0.95))
    ax1.set_xticks(x_cat)
    ax1.set_xticklabels([c[0] for c in cats], fontsize=8.8, fontweight='bold')
    ax1.set_ylabel('Empirical DLG Reconstruction Error (meters) [Higher = Better Defense]', fontsize=10.5, fontweight='bold')
    ax1.set_title('(b) Empirical Inversion Defense Across Privacy Mechanisms', fontsize=11.5, fontweight='bold')
    ax1.set_ylim(0.0, max(c[1] for c in cats) * 1.9)
    ax1.legend(frameon=True, loc='center right', fontsize=8.0)
    ax1.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save("fig10_accuracy_communication", "fig10_pareto_tradeoff_frontier.png")
    print("[Fig 10] Generated trade-off figure.")


def main():
    generate_fig1_distribution()
    generate_fig4_heatmap()
    generate_fig5_communication()
    generate_fig6_horizon()
    generate_fig7_heterogeneity()
    generate_fig8_privacy()
    generate_fig9_fairness()
    generate_fig10_pareto()
    for f in ["fig2_learning_curves.png", "fig3_trajectory_predictions.png"]:
        sync_file(f)
    print("All figures regenerated from results/.")


if __name__ == '__main__':
    main()
