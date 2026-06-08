#!/usr/bin/env python3
"""Summary plot of all EagleEye sensitivity results."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Results (hardcoded from completed runs) ───────────────────────────────────

# Model-matched time scaling (new), NuMix 1:1:1
tau_vals   = [20, 100, 200, 400]          # signal duration in seconds
sens_match = [30.5, 48.0, 62.5, 81.0]    # 50% power sensitivity

# Old approach (StandardScaler on all 14 features), random split
# τ here is the burst duration (2 × SIG_HALF_WIN)
sens_old_nue  = {'tau': [20, 200], 'sens': [91.2, 94.0]}
sens_old_mix  = {'tau': [200],     'sens': [84.8]}

# Features-only baseline (13D, no time), random split
sens_feat = {
    'NuE':   126.0,
    'NuMu':  None,    # not run standalone
    'NuTau': None,
    'NuMix': 126.9,
}

# Individual flavour results — old fixed-scale approach (τ=200s)
flavour_results_old = {
    'NuE':   94.0,
    'NuMu':  98.1,
    'NuTau': 82.1,
    'NuMix': 84.8,
}

# ── Plot ──────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(13, 5))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], wspace=0.35)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# ── Left: sensitivity vs signal duration ─────────────────────────────────────

# Model-matched (main result)
ax1.plot(tau_vals, sens_match, 'o-', color='steelblue', lw=2, ms=7,
         label='Model-matched time (14D, new)', zorder=3)

# Old fixed-scale approach for NuE
ax1.plot(sens_old_nue['tau'], sens_old_nue['sens'], 's--',
         color='cornflowerblue', lw=1.5, ms=7, alpha=0.8,
         label='Fixed-scale time, NuE (14D, old)')

# Old fixed-scale NuMix at τ=200s
ax1.plot(sens_old_mix['tau'], sens_old_mix['sens'], 'D--',
         color='slateblue', lw=1.5, ms=7, alpha=0.8,
         label='Fixed-scale time, NuMix (14D, old)')

# Features-only baseline
ax1.axhline(sens_feat['NuMix'], color='tomato', ls=':', lw=2,
            label=f'No time, NuMix (13D baseline = {sens_feat["NuMix"]:.0f})')
ax1.axhline(sens_feat['NuE'],   color='salmon',  ls=':', lw=1.5, alpha=0.7,
            label=f'No time, NuE   (13D baseline = {sens_feat["NuE"]:.0f})')

# Annotate model-matched points
for tau, s in zip(tau_vals, sens_match):
    ax1.annotate(f'{s:.0f}', xy=(tau, s), xytext=(6, 4),
                 textcoords='offset points', fontsize=9, color='steelblue')

ax1.set_xlabel('Signal duration  τ  (s)', fontsize=11)
ax1.set_ylabel('Signal events for 50% detection power', fontsize=11)
ax1.set_title('Sensitivity vs. signal duration  (NuMix 1:1:1)', fontsize=11)
ax1.set_xscale('log')
ax1.set_xlim(12, 600)
ax1.set_ylim(0, 160)
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xticks([20, 50, 100, 200, 400])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# ── Right: by-flavour comparison at τ=200s ────────────────────────────────────

flavours = ['NuE', 'NuMu', 'NuTau', 'NuMix']
colours  = ['#4c78a8', '#72b7b2', '#f58518', '#e45756']
x        = np.arange(len(flavours))
width    = 0.38

# Old fixed-scale (τ=200s)
bars_old = [flavour_results_old[fl] for fl in flavours]
# Model-matched at τ=200s: only NuMix available; show NaN for others
bars_new = [np.nan, np.nan, np.nan, 62.5]

b1 = ax2.bar(x - width/2, bars_old, width, label='Fixed-scale time (old)',
             color=[c + 'bb' for c in colours], edgecolor='grey', linewidth=0.5)
b2 = ax2.bar(x + width/2, bars_new, width, label='Model-matched τ=200s (new)',
             color=colours, edgecolor='grey', linewidth=0.5)

# Features-only baseline dots
feat_vals = [sens_feat['NuE'], None, None, sens_feat['NuMix']]
for xi, fv in zip(x, feat_vals):
    if fv is not None:
        ax2.plot(xi, fv, '_', color='tomato', ms=14, mew=2.5, zorder=5)

# Value labels
for bar in b1:
    h = bar.get_height()
    if not np.isnan(h):
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                 f'{h:.0f}', ha='center', va='bottom', fontsize=8)
for bar in b2:
    h = bar.get_height()
    if not np.isnan(h):
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                 f'{h:.0f}', ha='center', va='bottom', fontsize=8)

ax2.set_xticks(x)
ax2.set_xticklabels(flavours)
ax2.set_ylabel('Signal events for 50% detection power', fontsize=11)
ax2.set_title('By flavour  (τ = 200 s)', fontsize=11)
ax2.set_ylim(0, 160)
ax2.axhline(0, color='k', lw=0.5)
ax2.grid(True, axis='y', alpha=0.3)

# Legend: add marker for features-only baseline
from matplotlib.lines import Line2D
handles, labels = ax2.get_legend_handles_labels()
handles.append(Line2D([0], [0], marker='_', color='tomato',
                       markersize=10, markeredgewidth=2.5, linestyle='none'))
labels.append('No-time baseline (13D)')
ax2.legend(handles, labels, fontsize=8, loc='upper right')

fig.suptitle('EagleEye sensitivity  —  IceCube GW230808  '
             '(p_ext=1e-3, K_M=50, 200 null / 100 injection trials)',
             fontsize=10, y=1.01)

out = os.path.join(os.path.dirname(__file__), 'sensitivity_summary.pdf')
plt.savefig(out, bbox_inches='tight')
print(f"Saved to {out}")
