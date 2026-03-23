"""Generate figures for the paper from results.json."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

with open("results.json") as f:
    data = json.load(f)

ar_steps = [h[0] for h in data["ar"]["history"]]
ar_vals = [h[1] for h in data["ar"]["history"]]
mdlm_steps = [h[0] for h in data["mdlm"]["history"]]
mdlm_vals = [h[1] for h in data["mdlm"]["history"]]

# ── Figure 1: Dual-axis loss curves ────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(7, 4))

color_ar = '#2563eb'
color_mdlm = '#dc2626'

ax1.set_xlabel('Training Step', fontsize=12)
ax1.set_ylabel('AR Validation Loss (next-token)', color=color_ar, fontsize=11)
ln1 = ax1.plot(ar_steps, ar_vals, 'o-', color=color_ar, linewidth=2,
               markersize=5, label='AR (123.6M)')
ax1.tick_params(axis='y', labelcolor=color_ar)
ax1.set_ylim(1.4, 2.8)

# Mark AR best
best_ar_idx = ar_vals.index(min(ar_vals))
ax1.annotate(f'Best: {min(ar_vals):.3f}\n(step {ar_steps[best_ar_idx]:,})',
             xy=(ar_steps[best_ar_idx], min(ar_vals)),
             xytext=(ar_steps[best_ar_idx] + 2000, min(ar_vals) + 0.25),
             arrowprops=dict(arrowstyle='->', color=color_ar, lw=1.5),
             color=color_ar, fontsize=9)

ax2 = ax1.twinx()
ax2.set_ylabel('MDLM Validation Loss (masked-token)', color=color_mdlm, fontsize=11)
ln2 = ax2.plot(mdlm_steps, mdlm_vals, 's-', color=color_mdlm, linewidth=2,
               markersize=5, label='MDLM (162.7M)')
ax2.tick_params(axis='y', labelcolor=color_mdlm)
ax2.set_ylim(3.3, 4.1)

# Mark MDLM best
best_mdlm_idx = mdlm_vals.index(min(mdlm_vals))
ax2.annotate(f'Best: {min(mdlm_vals):.3f}\n(step {mdlm_steps[best_mdlm_idx]:,})',
             xy=(mdlm_steps[best_mdlm_idx], min(mdlm_vals)),
             xytext=(mdlm_steps[best_mdlm_idx] - 7000, min(mdlm_vals) + 0.2),
             arrowprops=dict(arrowstyle='->', color=color_mdlm, lw=1.5),
             color=color_mdlm, fontsize=9)

# Mark overfitting region
ax1.axvspan(14000, 20000, alpha=0.08, color=color_ar, label='AR overfitting region')

# Combined legend
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper right', fontsize=10, framealpha=0.9)

ax1.set_xlim(1000, 21000)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
ax1.grid(True, alpha=0.3)

plt.title('Validation Loss: AR vs. MDLM (H100, 20K steps)', fontsize=13, pad=12)
fig.tight_layout()
plt.savefig('fig_loss_curves.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_loss_curves.png', dpi=200, bbox_inches='tight')
print("Saved fig_loss_curves.pdf and .png")

# ── Figure 2: Throughput / time bar chart ──────────────────────────────
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(8, 3.5))

models = ['AR\n(123.6M)', 'MDLM\n(162.7M)']
colors = [color_ar, color_mdlm]

# Throughput
throughputs = [data["ar"]["tok_s"], data["mdlm"]["tok_s"]]
bars1 = ax3.bar(models, throughputs, color=colors, width=0.5, edgecolor='white')
ax3.set_ylabel('Tokens / second', fontsize=11)
ax3.set_title('Training Throughput', fontsize=12)
ax3.set_ylim(0, 58000)
for bar, val in zip(bars1, throughputs):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 800,
             f'{val:,.0f}', ha='center', fontsize=10, fontweight='bold')
ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
ax3.grid(axis='y', alpha=0.3)

# Time
times = [data["ar"]["time_min"], data["mdlm"]["time_min"]]
bars2 = ax4.bar(models, times, color=colors, width=0.5, edgecolor='white')
ax4.set_ylabel('Minutes', fontsize=11)
ax4.set_title('Total Training Time', fontsize=12)
ax4.set_ylim(0, 130)
for bar, val in zip(bars2, times):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 2,
             f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

fig2.tight_layout(w_pad=3)
plt.savefig('fig_throughput.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_throughput.png', dpi=200, bbox_inches='tight')
print("Saved fig_throughput.pdf and .png")
