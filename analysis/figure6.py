import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Data (micro subset counts)
# ----------------------------
N = 158
n00, n01, n10, n11 = 38, 23, 10, 87
counts = np.array([[n00, n01],
                   [n10, n11]], dtype=float)
perc = counts / N * 100

# ----------------------------
# Figure / axes
# ----------------------------
fig, ax = plt.subplots(figsize=(9.4, 4.8))  # slightly wider for callouts

# Heatmap
im = ax.imshow(counts, cmap="Greys", vmin=0, vmax=counts.max(), aspect="equal")

# Axis ticks/labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["LLM=0", "LLM=1"], fontsize=12)
ax.set_yticklabels(["Clin=0", "Clin=1"], fontsize=12)

# Title + a bit more top space
ax.set_title(
    "Microbiology-evaluable subset: active coverage vs index organism (paired 2×2)",
    fontsize=14,
    pad=14,
    loc="left"
)

# Cell annotations (n + %)
mx = counts.max()
for i in range(2):
    for j in range(2):
        color = "white" if counts[i, j] > mx * 0.55 else "black"
        ax.text(
            j, i,
            f"{int(counts[i, j])}\n({perc[i, j]:.1f}%)",
            ha="center", va="center",
            fontsize=13, color=color
        )

# Grid lines around cells
ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
ax.grid(which="minor", color="black", linestyle="-", linewidth=1.2)
ax.tick_params(which="minor", bottom=False, left=False)

# Keep (0,0) at top-left (matrix-like)
ax.set_xlim(-0.5, 1.5)     # reserve space for callouts
ax.set_ylim(1.5, -0.5)

# ----------------------------
# Callouts (placed in axes coords => stable, no clipping)
# ----------------------------
discord_text = (
    "Discordant pairs\n"
    f"Clin=1 / LLM=0: {n10}\n"
    f"Clin=0 / LLM=1: {n01}\n"
    f"(N={N})"
)
stats_text = "Matched OR 2.24 (95% CI 1.08–4.63)\nExact McNemar p=0.0351"

# Use axes fraction coordinates so boxes stay in the right margin
ax.text(
    1.03, 0.62, discord_text,
    transform=ax.transAxes,
    ha="left", va="center",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black", linewidth=1.0),
    clip_on=False
)
ax.text(
    1.03, 0.88, stats_text,
    transform=ax.transAxes,
    ha="left", va="center",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black", linewidth=1.0),
    clip_on=False
)

# ----------------------------
# Layout: generous margins so nothing is cut
# ----------------------------
fig.subplots_adjust(left=0.12, right=0.62, top=0.86, bottom=0.14)

# Optional: colorbar (comment out if you don’t want it)
# cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# cbar.ax.tick_params(labelsize=10)

# ----------------------------
# Save (Windows-safe: write into current folder)
# ----------------------------
fig.subplots_adjust(left=0.12, right=0.62, top=0.86, bottom=0.14)

pdf = "Figure_micro_activity_2x2_heatmapstyle.pdf"
png = "Figure_micro_activity_2x2_heatmapstyle.png"
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, dpi=300, bbox_inches="tight")
plt.close(fig)

print(pdf, png)
