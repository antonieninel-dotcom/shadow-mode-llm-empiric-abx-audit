import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Primary endpoint 2×2 (paired): contextual guardrail violation
# Counts from locked outputs: n00=393 n01=16 n10=76 n11=8, N=493
# Layout standards:
# - Use fig, ax = plt.subplots()
# - Keep heatmap strictly 2×2 (no blank extra column)
# - Place callouts in axes-fraction coords (stable) + clip_on=False
# - Control margins via subplots_adjust (avoid tight_layout surprises)
# - Save to working directory (Windows-safe) with bbox_inches="tight"
# ------------------------------------------------------------

# Data
N = 493
n00, n01, n10, n11 = 393, 16, 76, 8
counts = np.array([[n00, n01],
                   [n10, n11]], dtype=float)
perc = counts / N * 100

# Figure
fig, ax = plt.subplots(figsize=(8.6, 4.6))  # wider to reserve right margin for callout

# Heatmap (grayscale)
mx = counts.max()
ax.imshow(counts, cmap="Greys", vmin=0, vmax=mx, aspect="equal")

# Ticks / labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["LLM=0", "LLM=1"], fontsize=12)
ax.set_yticklabels(["Clin=0", "Clin=1"], fontsize=12)

ax.set_title(
    "Primary endpoint: contextual guardrail violation (paired 2×2)",
    fontsize=14,
    fontweight="bold",
    loc="left",
    pad=14
)

# Cell annotations (n + %), with contrast-aware text color
for i in range(2):
    for j in range(2):
        cell = counts[i, j]
        txt_color = "white" if cell > mx * 0.55 else "black"
        ax.text(
            j, i,
            f"{int(cell)}\n({perc[i, j]:.1f}%)",
            ha="center", va="center",
            fontsize=13,
            color=txt_color
        )

# Grid lines
ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
ax.grid(which="minor", color="black", linestyle="-", linewidth=1.2)
ax.tick_params(which="minor", bottom=False, left=False)

# Keep only 2×2 visible (no blank space inside heatmap)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(1.5, -0.5)

# Callout: discordant pairs (stable placement)
discord_text = (
    "Discordant pairs\n"
    f"Clin=1 / LLM=0: {n10}\n"
    f"Clin=0 / LLM=1: {n01}\n"
    f"(N={N})"
)
ax.text(
    1.03, 0.58, discord_text,
    transform=ax.transAxes,
    ha="left", va="center",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black", linewidth=1.0),
    clip_on=False
)

# Margins: reserve right-side space for callout + avoid clipping
fig.subplots_adjust(left=0.12, right=0.62, top=0.86, bottom=0.14)

# Save (current folder; bbox_inches captures callout)
pdf_path = "Figure_primary_2x2_heatmap.pdf"
png_path = "Figure_primary_2x2_heatmap.png"
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(pdf_path, png_path)
