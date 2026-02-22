import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Locked numbers (Table 2)
# -----------------------------
N = 493
lower = 79
tie = 398
higher = 16
non_ties = lower + higher

median = 0
q1 = 0
q3 = 0
mean = -0.219067
sd = 0.789078

# exact p-values (update if needed)
p_wilcoxon = 9.48e-11
p_sign = 3.47e-11

FS_TITLE = 11
FS_PANEL = 10
FS_ANN = 9
FS_BOX = 9
FS_NOTE = 8

# ---- helper: p formatting as mantissa×10^exp with unicode superscripts
_SUP = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")
def fmt_p(p: float, sig: int = 2) -> str:
    if p == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(p))))
    mant = p / (10 ** exp)
    return f"{mant:.{sig}f}×10{str(exp).translate(_SUP)}"

fig = plt.figure(figsize=(7.8, 4.2))
gs = fig.add_gridspec(
    2, 2,
    height_ratios=[1, 1],
    width_ratios=[1.35, 1],
    hspace=0.35,
    wspace=0.28
)

# =========================
# Panel A: 100% stacked bar
# =========================
axA = fig.add_subplot(gs[:, 0])
vals = np.array([lower, tie, higher], dtype=float)
perc = vals / N * 100.0
left = 0.0
labels = ["Δ<0 (lower in LLM)", "Δ=0 (tie)", "Δ>0 (higher in LLM)"]
shades = ["0.25", "0.70", "0.45"]  # grayscale

for idx, (v, p, shade) in enumerate(zip(vals, perc, shades)):
    axA.barh([0], [p], left=[left], height=0.52,
             color=shade, edgecolor="black", linewidth=0.8)

    # Always label inside; nudge inward at edges to avoid clipping
    x = left + p / 2.0
    pad = 1.3
    if idx == 0:
        x = max(x, left + pad)
    if idx == len(vals) - 1:
        pad_last = 3.0
        x = left + p - pad_last  # push to the left inside the tiny right segment

    ha = "center"
    if idx == len(vals) - 1:
        ha = "right"
    else:
        ha = "center"

    fs = FS_ANN if p >= 10 else FS_ANN - 1

    axA.text(
        x, 0,
        f"{int(v)}\n({p:.1f}%)",
        ha=ha, va="center",
        fontsize=fs, fontweight="bold",
        color="white" if shade == "0.25" else "black",
        clip_on=True
    )

    left += p

axA.set_xlim(0, 100)
axA.set_yticks([])
axA.set_xlabel("Proportion of admissions (%)", fontsize=FS_PANEL, labelpad=2)
axA.set_title("A. All admissions (N = 493)", loc="left",
              fontsize=FS_TITLE, fontweight="bold")

# =========================
# Panel B: Non-ties only
# =========================
axB = fig.add_subplot(gs[0, 1])
axB.bar(["Lower\n(Δ<0)", "Higher\n(Δ>0)"], [lower, higher],
        color=["0.25", "0.45"], edgecolor="black", linewidth=0.8)
axB.set_title(f"B. Non-tied pairs only (n = {non_ties})", loc="left",
              fontsize=FS_TITLE, fontweight="bold")
axB.set_ylabel("Admissions (n)", fontsize=FS_PANEL)

# Extra headroom for labels
ymax = max(lower, higher) * 1.60
axB.set_ylim(0, ymax)

for i, v in enumerate([lower, higher]):
    axB.text(
        i, v + ymax * 0.05,
        f"{v}\n({v / non_ties * 100:.1f}%)",
        ha="center", va="bottom",
        fontsize=FS_ANN, fontweight="bold"
    )
axB.tick_params(axis="x", labelsize=FS_PANEL)
axB.tick_params(axis="y", labelsize=FS_PANEL)

# =========================
# Panel C: Summary box
# =========================
axC = fig.add_subplot(gs[1, 1])
axC.axis("off")

summary = (
    "C. Summary\n"
    "Δ contextual guardrail penalty\n"
    "(LLM − clinician)\n\n"
    f"Median (IQR): {median} ({q1}–{q3})\n"
    "Median 95% CI: 0–0\n"
    f"Mean (SD): {mean:.3f} ({sd:.3f})\n"
    f"Ties: {tie}/{N} ({tie / N * 100:.1f}%)\n"
    f"Wilcoxon signed-rank: p = {fmt_p(p_wilcoxon)}\n"
    f"Sign test (non-ties): p = {fmt_p(p_sign)}"
)

axC.text(
    0.0, 1.0, summary,
    ha="left", va="top", fontsize=FS_BOX,
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", linewidth=0.8)
)

# =========================
# Title + footnotes (FIG-LEVEL, not axis-level)
# =========================
fig.suptitle("Figure 3. Key secondary confirmatory endpoint",
             y=0.98, fontsize=FS_TITLE + 1, fontweight="bold")

# Footnotes placed at figure level to avoid overlap with Panel A axis
fig.text(0.06, 0.055, "Δ<0 lower in LLM   |   Δ=0 tie   |   Δ>0 higher in LLM",
         ha="left", va="center", fontsize=FS_NOTE)
fig.text(0.06, 0.028, "Negative Δ indicates a lower contextual guardrail penalty in the LLM arm.",
         ha="left", va="center", fontsize=FS_NOTE)

# IMPORTANT: explicit margins (no tight_layout)
fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.22)

fig.savefig("Figure3_keysecondary_cleartriptych_final.pdf")
fig.savefig("Figure3_keysecondary_cleartriptych_final.png", dpi=300)
plt.close(fig)

print("Wrote Figure3_keysecondary_cleartriptych_final.pdf/.png")