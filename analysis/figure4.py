import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Inputs
# -----------------------------
csv_path = "tableS_guardrail_context_violations.csv"

order = ["carb_violation", "aps_violation", "mrsa_violation"]
name_map = {
    "carb_violation": "Carbapenem contextual violation",
    "aps_violation": "Antipseudomonal contextual violation",
    "mrsa_violation": "Anti-MRSA contextual violation",
}

# -----------------------------
# Load + reshape
# -----------------------------
ctx = pd.read_csv(csv_path).set_index("component")

rows = []
for k in order:
    r = ctx.loc[k]
    rows.append({
        "label": name_map[k],
        "or_": float(r["matched_or"]),
        "lo": float(r["matched_or_ci_lo"]),
        "hi": float(r["matched_or_ci_hi"]),
        "n10": int(r["n10"]),
        "n01": int(r["n01"]),
        "p": float(r["p_mcnemar_exact"]),
        "N": int(r["n"]),
    })

df = pd.DataFrame(rows)

def p_fmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"

# -----------------------------
# Figure (PNG-optimized)
# -----------------------------
fig = plt.figure(figsize=(10.2, 4.0))
gs = fig.add_gridspec(
    1, 2,
    width_ratios=[1.0, 1.45],  # more room for the text panel
    wspace=0.04
)

ax = fig.add_subplot(gs[0, 0])
ax_txt = fig.add_subplot(gs[0, 1])
ax_txt.axis("off")

y = np.arange(len(df))[::-1]

for i, row in enumerate(df.itertuples(index=False)):
    yy = y[i]
    ax.plot([row.lo, row.hi], [yy, yy], lw=2, color="black")
    ax.plot(row.or_, yy, marker="s", ms=7, color="black")

ax.axvline(1.0, lw=1.5, color="black")
ax.set_xscale("log")
ax.set_ylim(-0.7, len(df) - 0.3)

ax.set_yticks(y)
ax.set_yticklabels(df["label"], fontsize=10)

ax.set_xlabel("Matched odds ratio (log scale)", fontsize=9)
ax.set_title(
    "Secondary contextual guardrail components (paired matched OR)",
    fontsize=11,
    fontweight="bold",
    loc="left"
)

xmin = min(df["lo"].min() * 0.8, 0.01)
xmax = max(df["hi"].max() * 1.2, 2.0)
ax.set_xlim(xmin, xmax)
ax.tick_params(axis="x", labelsize=9)

# -----------------------------
# Text panel
# -----------------------------
header = "OR (95% CI)              n10/n01    p"
ax_txt.text(
    0.02, 0.96, header,
    fontsize=9, fontweight="bold",
    va="top", family="monospace"
)

for i, row in enumerate(df.itertuples(index=False)):
    yy = 0.82 - i * 0.26
    or_ci = f"{row.or_:.3f} ({row.lo:.3f}–{row.hi:.3f})"
    disc = f"{row.n10}/{row.n01}"
    pval = p_fmt(row.p)
    line = f"{or_ci:<24} {disc:<8} {pval}"
    ax_txt.text(
        0.02, yy, line,
        fontsize=9, va="top",
        family="monospace"
    )

# -----------------------------
# Footnote
# -----------------------------
fig.text(
    0.02, 0.015,
    "n10 = Clin=1/AI=0; n01 = Clin=0/AI=1. Secondary endpoints: multiplicity caution.",
    fontsize=9, 
)

# -----------------------------
# Save (PNG only, no cropping)
# -----------------------------
fig.subplots_adjust(left=0.32, top=0.90, bottom=0.20)

png = "Figure_forest_or_secondaryonly_v3.png"
fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.18)
plt.close(fig)

png
