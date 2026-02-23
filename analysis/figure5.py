#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ----------------------------
# CONFIG
# ----------------------------

# CSV expected at: ../table2_secondary_deltas.csv relative to this script
BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "table2_secondary_deltas.csv"

ENDPOINT_24 = "delta_empiric_cost_24h_eur_median_prices"
ENDPOINT_72 = "delta_empiric_cost_72h_eur_median_prices_true"

OUT_PNG = Path("Figure_cost_deltas_v4.png")
OUT_PDF = Path("Figure_cost_deltas_v4.pdf")


# ----------------------------
# Helpers
# ----------------------------

def norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s


def require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")


def get_row(df: pd.DataFrame, endpoint: str) -> pd.Series:
    if "endpoint" not in df.columns:
        raise KeyError(f"No 'endpoint' column. Available: {list(df.columns)}")
    target = norm(endpoint)
    hit = df.loc[df["endpoint"].map(norm) == target]
    if hit.empty:
        print("\n[DEBUG] Endpoint not found:", endpoint)
        print("[DEBUG] Available endpoints:")
        for x in df["endpoint"].astype(str).tolist():
            print(" -", x)
        raise ValueError(f"Endpoint not found: {endpoint}")
    return hit.iloc[0]


def _superscript_int(n: int) -> str:
    sup = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")
    return str(n).translate(sup)


def p_sci(p, sig=2) -> str:
    """Always scientific notation, e.g. 6.61×10⁻³, 2.80×10⁻¹³."""
    if p is None:
        return "NA"
    p = float(p)
    if pd.isna(p):
        return "NA"
    if p == 0:
        return "<1×10" + _superscript_int(-300)
    s = f"{p:.{sig}e}"  # e.g. '2.80e-13'
    mant, exp = s.split("e")
    return f"{mant}×10{_superscript_int(int(exp))}"


def add_iqr_box(ax, y, median, q1, q3, height=0.32):
    """Draw an IQR box with median line at y."""
    rect = plt.Rectangle(
        (q1, y - height / 2),
        q3 - q1,
        height,
        facecolor="0.87",
        edgecolor="black",
        linewidth=1,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.plot([median, median], [y - height / 2, y + height / 2], color="black", linewidth=2, zorder=3)


def label_median(ax, x, y, text):
    """Median label with small white background to avoid visual collisions."""
    x = float(x)
    dx = 0.7 if abs(x) < 1.5 else 0.0
    ax.text(
        x + dx, y, text,
        ha="center", va="center", fontsize=9,
        bbox=dict(fc="white", ec="none", pad=0.2),
        zorder=4
    )


def format_block(row: pd.Series, title: str, include_mean_sd: bool = False) -> str:
    """Text block for one endpoint (24h / 72h)."""
    lines = []
    lines.append(f"{title} (N={int(row['n'])})")
    lines.append(
        f"Median (IQR): {float(row['median']):.2f} "
        f"({float(row['q1']):.2f} to {float(row['q3']):.2f})"
    )
    lines.append(
        f"Δ<0: {int(row['improved_n'])}   "
        f"Δ=0: {int(row['tied_n'])}   "
        f"Δ>0: {int(row['worsened_n'])}"
    )

    # p-values
    line_p = f"Wilcoxon p={p_sci(row['p_value'])}"
    if not pd.isna(row.get("p_sign")):
        line_p += f"; Sign p={p_sci(row['p_sign'])}"
    lines.append(line_p)

    # bootstrap CI (median)
    if (not pd.isna(row.get("median_ci_low"))) and (not pd.isna(row.get("median_ci_high"))):
        lines.append(
            f"Median 95% CI (bootstrap): "
            f"{float(row['median_ci_low']):.2f} to {float(row['median_ci_high']):.2f}"
        )

    # optional mean/sd
    if include_mean_sd and (not pd.isna(row.get("mean"))) and (not pd.isna(row.get("sd"))):
        lines.append(f"Mean (SD): {float(row['mean']):.2f} ({float(row['sd']):.2f})")

    return "\n".join(lines)


# ----------------------------
# Load data
# ----------------------------

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Missing file: {CSV_PATH}")

t2d = pd.read_csv(CSV_PATH)

need = ["endpoint", "n", "median", "q1", "q3", "improved_n", "worsened_n", "tied_n", "p_value"]
require_cols(t2d, need)

# Optional columns (won't fail if absent)
opt = ["p_sign", "median_ci_low", "median_ci_high", "mean", "sd"]
for c in opt:
    if c not in t2d.columns:
        t2d[c] = pd.NA

row24 = get_row(t2d, ENDPOINT_24)
row72 = get_row(t2d, ENDPOINT_72)


# ----------------------------
# Axis limits (IQR range + padding; ensure 0 visible)
# ----------------------------

mins = min(float(row24["q1"]), float(row72["q1"]))
maxs = max(float(row24["q3"]), float(row72["q3"]))
span = maxs - mins
pad = span * 0.12 if span > 0 else 1.0

xmin = min(mins - pad, 0 - pad)
xmax = max(maxs + pad, 0 + pad)


# ----------------------------
# Figure (two-panel layout)
# ----------------------------

# Compact, journal-friendly
fig = plt.figure(figsize=(11.0, 4.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.15], wspace=0.12)

ax = fig.add_subplot(gs[0, 0])
axr = fig.add_subplot(gs[0, 1])
axr.axis("off")

# Title
fig.suptitle(
    "Secondary endpoint: paired cost deltas (summary)",
    fontsize=15,
    fontweight="bold",
    y=0.98
)

# Main axis styling
ax.set_xlim(xmin, xmax)
ax.set_ylim(-0.6, 1.6)

# Δ=0 reference line
ax.axvline(x=0, color="black", linewidth=1.5, zorder=1)

# y labels
ax.set_yticks([1, 0])
ax.set_yticklabels(
    [f"24h (N={int(row24['n'])})", f"72h continued-therapy subset (N={int(row72['n'])})"],
    fontsize=11,
)

ax.tick_params(axis="x", labelsize=10)
ax.set_xlabel("Δ empiric antibiotic cost (EUR; LLM − Clinician)", fontsize=12, labelpad=18)

# Draw IQR boxes + median lines
add_iqr_box(ax, 1, float(row24["median"]), float(row24["q1"]), float(row24["q3"]))
add_iqr_box(ax, 0, float(row72["median"]), float(row72["q1"]), float(row72["q3"]))

# Median labels
label_median(ax, row24["median"], 1.28, f"Median {float(row24['median']):.2f} EUR")
label_median(ax, row72["median"], 0.28, f"Median {float(row72['median']):.2f} EUR")

# Right text box (wrapped to avoid ultra-wide export)
block24 = format_block(row24, "24h", include_mean_sd=True)
block72 = format_block(row72, "72h continued-therapy subset", include_mean_sd=False)

txt = (
    block24
    + "\n\n"
    + block72
    + "\n\n"
    + "Interpretation: negative Δ indicates lower cost\n"
    "in the LLM arm (process-level comparison)."
)

# Wrap only long lines; keep existing newlines
wrapped = "\n".join(textwrap.fill(line, width=64) for line in txt.splitlines())

axr.text(
    0.0, 1.0, wrapped,
    ha="left", va="top",
    fontsize=10.5,
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", linewidth=0.9),
)

# Footer (small)
fig.text(
    0.24, 0.04,
    "Boxes indicate the IQR; vertical line marks Δ=0.",
    fontsize=9,
    ha="left",
    va="bottom"
)

# Margins: enough room for long y-labels, but compact overall
fig.subplots_adjust(left=0.24, right=0.98, bottom=0.22, top=0.88)

# Save

# --- force draw so renderer knows text extents ---
fig.canvas.draw()

# --- auto-expand left margin to fit y-tick labels ---
renderer = fig.canvas.get_renderer()
bbox = ax.get_tightbbox(renderer)

# convert bbox width from pixels to figure fraction
left_margin = bbox.x0 / fig.bbox.width
fig.subplots_adjust(left=max(left_margin - 0.02, 0.30))

fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=300)
plt.close(fig)

print(f"[OK] Read from: {CSV_PATH}")
print(f"[SAVED] {OUT_PDF}")
print(f"[SAVED] {OUT_PNG}")