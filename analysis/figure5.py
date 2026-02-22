import matplotlib.pyplot as plt

import pandas as pd

from pathlib import Path

import re



# ----------------------------

# 1) PATH (Q1-safe)

# ----------------------------

# Structure assumed:

# tabele/

#   table2_secondary_deltas.csv

#   figure 5-6/

#       test.py   <- this script



from pathlib import Path



BASE_DIR = Path(__file__).resolve().parents[1]



csv_path = BASE_DIR / "table2_secondary_deltas.csv"



# If you instead keep the CSV inside "figure 5-6", use:

# csv_path = Path(__file__).resolve().parent / "table2_secondary_deltas.csv"



# ----------------------------

# 2) Endpoints

# ----------------------------

ENDPOINT_24 = "delta_empiric_cost_24h_eur_median_prices"

ENDPOINT_72 = "delta_empiric_cost_72h_eur_median_prices_true"  # exists in your CSV



out_png = Path("Figure_cost_deltas_v3.png")

out_pdf = Path("Figure_cost_deltas_v3.pdf")





def p_classic(p):

    p = float(p)

    return "<0.001" if p < 0.001 else f"{p:.3f}"





def norm(s: str) -> str:

    s = "" if s is None else str(s)

    s = s.strip().lower()

    s = re.sub(r"\s+", "_", s)

    return s





def require_cols(df, cols):

    missing = [c for c in cols if c not in df.columns]

    if missing:

        raise KeyError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")





def get_row(df, endpoint: str):

    target = norm(endpoint)

    if "endpoint" not in df.columns:

        raise KeyError(f"No 'endpoint' column. Available: {list(df.columns)}")

    s_norm = df["endpoint"].map(norm)

    hit = df.loc[s_norm == target]

    if hit.empty:

        # print helpful debug

        print("\n[DEBUG] Endpoint not found:", endpoint)

        print("[DEBUG] Available endpoints:")

        for x in df["endpoint"].astype(str).tolist():

            print(" -", x)

        raise ValueError(f"Endpoint not found: {endpoint}")

    return hit.iloc[0]





def add_iqr_box(ax, y, median, q1, q3, height=0.32):

    rect = plt.Rectangle((q1, y - height / 2), q3 - q1, height,

                         facecolor="0.85", edgecolor="black", linewidth=1)

    ax.add_patch(rect)

    ax.plot([median, median], [y - height / 2, y + height / 2], color="black", linewidth=2)

    ax.plot([q1, q1], [y - height / 2, y + height / 2], color="black", linewidth=1)

    ax.plot([q3, q3], [y - height / 2, y + height / 2], color="black", linewidth=1)





# ----------------------------

# Load + validate

# ----------------------------

if not csv_path.exists():

    raise FileNotFoundError(f"Missing file: {csv_path}")



t2d = pd.read_csv(csv_path)



need = ["endpoint", "n", "median", "q1", "q3", "improved_n", "worsened_n", "tied_n", "p_value"]

require_cols(t2d, need)



row24 = get_row(t2d, ENDPOINT_24)

row72 = get_row(t2d, ENDPOINT_72)



# ----------------------------

# Axis limits (IQR + padding; ensure 0 visible)

# ----------------------------

mins = min(float(row24["q1"]), float(row72["q1"]))

maxs = max(float(row24["q3"]), float(row72["q3"]))

pad = (maxs - mins) * 0.12 if maxs > mins else 1.0

xmin = float(min(mins - pad, 0 - pad))

xmax = float(max(maxs + pad, 0 + pad))



# ----------------------------

# Plot

# ----------------------------

fig = plt.figure(figsize=(10, 4.2))

gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.15)

ax = fig.add_subplot(gs[0, 0])

axr = fig.add_subplot(gs[0, 1])

axr.axis("off")



ax.set_xlim(xmin, xmax)

ax.set_ylim(-0.6, 1.6)

ax.axvline(x=0, ymin=-0.15, ymax=1.15, color="black", linewidth=1.5, zorder=0)



ax.set_yticks([1, 0])

ax.set_yticklabels(

    [f"24h (N={int(row24['n'])})", f"72h continued-therapy subset (N={int(row72['n'])})"],

    fontsize=10,

)

ax.set_xlabel("Δ empiric antibiotic cost (EUR; AI − Clinician)", fontsize=10)

ax.set_title(

    "Secondary endpoint: paired cost deltas (summary)",

    fontsize=11,

    fontweight="bold",

    loc="left",

    pad=22

)

ax.tick_params(axis="x", labelsize=9)



add_iqr_box(ax, 1, float(row24["median"]), float(row24["q1"]), float(row24["q3"]))

add_iqr_box(ax, 0, float(row72["median"]), float(row72["q1"]), float(row72["q3"]))



def median_label(ax, x, y, text):

    # offset slightly away from 0 line, plus white background

    x = float(x)

    dx = 0.6 if abs(x) < 1.5 else 0.0  # push if near Δ=0

    ax.text(

        x + dx, y, text,

        ha="center", va="center", fontsize=9,

        bbox=dict(fc="white", ec="none", pad=0.2)

    )



median_label(ax, row24["median"], 1.28, f"Median {float(row24['median']):.2f} EUR")

median_label(ax, row72["median"], 0.28, f"Median {float(row72['median']):.2f} EUR")



txt = (

    f"24h (N={int(row24['n'])})\n"

    f"Median (IQR): {float(row24['median']):.2f} ({float(row24['q1']):.2f} to {float(row24['q3']):.2f})\n"

    f"Δ<0: {int(row24['improved_n'])}   Δ=0: {int(row24['tied_n'])}   Δ>0: {int(row24['worsened_n'])}\n"

    f"Wilcoxon: {p_classic(row24['p_value'])}\n\n"

    f"72h true subset (N={int(row72['n'])})\n"

    f"Median (IQR): {float(row72['median']):.2f} ({float(row72['q1']):.2f} to {float(row72['q3']):.2f})\n"

    f"Δ<0: {int(row72['improved_n'])}   Δ=0: {int(row72['tied_n'])}   Δ>0: {int(row72['worsened_n'])}\n"

    f"Wilcoxon: {p_classic(row72['p_value'])}\n\n"

    "Interpretation: negative Δ indicates lower cost\nin the AI arm (process-level comparison)."

)

axr.text(

    0.0, 1.0, txt,

    ha="left", va="top", fontsize=9,

    bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black", linewidth=0.8)

)



# ---- spacing / margins (prevents left clipping + bottom overlap) ----

fig.subplots_adjust(

    left=0.35,   # more space for long y tick labels

    right=0.92,

    top=0.85,

    bottom=0.20, # more space for xlabel + footer

    wspace=0.20

)



# give xlabel some breathing room above footer

ax.set_xlabel("Δ empiric antibiotic cost (EUR; AI − Clinician)", fontsize=10, labelpad=12)



# footer moved lower

fig.text(

    0.35,  # align with left margin

    0.04,  # low enough to not collide with xlabel

    "Boxes indicate the IQR; vertical line marks Δ=0.",

    fontsize=8,

    ha="left",

    va="bottom"

)



fig.savefig(out_pdf)

fig.savefig(out_png, dpi=300)

plt.close(fig)



print(f"[OK] Read from: {csv_path}")

print(f"[SAVED] {out_pdf}")

print(f"[SAVED] {out_png}")