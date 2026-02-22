#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sap_runner.py

Hardened analysis runner for a single-center retrospective *paired* comparison
(Clinician vs AI empiric regimen within first 24h).

Primary endpoint (ID/AMS-friendly):
  - Any contextual guardrail violation (binary; paired McNemar)
    = any of: carbapenem-when-not-justified OR antipseudomonal-when-not-justified OR anti-MRSA-when-not-justified.

Key secondary endpoint:
  - Δ contextual guardrail penalty score (AI − Clinician), weights: carb=2, APS=1, anti-MRSA=1.

Expected workbook: --xlsx data.xlsx with sheets:
  - data
  - endpoints
  - optional: abx_codes (for AWaRe mapping)

Run:
  python sap_runner.py --xlsx data.xlsx --out outputs --models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import scipy  # for scipy.__version__

try:
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError
except Exception:
    sm = None

from numpy.linalg import LinAlgError


# ----------------------------- small utils -----------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_df_stable(df: pd.DataFrame) -> str:
    # Stable-ish hash: deterministic column order + csv serialization.
    b = df.sort_index(axis=1).to_csv(index=False).encode("utf-8")
    return sha256_bytes(b)


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def as_str(s: pd.Series) -> pd.Series:
    return s.astype("string")


def holm_adjust(pvals: List[float]) -> List[float]:
    p = np.asarray([1.0 if (v is None or np.isnan(v)) else float(v) for v in pvals], dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for k, idx in enumerate(order):
        val = (m - k) * p[idx]
        val = min(max(val, prev), 1.0)
        adj[idx] = val
        prev = val
    return adj.tolist()


def bootstrap_ci(x: np.ndarray, stat: str, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    n = x.size
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samp = rng.choice(x, size=n, replace=True)
        vals[i] = np.nanmedian(samp) if stat == "median" else np.nanmean(samp)
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def wilcoxon_pratt(diff: np.ndarray) -> Tuple[float, float, int]:
    d = diff[~np.isnan(diff)]
    if d.size == 0:
        return (np.nan, np.nan, 0)
    try:
        res = stats.wilcoxon(d, zero_method="pratt", alternative="two-sided", method="auto")
        return float(res.statistic), float(res.pvalue), int(d.size)
    except TypeError:
        res = stats.wilcoxon(d, zero_method="pratt", alternative="two-sided")
        return float(res.statistic), float(res.pvalue), int(d.size)
    except Exception:
        return (0.0, 1.0, int(d.size))


def sign_test_non_ties(diff: np.ndarray) -> Dict[str, Any]:
    """Two-sided binomial sign test on non-ties.
    Convention: for deltas (AI-Clin), negative = favorable for AI.
    """
    d = diff[~np.isnan(diff)]
    pos = int((d > 0).sum())
    neg = int((d < 0).sum())
    n_nt = pos + neg
    p = 1.0 if n_nt == 0 else float(stats.binomtest(k=min(pos, neg), n=n_nt, p=0.5).pvalue)
    fav = neg
    if n_nt > 0:
        ci = stats.binomtest(k=fav, n=n_nt, p=0.5).proportion_ci(confidence_level=0.95, method="exact")
        fav_prop = fav / n_nt
        ci_lo, ci_hi = float(ci.low), float(ci.high)
    else:
        fav_prop, ci_lo, ci_hi = np.nan, np.nan, np.nan
    return {
        "sign_pos_n": pos, "sign_neg_n": neg, "sign_n_nontie": n_nt, "sign_p_value": p,
        "fav_prop_among_nonties": float(fav_prop) if n_nt else np.nan,
        "fav_prop_ci_lo": ci_lo, "fav_prop_ci_hi": ci_hi
    }


def rank_biserial_from_diffs(diff: np.ndarray) -> float:
    d = diff[~np.isnan(diff)]
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    t_pos = ranks[d > 0].sum()
    t_neg = ranks[d < 0].sum()
    return float((t_pos - t_neg) / (t_pos + t_neg))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = a.astype(int); b = b.astype(int)
    n00 = int(((a == 0) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n11 = int(((a == 1) & (b == 1)).sum())
    n = n00 + n01 + n10 + n11

    rd = (n01 - n10) / n if n else np.nan
    cc = 0.5
    matched_or = (n01 + cc) / (n10 + cc)
    se_log_or = np.sqrt(1.0 / (n01 + cc) + 1.0 / (n10 + cc))
    ci_lo = float(np.exp(np.log(matched_or) - 1.96 * se_log_or))
    ci_hi = float(np.exp(np.log(matched_or) + 1.96 * se_log_or))

    discord = n01 + n10
    p = 1.0 if discord == 0 else float(stats.binomtest(k=min(n01, n10), n=discord, p=0.5).pvalue)

    return {
        "n": float(n), "n00": float(n00), "n01": float(n01), "n10": float(n10), "n11": float(n11),
        "rd": float(rd),
        "matched_or": float(matched_or), "matched_or_ci_lo": ci_lo, "matched_or_ci_hi": ci_hi,
        "p_mcnemar_exact": float(p),
    }


# ----------------------------- config -----------------------------

@dataclass
class RunConfig:
    xlsx: Path
    out: Path

    # Primary (binary paired) columns (created by add_guardrail_composites)
    primary_binary_clin_col: str
    primary_binary_ai_col: str

    # Key secondary (delta) column (created by compute_contextual_guardrails)
    key_secondary_delta_col: str

    # Which delta column to use for histogram + exploratory predictor modeling:
    delta_for_hist_col: str
    delta_for_models_col: str

    n_boot: int
    seed: int
    models: bool
    min_subgroup_n: int

    mdr_text_col: str
    clin_cov_col: str
    ai_cov_col: str

    admitting_ward_col: str
    icu_transfer_24h_col: str
    fail_on_icu_mismatch: bool


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)

    p.add_argument("--primary-binary-clin-col", default="clin_any_violation")
    p.add_argument("--primary-binary-ai-col", default="ai_any_violation")

    p.add_argument("--key-secondary-delta-col", default="delta_guardrail_penalty_context")
    p.add_argument("--delta-for-hist-col", default=None)
    p.add_argument("--delta-for-models-col", default=None)

    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--models", action="store_true")
    p.add_argument("--min-subgroup-n", type=int, default=30)

    p.add_argument("--mdr-text-col", default="mdr_risk_text")
    p.add_argument("--clin-cov-col", default="clin_empiric_active_vs_index")
    p.add_argument("--ai-cov-col", default="ai_empiric_active_vs_index")

    p.add_argument("--admitting-ward-col", default="admitting_ward")
    p.add_argument("--icu-transfer-24h-col", default="icu_transfer_24h")
    p.add_argument("--fail-on-icu-mismatch", action="store_true")
    a = p.parse_args()

    key_delta = a.key_secondary_delta_col
    delta_hist = a.delta_for_hist_col if a.delta_for_hist_col else key_delta
    delta_models = a.delta_for_models_col if a.delta_for_models_col else key_delta

    return RunConfig(
        xlsx=a.xlsx,
        out=a.out,
        primary_binary_clin_col=a.primary_binary_clin_col,
        primary_binary_ai_col=a.primary_binary_ai_col,
        key_secondary_delta_col=key_delta,
        delta_for_hist_col=delta_hist,
        delta_for_models_col=delta_models,
        n_boot=a.n_boot,
        seed=a.seed,
        models=bool(a.models),
        min_subgroup_n=a.min_subgroup_n,
        mdr_text_col=a.mdr_text_col,
        clin_cov_col=a.clin_cov_col,
        ai_cov_col=a.ai_cov_col,
        admitting_ward_col=a.admitting_ward_col,
        icu_transfer_24h_col=a.icu_transfer_24h_col,
        fail_on_icu_mismatch=bool(a.fail_on_icu_mismatch),
    )


# ----------------------------- IO & merge policy -----------------------------

def load_workbook(xlsx: Path) -> Dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(xlsx)
    return {name: pd.read_excel(xlsx, sheet_name=name) for name in xl.sheet_names}


def ensure_crt_id(df: pd.DataFrame) -> pd.DataFrame:
    if "crt_id" not in df.columns:
        raise ValueError("Missing required column: crt_id")
    out = df.copy()
    out["crt_id"] = pd.to_numeric(out["crt_id"], errors="coerce").astype("Int64")
    return out


def merge_endpoints_wins(data: pd.DataFrame, endpoints: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    data = ensure_crt_id(data)
    endpoints = ensure_crt_id(endpoints)
    common = sorted(list((set(data.columns).intersection(set(endpoints.columns))) - {"crt_id"}))
    if common:
        data = data.drop(columns=common)
    merged = data.merge(endpoints, on="crt_id", how="inner", validate="one_to_one")
    return merged, common


def flow_counts(data: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    n_full = int(data["crt_id"].nunique())
    n_analysis = int(merged["crt_id"].nunique())
    return pd.DataFrame([
        ("full_rows_data_sheet", int(len(data))),
        ("full_unique_admissions", n_full),
        ("analysis_rows_after_merge", int(len(merged))),
        ("analysis_unique_admissions", n_analysis),
        ("excluded_no_matching_endpoints", int(n_full - n_analysis)),
    ], columns=["metric", "value"])


# ----------------------------- concordance -----------------------------

REGIMEN_COLS_CLIN = ["clin_abx_code_1", "clin_abx_code_2", "clin_abx_code_3"]
REGIMEN_COLS_AI   = ["ai_abx_code_1", "ai_abx_code_2", "ai_abx_code_3"]
EXCLUDE_TOKENS = {"", "NA", "N/A", "NONE", "NULL", "NAN"}  # NOTE: we handle NO_ANTIBIOTIC separately

def _clean_code(v) -> Optional[str]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    s = str(v).strip()
    if not s:
        return None
    su = s.upper()
    if su in EXCLUDE_TOKENS:
        return None
    # Keep NO_ANTIBIOTIC as a special sentinel
    return su


def regimen_set(row: pd.Series, cols: List[str]) -> Set[str]:
    out: Set[str] = set()
    for c in cols:
        if c in row.index:
            cv = _clean_code(row[c])
            if cv and cv != "NO_ANTIBIOTIC":
                out.add(cv)
    return out


def add_concordance(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in (REGIMEN_COLS_CLIN + REGIMEN_COLS_AI) if c not in df.columns]
    if missing:
        return df

    overlap_n, identical_set, identical_primary, discordant, label = [], [], [], [], []
    for _, r in df.iterrows():
        s_ai = regimen_set(r, REGIMEN_COLS_AI)
        s_cl = regimen_set(r, REGIMEN_COLS_CLIN)
        both_empty = (len(s_ai) == 0 and len(s_cl) == 0)

        ov = int(len(s_ai.intersection(s_cl))) if len(s_ai) else 0

        if both_empty:
            idset = 1
        else:
            idset = int((len(s_ai) > 0 or len(s_cl) > 0) and (s_ai == s_cl))

        c1 = _clean_code(r.get("clin_abx_code_1", None))
        a1 = _clean_code(r.get("ai_abx_code_1", None))
        ip = int((c1 is not None) and (a1 is not None) and (c1 == a1))

        overlap_n.append(ov)
        identical_set.append(idset)
        identical_primary.append(ip)
        discordant.append(int(idset == 0))

        if both_empty:
            label.append("Both no antibiotic")
        else:
            label.append(
                "Identical regimen" if idset else
                ("Same primary agent" if ip else ("Partial overlap" if ov > 0 else "No overlap"))
            )

    out = df.copy()

    # Python-derived concordance columns (do not overwrite workbook columns)
    out["regimen_overlap_n_py"] = np.array(overlap_n, dtype=int)
    out["regimen_identical_set_py"] = np.array(identical_set, dtype=int)
    out["identical_primary_agent_py"] = np.array(identical_primary, dtype=int)
    out["regimen_discordant_py"] = np.array(discordant, dtype=int)
    out["regimen_agreement_label_py"] = pd.Series(label, dtype="string")

    # Back-compat: if canonical columns are missing, populate them (but don't overwrite if already present)
    back_compat = [
        ("regimen_overlap_n", "regimen_overlap_n_py"),
        ("regimen_identical_set", "regimen_identical_set_py"),
        ("identical_primary_agent", "identical_primary_agent_py"),
        ("regimen_discordant", "regimen_discordant_py"),
        ("regimen_agreement_label", "regimen_agreement_label_py"),
    ]
    for dst, src in back_compat:
        if dst not in out.columns:
            out[dst] = out[src]

    return out


def tableS_concordance_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer python-derived columns if present
    idset_col = "regimen_identical_set_py" if "regimen_identical_set_py" in df.columns else "regimen_identical_set"
    disc_col = "regimen_discordant_py" if "regimen_discordant_py" in df.columns else "regimen_discordant"
    ip_col = "identical_primary_agent_py" if "identical_primary_agent_py" in df.columns else "identical_primary_agent"
    ov_col = "regimen_overlap_n_py" if "regimen_overlap_n_py" in df.columns else "regimen_overlap_n"
    lab_col = "regimen_agreement_label_py" if "regimen_agreement_label_py" in df.columns else "regimen_agreement_label"

    if idset_col not in df.columns:
        return pd.DataFrame()

    n = len(df)
    rows: List[Dict[str, Any]] = []

    def add(name: str, mask: pd.Series):
        k = int(mask.sum())
        rows.append({"metric": name, "n": k, "pct": round(100.0 * k / n, 1)})

    add("Exact identical regimen set", df[idset_col].eq(1))
    add("Discordant (not identical set)", df[disc_col].eq(1))
    add("Same primary agent (drug_1)", df[ip_col].eq(1))
    add("Any overlap (AI∩Clin>=1)", df[ov_col].ge(1))
    add("No overlap (AI∩Clin=0)", df[ov_col].eq(0))

    vc = df[lab_col].astype("string").fillna("NA").value_counts()
    for level, k in vc.items():
        rows.append({"metric": f"Agreement label: {level}", "n": int(k), "pct": round(100.0 * int(k) / n, 1)})

    return pd.DataFrame(rows)


# ----------------------------- coverage evaluability -----------------------------

def normalize_coverage_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    x = x.where(x.isin([0, 1, 9]), np.nan)
    return x.replace({9: np.nan})


def add_micro_evaluable_pair(df: pd.DataFrame, clin_col: str, ai_col: str) -> pd.DataFrame:
    out = df.copy()
    if clin_col not in out.columns or ai_col not in out.columns:
        out["micro_evaluable_pair"] = np.nan
        return out
    a = normalize_coverage_series(out[clin_col])
    b = normalize_coverage_series(out[ai_col])
    out["micro_evaluable_pair"] = (a.notna() & b.notna()).astype(int)
    return out


def tableS_micro_evaluable_bias_qc(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    if "micro_evaluable_pair" not in df.columns or df["micro_evaluable_pair"].isna().all():
        return pd.DataFrame()
    g1 = df[df["micro_evaluable_pair"] == 1]
    g0 = df[df["micro_evaluable_pair"] == 0]
    rows: List[Dict[str, Any]] = []

    def add_bin(label: str, col: str):
        if col not in df.columns:
            return
        b1 = to_int(g1[col]); b0 = to_int(g0[col])
        n1 = int(len(b1)); n0 = int(len(b0))
        p1 = float(b1.mean()) if n1 else np.nan
        p0 = float(b0.mean()) if n0 else np.nan
        p = np.nan
        try:
            tab = np.array([
                [int((b1 == 1).sum()), int((b1 == 0).sum())],
                [int((b0 == 1).sum()), int((b0 == 0).sum())],
            ])
            _, p, _, _ = stats.chi2_contingency(tab, correction=True)
            p = float(p)
        except Exception:
            p = np.nan
        rows.append({
            "variable": label, "type": "binary",
            "n_evaluable": n1, "prop_evaluable": p1,
            "n_nonevaluable": n0, "prop_nonevaluable": p0,
            "p_value": p
        })

    def add_cont(label: str, col: str):
        if col not in df.columns:
            return
        x1 = pd.to_numeric(g1[col], errors="coerce").dropna()
        x0 = pd.to_numeric(g0[col], errors="coerce").dropna()
        p = np.nan
        if len(x1) and len(x0):
            try:
                p = float(stats.mannwhitneyu(x1, x0, alternative="two-sided").pvalue)
            except Exception:
                p = np.nan
        rows.append({
            "variable": label, "type": "continuous",
            "n_evaluable": int(len(x1)), "median_evaluable": float(x1.median()) if len(x1) else np.nan,
            "n_nonevaluable": int(len(x0)), "median_nonevaluable": float(x0.median()) if len(x0) else np.nan,
            "p_value": p
        })

    # ICU admission proxy (from admitting_ward)
    if cfg.admitting_ward_col in df.columns:
        ward = as_str(df[cfg.admitting_ward_col]).str.upper().str.strip()
        tmp_icu = (ward == "ICU").astype(int)
        df2 = df.copy()
        df2["_tmp_ward_icu"] = tmp_icu
        g1 = df2[df2["micro_evaluable_pair"] == 1]
        g0 = df2[df2["micro_evaluable_pair"] == 0]
        add_bin("Admitted ward ICU", "_tmp_ward_icu")

    add_cont("Age (years)", "age_years")
    add_cont("LOS (days)", "los_days")

    for label, col in [
        ("Sepsis documented", "sepsis_documented"),
        ("Septic shock documented", "septic_shock_documented"),
        ("Vasopressors any", "vasopressors_any"),
        ("Mechanical ventilation any", "mechanical_ventilation_any"),
        ("Respiratory failure documented", "respiratory_failure_documented"),
        ("ABX last 90d", "abx_last_90d"),
        ("Hospitalization last 90d", "hospitalization_last_90d"),
        ("LTC resident", "ltc_facility_resident"),
        ("Prior ESBL/CRE/VRE", "prior_esbl_cre_vre"),
        ("Prior MRSA colonization", "prior_mrsa_colonization"),
        ("Any culture collected within 24h", "culture_any_collected_24h"),
        ("Any culture positive", "culture_result_any_positive"),
        ("Viral screening positive", "viral_screening_pozitive"),
    ]:
        add_bin(label, col)

    rows.append({
        "variable": "__counts__", "type": "meta",
        "n_total": int(len(df)),
        "n_evaluable_pairs": int((df["micro_evaluable_pair"] == 1).sum()),
        "n_nonevaluable_pairs": int((df["micro_evaluable_pair"] == 0).sum())
    })
    return pd.DataFrame(rows)


# ----------------------------- contextual guardrails & penalties -----------------------------

def _series_or_blank(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([""] * len(df), index=df.index, dtype="string")


def compute_contextual_guardrails(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """
    Requires regimen class binaries (usually in endpoints):
      clin/ai_carbapenem_any, clin/ai_antipseudomonal_any, clin/ai_anti_mrsa_any

    Produces:
      SEVERE (no icu_transfer_any), MDR_RISK (excludes mdr_risk_text),
      violations + penalties, delta_guardrail_penalty_context (AI-Clin).
    """
    req = [
        "clin_carbapenem_any", "clin_antipseudomonal_any", "clin_anti_mrsa_any",
        "ai_carbapenem_any", "ai_antipseudomonal_any", "ai_anti_mrsa_any",
    ]
    if not all(c in df.columns for c in req):
        return df

    out = df.copy()

    # SEVERE (baseline proxies; no icu_transfer_any)
    ward = as_str(_series_or_blank(out, cfg.admitting_ward_col)).str.upper().str.strip()
    severe = (ward == "ICU").astype(int).to_numpy()
    for c in ["septic_shock_documented", "vasopressors_any", "mechanical_ventilation_any", "respiratory_failure_documented"]:
        if c in out.columns:
            severe = np.maximum(severe, to_int(out[c]).to_numpy())
    out["SEVERE"] = severe.astype(int)

    # MDR_RISK (structured only + acquisition_text; EXCLUDES mdr_risk_text)
    mdr = np.zeros(len(out), dtype=int)
    for c in ["prior_esbl_cre_vre", "abx_last_90d", "hospitalization_last_90d", "ltc_facility_resident"]:
        if c in out.columns:
            mdr = np.maximum(mdr, to_int(out[c]).to_numpy())
    acq = as_str(_series_or_blank(out, "acquisition_text")).str.lower().str.strip()
    mdr = np.maximum(mdr, (acq == "healthcare-associated").astype(int).to_numpy())
    out["MDR_RISK"] = mdr.astype(int)

    out["MRSA_JUSTIFIED"] = to_int(out.get("prior_mrsa_colonization", pd.Series([0]*len(out), index=out.index)))
    out["APS_JUSTIFIED"] = (out["SEVERE"].astype(int) | out["MDR_RISK"].astype(int)).astype(int)

    # Legacy: CARB_JUSTIFIED = prior_esbl_cre_vre OR (MDR_RISK AND SEVERE)
    prior_esbl = to_int(out.get("prior_esbl_cre_vre", pd.Series([0]*len(out), index=out.index))).to_numpy()
    out["CARB_JUSTIFIED"] = np.maximum(prior_esbl, (out["MDR_RISK"].to_numpy() & out["SEVERE"].to_numpy()).astype(int)).astype(int)

    def viol(use_col: str, just_col: str) -> np.ndarray:
        use = to_int(out[use_col]).to_numpy()
        just = to_int(out[just_col]).to_numpy()
        return ((use == 1) & (just == 0)).astype(int)

    out["clin_carb_violation"] = viol("clin_carbapenem_any", "CARB_JUSTIFIED")
    out["clin_aps_violation"]  = viol("clin_antipseudomonal_any", "APS_JUSTIFIED")
    out["clin_mrsa_violation"] = viol("clin_anti_mrsa_any", "MRSA_JUSTIFIED")

    out["ai_carb_violation"] = viol("ai_carbapenem_any", "CARB_JUSTIFIED")
    out["ai_aps_violation"]  = viol("ai_antipseudomonal_any", "APS_JUSTIFIED")
    out["ai_mrsa_violation"] = viol("ai_anti_mrsa_any", "MRSA_JUSTIFIED")

    out["clin_guardrail_penalty_context"] = (2*out["clin_carb_violation"] + out["clin_aps_violation"] + out["clin_mrsa_violation"]).astype(int)
    out["ai_guardrail_penalty_context"]   = (2*out["ai_carb_violation"]   + out["ai_aps_violation"]   + out["ai_mrsa_violation"]).astype(int)
    out["delta_guardrail_penalty_context"] = (out["ai_guardrail_penalty_context"] - out["clin_guardrail_penalty_context"]).astype(int)

    return out


# ----------------------------- composites / trade-off endpoints -----------------------------

def add_guardrail_composites(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite stewardship indicators intuitive for ID/AMS audiences.

    Requires:
      - use binaries: clin/ai_carbapenem_any, clin/ai_antipseudomonal_any, clin/ai_anti_mrsa_any
      - contextual violations: clin/ai_carb_violation, clin/ai_aps_violation, clin/ai_mrsa_violation

    Produces:
      - clin/ai_any_broad_use (0/1)
      - clin/ai_broad_use_count (0..3) and delta_broad_use_count (AI-Clin)
      - clin/ai_any_violation (0/1) and delta_any_violation (AI-Clin; -1/0/1)
      - delta_any_broad_use (AI-Clin; -1/0/1)
    """
    out = df.copy()

    req_use = [
        "clin_carbapenem_any", "clin_antipseudomonal_any", "clin_anti_mrsa_any",
        "ai_carbapenem_any", "ai_antipseudomonal_any", "ai_anti_mrsa_any",
    ]
    req_viol = [
        "clin_carb_violation", "clin_aps_violation", "clin_mrsa_violation",
        "ai_carb_violation", "ai_aps_violation", "ai_mrsa_violation",
    ]

    if all(c in out.columns for c in req_use):
        clin_any = (
            (to_int(out["clin_carbapenem_any"]) == 1) |
            (to_int(out["clin_antipseudomonal_any"]) == 1) |
            (to_int(out["clin_anti_mrsa_any"]) == 1)
        ).astype(int)
        ai_any = (
            (to_int(out["ai_carbapenem_any"]) == 1) |
            (to_int(out["ai_antipseudomonal_any"]) == 1) |
            (to_int(out["ai_anti_mrsa_any"]) == 1)
        ).astype(int)

        out["clin_any_broad_use"] = clin_any
        out["ai_any_broad_use"] = ai_any
        out["delta_any_broad_use"] = (ai_any - clin_any).astype(int)

        clin_cnt = (
            to_int(out["clin_carbapenem_any"]) +
            to_int(out["clin_antipseudomonal_any"]) +
            to_int(out["clin_anti_mrsa_any"])
        ).astype(int)
        ai_cnt = (
            to_int(out["ai_carbapenem_any"]) +
            to_int(out["ai_antipseudomonal_any"]) +
            to_int(out["ai_anti_mrsa_any"])
        ).astype(int)

        out["clin_broad_use_count"] = clin_cnt
        out["ai_broad_use_count"] = ai_cnt
        out["delta_broad_use_count"] = (ai_cnt - clin_cnt).astype(int)

    if all(c in out.columns for c in req_viol):
        clin_v = (
            (to_int(out["clin_carb_violation"]) == 1) |
            (to_int(out["clin_aps_violation"]) == 1) |
            (to_int(out["clin_mrsa_violation"]) == 1)
        ).astype(int)
        ai_v = (
            (to_int(out["ai_carb_violation"]) == 1) |
            (to_int(out["ai_aps_violation"]) == 1) |
            (to_int(out["ai_mrsa_violation"]) == 1)
        ).astype(int)

        out["clin_any_violation"] = clin_v
        out["ai_any_violation"] = ai_v
        out["delta_any_violation"] = (ai_v - clin_v).astype(int)

    return out


def table3b_guardrails_composites(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """Paired McNemar for composite, reader-friendly binary stewardship endpoints."""
    rows: List[Dict[str, Any]] = []
    pairs = [
        ("Any broad-spectrum class used (carb OR APS OR anti-MRSA)", "clin_any_broad_use", "ai_any_broad_use"),
        ("Any contextual guardrail violation (any of 3)", cfg.primary_binary_clin_col, cfg.primary_binary_ai_col),
    ]
    for label, c1, c2 in pairs:
        if c1 in df.columns and c2 in df.columns:
            res = mcnemar_exact(to_int(df[c1]).to_numpy(), to_int(df[c2]).to_numpy())
            res["endpoint"] = label
            rows.append(res)
    return pd.DataFrame(rows)


def paired_delta_summary(df: pd.DataFrame, col: str, cfg: RunConfig) -> Dict[str, object]:
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    n = int(np.sum(~np.isnan(x)))
    if n == 0:
        return {"endpoint": col, "n": 0}

    med = float(np.nanmedian(x))
    q1 = float(np.nanquantile(x, 0.25))
    q3 = float(np.nanquantile(x, 0.75))
    mean = float(np.nanmean(x))
    sd = float(np.nanstd(x, ddof=1)) if n > 1 else np.nan

    med_lo, med_hi = bootstrap_ci(x, "median", cfg.n_boot, cfg.seed)
    mean_lo, mean_hi = bootstrap_ci(x, "mean", cfg.n_boot, cfg.seed)
    w_stat, p, _ = wilcoxon_pratt(x)
    rrb = rank_biserial_from_diffs(x)

    out = {
        "endpoint": col, "n": n,
        "median": med, "q1": q1, "q3": q3,
        "median_ci_lo": med_lo, "median_ci_hi": med_hi,
        "mean": mean, "sd": sd,
        "mean_ci_lo": mean_lo, "mean_ci_hi": mean_hi,
        "wilcoxon_stat": w_stat, "p_value": float(p),
        "rank_biserial": rrb,
        "improved_n": int(np.nansum(x < 0)),
        "worsened_n": int(np.nansum(x > 0)),
        "tied_n": int(np.nansum(x == 0)),
    }
    out.update(sign_test_non_ties(x))
    return out


def table2_primary_key_secondary(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    # PRIMARY (binary paired): any contextual violation
    c1 = cfg.primary_binary_clin_col
    c2 = cfg.primary_binary_ai_col
    if c1 in df.columns and c2 in df.columns:
        res = mcnemar_exact(to_int(df[c1]).to_numpy(), to_int(df[c2]).to_numpy())
        res.update({
            "endpoint": "PRIMARY: Any contextual guardrail violation (binary, paired)",
            "type": "binary_paired",
            "clin_rate": float(to_int(df[c1]).mean()),
            "ai_rate": float(to_int(df[c2]).mean()),
        })
        rows.append(res)
    else:
        rows.append({"endpoint": "PRIMARY: Any contextual guardrail violation", "type": "binary_paired", "error": f"missing cols: {c1}/{c2}"})

    # KEY SECONDARY (delta): contextual penalty score
    key = cfg.key_secondary_delta_col
    if key in df.columns:
        d = paired_delta_summary(df, key, cfg)
        d.update({"endpoint": "KEY SECONDARY: Δ contextual guardrail penalty score (AI-Clin)", "type": "delta_continuous"})
        rows.append(d)
    else:
        rows.append({"endpoint": "KEY SECONDARY: Δ contextual guardrail penalty score", "type": "delta_continuous", "error": f"missing col: {key}"})

    return pd.DataFrame(rows)


def table2_secondary_deltas(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """All other deltas (including the key secondary, for convenience)."""
    candidates = [
        cfg.key_secondary_delta_col,
        "delta_guardrail_penalty",
        "delta_aware_mean", "delta_aware_max", "delta_aware_sum",
        "ΔDDD/24h",
        "delta_empiric_cost_24h_eur_median_prices",
        "delta_empiric_cost_72h_eur_median_prices",
        "delta_empiric_cost_72h_eur_median_prices_true",
    ]
    present = [c for c in candidates if c in df.columns]
    out = pd.DataFrame([paired_delta_summary(df, c, cfg) for c in present]) if present else pd.DataFrame()
    if out.empty:
        return out
    out["p_holm_all"] = holm_adjust(out["p_value"].astype(float).tolist())
    out["is_key_secondary"] = out["endpoint"].eq(cfg.key_secondary_delta_col)
    return out.sort_values(["is_key_secondary", "endpoint"], ascending=[False, True]).reset_index(drop=True)


def table2b_endpoints_plus(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    candidates = ["delta_any_broad_use", "delta_broad_use_count", "delta_any_violation"]
    present = [c for c in candidates if c in df.columns]
    if not present:
        return pd.DataFrame()
    out = pd.DataFrame([paired_delta_summary(df, c, cfg) for c in present])
    out["p_holm_all"] = holm_adjust(out["p_value"].astype(float).tolist())
    return out.sort_values(["endpoint"]).reset_index(drop=True)


# ----------------------------- guardrails paired tables -----------------------------

def table3_guardrails_components(df: pd.DataFrame) -> pd.DataFrame:
    comps = [
        ("carbapenem_any", "clin_carbapenem_any", "ai_carbapenem_any"),
        ("antipseudomonal_any", "clin_antipseudomonal_any", "ai_antipseudomonal_any"),
        ("anti_mrsa_any", "clin_anti_mrsa_any", "ai_anti_mrsa_any"),
    ]
    rows = []
    for label, c1, c2 in comps:
        if c1 in df.columns and c2 in df.columns:
            res = mcnemar_exact(to_int(df[c1]).to_numpy(), to_int(df[c2]).to_numpy())
            res["component"] = label
            rows.append(res)
    return pd.DataFrame(rows)


def tableS_guardrail_context_violations(df: pd.DataFrame) -> pd.DataFrame:
    comps = [
        ("carb_violation", "clin_carb_violation", "ai_carb_violation"),
        ("aps_violation",  "clin_aps_violation",  "ai_aps_violation"),
        ("mrsa_violation", "clin_mrsa_violation", "ai_mrsa_violation"),
    ]
    rows = []
    for label, c1, c2 in comps:
        if c1 in df.columns and c2 in df.columns:
            res = mcnemar_exact(to_int(df[c1]).to_numpy(), to_int(df[c2]).to_numpy())
            res["component"] = label
            rows.append(res)
    return pd.DataFrame(rows)


def table4_coverage_paired(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    if cfg.clin_cov_col not in df.columns or cfg.ai_cov_col not in df.columns:
        return pd.DataFrame()
    a = normalize_coverage_series(df[cfg.clin_cov_col])
    b = normalize_coverage_series(df[cfg.ai_cov_col])
    mask = a.notna() & b.notna()
    if int(mask.sum()) == 0:
        return pd.DataFrame()
    res = mcnemar_exact(to_int(a[mask]).to_numpy(), to_int(b[mask]).to_numpy())
    res["coverage_definition"] = f"{cfg.clin_cov_col} vs {cfg.ai_cov_col} (9 -> NaN; evaluable pairs only)"
    res["n_evaluable"] = float(mask.sum())
    return pd.DataFrame([res])


def tableS_micro_tradeoff_success(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """
    Micro-evaluable paired SUCCESS:
      SUCCESS = active vs index organism (1) AND no contextual guardrail violation (0)

    Restricted to evaluable pairs only (coverage 0/1 in both arms; 9 treated as non-evaluable).
    This is a trade-off endpoint (adequacy + stewardship), NOT a clinical outcome claim.
    """
    if cfg.clin_cov_col not in df.columns or cfg.ai_cov_col not in df.columns:
        return pd.DataFrame()
    if cfg.primary_binary_clin_col not in df.columns or cfg.primary_binary_ai_col not in df.columns:
        return pd.DataFrame()

    a = normalize_coverage_series(df[cfg.clin_cov_col])
    b = normalize_coverage_series(df[cfg.ai_cov_col])
    mask = a.notna() & b.notna()
    n = int(mask.sum())
    if n == 0:
        return pd.DataFrame()

    tmp = df.loc[mask].copy()
    clin_success = ((a[mask] == 1) & (to_int(tmp[cfg.primary_binary_clin_col]) == 0)).astype(int)
    ai_success = ((b[mask] == 1) & (to_int(tmp[cfg.primary_binary_ai_col]) == 0)).astype(int)

    res = mcnemar_exact(clin_success.to_numpy(), ai_success.to_numpy())
    res.update({
        "endpoint": "Micro-evaluable success: active AND no contextual violation",
        "n_evaluable_pairs": float(n),
        "clin_success_rate": float(clin_success.mean()),
        "ai_success_rate": float(ai_success.mean()),
        "clin_active_rate": float((a[mask] == 1).mean()),
        "ai_active_rate": float((b[mask] == 1).mean()),
        "clin_any_violation_rate": float((to_int(tmp[cfg.primary_binary_clin_col]) == 1).mean()),
        "ai_any_violation_rate": float((to_int(tmp[cfg.primary_binary_ai_col]) == 1).mean()),
    })
    return pd.DataFrame([res])


def tableS_subgroup_lowrisk_guardrails(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """
    Pre-specified subgroup: LOW RISK = SEVERE==0 AND MDR_RISK==0.
    Paired McNemar for composites + key secondary delta summary in subgroup.
    """
    if "SEVERE" not in df.columns or "MDR_RISK" not in df.columns:
        return pd.DataFrame()

    g = df[(to_int(df["SEVERE"]) == 0) & (to_int(df["MDR_RISK"]) == 0)].copy()
    n = int(len(g))
    if n < max(10, int(cfg.min_subgroup_n)):
        return pd.DataFrame([{
            "note": "Low-risk subgroup too small for inferential reporting under current min_subgroup_n.",
            "n_lowrisk": n,
            "min_subgroup_n": cfg.min_subgroup_n,
        }])

    rows: List[Dict[str, Any]] = []
    for label, c1, c2 in [
        ("Low-risk: any broad-spectrum class used", "clin_any_broad_use", "ai_any_broad_use"),
        ("Low-risk: any contextual guardrail violation", cfg.primary_binary_clin_col, cfg.primary_binary_ai_col),
    ]:
        if c1 in g.columns and c2 in g.columns:
            res = mcnemar_exact(to_int(g[c1]).to_numpy(), to_int(g[c2]).to_numpy())
            res["endpoint"] = label
            res["n_lowrisk"] = float(n)
            rows.append(res)

    key = cfg.key_secondary_delta_col
    if key in g.columns:
        d = paired_delta_summary(g, key, cfg)
        d["endpoint"] = f"Low-risk subgroup: {key}"
        d["n_lowrisk"] = float(n)
        rows.append(d)

    return pd.DataFrame(rows)


# ----------------------------- publication-safe exploratory predictors -----------------------------

def _collapse_topk(series: pd.Series, topk: int = 8, min_count: int = 15) -> pd.Series:
    s = series.astype("string").fillna("NA")
    vc = s.value_counts()
    top = vc.head(topk).index.tolist()
    out = pd.Series(np.where(s.isin(top), s, "Other"), index=s.index, dtype="string")
    vc2 = out.value_counts()
    rare = vc2[vc2 < min_count].index.tolist()
    if rare:
        out = out.where(~out.isin(rare), "Other")
    return out


def _safe_numeric_or_string(s: pd.Series) -> pd.Series:
    if s.dtype == "object" or str(s.dtype).startswith("string"):
        return s.astype("string")
    return pd.to_numeric(s, errors="coerce")


def _prep_model_df(
    df: pd.DataFrame,
    outcome_col: str,
    predictors: List[str],
    collapse_categoricals: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {"outcome": outcome_col, "predictors_requested": predictors}

    if outcome_col not in df.columns:
        return None, None, f"Missing outcome column: {outcome_col}", meta

    y_raw = pd.to_numeric(df[outcome_col], errors="coerce")
    X = df[predictors].copy()

    if collapse_categoricals:
        for c in X.columns:
            if X[c].dtype == "object" or str(X[c].dtype).startswith("string"):
                X[c] = _collapse_topk(X[c], topk=8, min_count=15)

    for c in X.columns:
        X[c] = _safe_numeric_or_string(X[c])

    cat_cols = [c for c in X.columns if (X[c].dtype == "object" or str(X[c].dtype).startswith("string"))]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    for c in X.columns:
        dt = str(X[c].dtype)
        if dt in ("boolean", "bool"):
            X[c] = X[c].astype(int)

    nunique = X.nunique(dropna=True)
    X = X.loc[:, nunique > 1]

    model_df = pd.concat([y_raw.rename("y"), X], axis=1)
    meta["n_before_dropna"] = int(len(model_df))
    meta["missing_by_var"] = {c: int(model_df[c].isna().sum()) for c in model_df.columns}

    model_df = model_df.dropna(axis=0)
    meta["n_complete_case"] = int(len(model_df))

    if model_df.empty:
        return None, None, "No complete-case rows after filtering", meta
    if model_df["y"].nunique() < 2:
        return None, None, "Outcome has <2 classes after filtering", meta

    X2 = sm.add_constant(model_df.drop(columns=["y"]), has_constant="add")
    X2 = X2.apply(pd.to_numeric, errors="coerce")
    y2 = model_df["y"].astype(int)
    meta["n"] = int(len(y2))
    meta["events"] = int(y2.sum())
    meta["terms_after_dummies"] = X2.columns.tolist()
    return X2, y2, None, meta


def fit_logit_hc3(X: pd.DataFrame, y: pd.Series):
    if sm is None:
        raise RuntimeError("statsmodels not available")
    model = sm.Logit(y, X)
    res = model.fit(disp=False, maxiter=200, cov_type="HC3")
    return res


def _or_ci_from_beta(b: float, se: float) -> Tuple[float, float, float]:
    or_ = float(np.exp(b))
    lo = float(np.exp(b - 1.96 * se))
    hi = float(np.exp(b + 1.96 * se))
    return or_, lo, hi


def _summarize_logit_result(res, terms: List[str], method_label: str, n: int, events: int, outcome: str) -> pd.DataFrame:
    params = res.params
    cov = res.cov_params()
    rows = []
    for term in terms:
        if term == "const":
            continue
        if term not in params.index:
            continue
        b = float(params.loc[term])
        se = float(np.sqrt(cov.loc[term, term])) if term in cov.index else np.nan
        if not np.isfinite(se):
            continue
        or_, lo, hi = _or_ci_from_beta(b, se)
        p = float(res.pvalues.loc[term]) if term in res.pvalues.index else np.nan
        rows.append({
            "outcome": outcome, "term": term,
            "OR": or_, "CI95_low": lo, "CI95_high": hi,
            "p_value": p, "method": method_label,
            "n": int(n), "events": int(events),
        })
    return pd.DataFrame(rows)


def _fisher_exact_for_binary(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    x = x.astype(int); y = y.astype(int)
    a = int(((x == 1) & (y == 1)).sum())
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    d = int(((x == 0) & (y == 0)).sum())
    table = np.array([[a, b], [c, d]], dtype=int)
    try:
        _, p = stats.fisher_exact(table, alternative="two-sided")
        p = float(p)
    except Exception:
        p = np.nan
    cc = 0.5
    or_ = float((a + cc) * (d + cc) / ((b + cc) * (c + cc)))
    se = float(np.sqrt(1.0/(a+cc) + 1.0/(b+cc) + 1.0/(c+cc) + 1.0/(d+cc)))
    lo = float(np.exp(np.log(or_) - 1.96*se))
    hi = float(np.exp(np.log(or_) + 1.96*se))
    return {"OR": or_, "CI95_low": lo, "CI95_high": hi, "p_value": p, "a": a, "b": b, "c": c, "d": d}


def _is_binary_series(s: pd.Series) -> bool:
    v = pd.to_numeric(s, errors="coerce").dropna().unique()
    if len(v) == 0:
        return False
    return set(v.tolist()).issubset({0, 1})


def _model_predictor_list(tmp: pd.DataFrame, cfg: RunConfig) -> List[str]:
    candidate_predictors = [
        cfg.admitting_ward_col,
        "acquisition_text",

        "sepsis_documented",
        "septic_shock_documented",
        "vasopressors_any",
        "mechanical_ventilation_any",
        "respiratory_failure_documented",
        "SEVERE",

        "prior_esbl_cre_vre",
        "abx_last_90d",
        "hospitalization_last_90d",
        "ltc_facility_resident",
        "MDR_RISK",

        "age_years",
        "sex",
        "viral_screening_pozitive",
        "home_abx_before_hosp",
    ]
    predictors = [c for c in candidate_predictors if c in tmp.columns]

    severe_components = [
        "sepsis_documented",
        "septic_shock_documented",
        "vasopressors_any",
        "mechanical_ventilation_any",
        "respiratory_failure_documented",
    ]
    if "SEVERE" in predictors:
        predictors = [c for c in predictors if c not in severe_components]

    mdr_components = [
        "prior_esbl_cre_vre",
        "abx_last_90d",
        "hospitalization_last_90d",
        "ltc_facility_resident",
        "acquisition_text",
    ]
    if "MDR_RISK" in predictors:
        predictors = [c for c in predictors if c not in mdr_components]

    return predictors


def run_exploratory_predictors(df_merged: pd.DataFrame, out_dir: Path, cfg: RunConfig) -> Dict[str, Any]:
    """
    Exploratory, publication-safe modeling:
      Outcomes derived from cfg.delta_for_models_col (AI - Clinician; negative favors AI):
        - nonzero_delta_guardrail = 1[delta != 0]
        - ai_improved_guardrail = 1[delta < 0]
    """
    status: Dict[str, Any] = {"ran": False}

    if sm is None:
        err = "statsmodels not available"
        for fname in [
            "tableS_predictors_of_nonzero_delta_guardrail.csv",
            "tableS_predictors_of_ai_improvement_guardrail.csv",
        ]:
            pd.DataFrame([{"error": err}]).to_csv(out_dir / fname, index=False)
        pd.DataFrame([{"error": err}]).to_csv(out_dir / "tableS_models_missingness.csv", index=False)
        status.update({"ran": False, "error": err})
        return status

    delta_col = cfg.delta_for_models_col
    if delta_col not in df_merged.columns:
        err = f"Missing delta column for models: {delta_col}"
        for fname in [
            "tableS_predictors_of_nonzero_delta_guardrail.csv",
            "tableS_predictors_of_ai_improvement_guardrail.csv",
        ]:
            pd.DataFrame([{"error": err}]).to_csv(out_dir / fname, index=False)
        pd.DataFrame([{"error": err}]).to_csv(out_dir / "tableS_models_missingness.csv", index=False)
        status.update({"ran": False, "error": err})
        return status

    tmp = df_merged.copy()
    delta = pd.to_numeric(tmp[delta_col], errors="coerce")
    tmp["nonzero_delta_guardrail"] = (delta != 0).astype(int)
    tmp["ai_improved_guardrail"] = (delta < 0).astype(int)

    predictors = _model_predictor_list(tmp, cfg)
    outcomes = [
        ("nonzero_delta_guardrail", "tableS_predictors_of_nonzero_delta_guardrail.csv"),
        ("ai_improved_guardrail", "tableS_predictors_of_ai_improvement_guardrail.csv"),
    ]

    missing_summary_rows: List[Dict[str, Any]] = []
    missing_byvar_rows: List[Dict[str, Any]] = []

    for outcome_col, filename in outcomes:
        X, y, err, meta = _prep_model_df(tmp, outcome_col, predictors, collapse_categoricals=True)

        missing_summary_rows.append({
            "outcome": outcome_col,
            "variable": "__MODEL__",
            "missing_n": 0,
            "n_before_dropna": int(meta.get("n_before_dropna", np.nan)) if meta.get("n_before_dropna") is not None else np.nan,
            "n_complete_case": int(meta.get("n_complete_case", np.nan)) if meta.get("n_complete_case") is not None else np.nan,
            "events": int(meta.get("events", np.nan)) if meta.get("events") is not None else np.nan,
            "dropped_n": int(meta.get("n_before_dropna", 0) - meta.get("n_complete_case", 0)) if (
                meta.get("n_before_dropna") is not None and meta.get("n_complete_case") is not None
            ) else np.nan,
            "note": "complete-case sample used for modeling",
        })

        mbv = meta.get("missing_by_var", {}) or {}
        for var, miss_n in mbv.items():
            var_name = outcome_col if var == "y" else str(var)
            missing_byvar_rows.append({
                "outcome": outcome_col,
                "variable": var_name,
                "missing_n": int(miss_n),
                "n_before_dropna": np.nan,
                "n_complete_case": np.nan,
                "dropped_n": np.nan,
                "events": np.nan,
                "note": "missing counted before complete-case drop",
            })

        if err is not None:
            pd.DataFrame([{"outcome": outcome_col, "error": err}]).to_csv(out_dir / filename, index=False)
            continue

        n = int(meta["n"]); events = int(meta["events"])
        terms = X.columns.tolist()

        mv_table = None
        mv_err = None
        try:
            res = fit_logit_hc3(X, y)
            mv_table = _summarize_logit_result(res, terms, "mv_logit_HC3", n, events, outcome_col)
        except (PerfectSeparationError, LinAlgError, FloatingPointError, ValueError) as e:
            msg = str(e).strip().replace("\n", " ")
            if len(msg) > 220:
                msg = msg[:217] + "..."
            mv_err = f"{type(e).__name__}: {msg}" if msg else f"{type(e).__name__}"

        if mv_table is not None and not mv_table.empty:
            mv_table.to_csv(out_dir / filename, index=False)
            continue

        uv_rows = []
        for term in [c for c in terms if c != "const"]:
            X1 = X[["const", term]]
            try:
                res1 = fit_logit_hc3(X1, y)
                uv_rows.append(_summarize_logit_result(res1, X1.columns.tolist(), "uv_logit_HC3", n, events, outcome_col))
            except PerfectSeparationError:
                xterm = X1[term]
                if _is_binary_series(xterm):
                    fx = _fisher_exact_for_binary(pd.to_numeric(xterm, errors="coerce").fillna(0).to_numpy(), y.to_numpy())
                    uv_rows.append(pd.DataFrame([{
                        "outcome": outcome_col, "term": term,
                        "OR": fx["OR"], "CI95_low": fx["CI95_low"], "CI95_high": fx["CI95_high"],
                        "p_value": fx["p_value"],
                        "method": "uv_fisher_exact",
                        "n": int(n), "events": int(events),
                        "a": fx["a"], "b": fx["b"], "c": fx["c"], "d": fx["d"],
                    }]))
                else:
                    uv_rows.append(pd.DataFrame([{
                        "outcome": outcome_col, "term": term,
                        "OR": np.nan, "CI95_low": np.nan, "CI95_high": np.nan, "p_value": np.nan,
                        "method": "uv_separation_unhandled",
                        "n": int(n), "events": int(events),
                    }]))
            except (LinAlgError, FloatingPointError, ValueError) as e:
                xterm = X1[term]
                if _is_binary_series(xterm):
                    fx = _fisher_exact_for_binary(pd.to_numeric(xterm, errors="coerce").fillna(0).to_numpy(), y.to_numpy())
                    uv_rows.append(pd.DataFrame([{
                        "outcome": outcome_col, "term": term,
                        "OR": fx["OR"], "CI95_low": fx["CI95_low"], "CI95_high": fx["CI95_high"],
                        "p_value": fx["p_value"],
                        "method": "uv_fisher_exact_after_uv_fail",
                        "n": int(n), "events": int(events),
                        "a": fx["a"], "b": fx["b"], "c": fx["c"], "d": fx["d"],
                        "uv_fail_reason": f"{type(e).__name__}: {str(e).strip().replace(chr(10), ' ')[:220]}",
                    }]))
                else:
                    msg = str(e).strip().replace("\n", " ")
                    if len(msg) > 220:
                        msg = msg[:217] + "..."
                    uv_rows.append(pd.DataFrame([{
                        "outcome": outcome_col, "term": term,
                        "OR": np.nan, "CI95_low": np.nan, "CI95_high": np.nan, "p_value": np.nan,
                        "method": "uv_fit_failed",
                        "n": int(n), "events": int(events),
                        "uv_fail_reason": f"{type(e).__name__}: {msg}" if msg else f"{type(e).__name__}",
                    }]))

        if uv_rows:
            out_tbl = pd.concat(uv_rows, ignore_index=True)
            out_tbl["mv_attempt"] = "failed" if mv_err else "skipped"
            out_tbl["mv_fail_reason"] = mv_err if mv_err else ""
            out_tbl = out_tbl[~(out_tbl["method"].isin(["uv_fit_failed", "uv_separation_unhandled"]) & out_tbl["p_value"].isna() & out_tbl["OR"].isna())].copy()
            out_tbl.to_csv(out_dir / filename, index=False)
        else:
            pd.DataFrame([{"outcome": outcome_col, "error": "No models fit in fallback univariable step."}]).to_csv(out_dir / filename, index=False)

    miss_df = pd.concat([pd.DataFrame(missing_summary_rows), pd.DataFrame(missing_byvar_rows)], ignore_index=True)
    col_order = ["outcome", "variable", "missing_n", "n_before_dropna", "n_complete_case", "dropped_n", "events", "note"]
    for c in col_order:
        if c not in miss_df.columns:
            miss_df[c] = np.nan
    miss_df = miss_df[col_order].sort_values(["outcome", "variable"], kind="mergesort")
    miss_df.to_csv(out_dir / "tableS_models_missingness.csv", index=False)

    status.update({"ran": True, "predictors_used": predictors, "delta_used": delta_col})
    return status


# ----------------------------- AWaRe (optional) -----------------------------

def _guess_aware_col(df: pd.DataFrame) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for key in ["aware_category", "aware_group", "aware_class", "aware"]:
        if key in lower:
            return lower[key]
    for lc, orig in lower.items():
        if "aware" in lc:
            return orig
    return None


def _normalize_aware_cat(x: str) -> Optional[str]:
    s = str(x).strip().lower()
    if s in ("", "nan", "none", "na"):
        return None
    if "access" in s:
        return "Access"
    if "watch" in s:
        return "Watch"
    if "reserve" in s:
        return "Reserve"
    return None


def compute_aware_addons(df: pd.DataFrame, sheets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, object], Optional[Set[str]]]:
    meta: Dict[str, object] = {"aware_ran": False}
    mapping_keys: Optional[Set[str]] = None
    if "abx_codes" not in sheets:
        meta["reason_skipped"] = "missing_sheet_abx_codes"
        return df, meta, mapping_keys
    abx = sheets["abx_codes"].copy()
    if "abx_code" not in abx.columns:
        meta["reason_skipped"] = "missing_abx_code_col"
        return df, meta, mapping_keys
    aware_col = _guess_aware_col(abx)
    if not aware_col:
        meta["reason_skipped"] = "missing_aware_col"
        meta["abx_codes_columns"] = list(abx.columns)
        return df, meta, mapping_keys
    if not all(c in df.columns for c in (REGIMEN_COLS_CLIN + REGIMEN_COLS_AI)):
        meta["reason_skipped"] = "missing_regimen_code_cols"
        return df, meta, mapping_keys

    mapping: Dict[str, str] = {}
    for _, row in abx[["abx_code", aware_col]].dropna().iterrows():
        code = str(row["abx_code"]).strip().upper()
        cat = _normalize_aware_cat(row[aware_col])
        if code and cat:
            mapping[code] = cat
    mapping_keys = set(mapping.keys())
    meta.update({"aware_ran": True, "aware_col": aware_col, "mapping_size": int(len(mapping))})

    def regimen_cats(row: pd.Series, cols: List[str]) -> List[str]:
        cats = []
        for c in cols:
            cv = _clean_code(row.get(c, None))
            if not cv or cv == "NO_ANTIBIOTIC":
                continue
            cat = mapping.get(cv)
            if cat in ("Access", "Watch", "Reserve"):
                cats.append(cat)
        return cats

    def flags_and_cat(row: pd.Series, cols: List[str]) -> Tuple[int, int, str]:
        cats = regimen_cats(row, cols)
        if not cats:
            return 0, 0, "No antibiotic"
        has_res = any(c == "Reserve" for c in cats)
        has_watch = any(c == "Watch" for c in cats)
        has_access = any(c == "Access" for c in cats)
        wr_any = int(has_watch or has_res)
        access_only = int(has_access and not has_watch and not has_res)
        if has_res:
            reg_cat = "Any Reserve"
        elif has_access and not has_watch:
            reg_cat = "Access-only"
        elif has_watch and not has_access:
            reg_cat = "Watch-only"
        else:
            reg_cat = "Access+Watch"
        return wr_any, access_only, reg_cat

    out = df.copy()
    clin_wr, ai_wr, clin_acc, ai_acc, clin_cat, ai_cat = [], [], [], [], [], []
    for _, r in out.iterrows():
        w1, a1, c1 = flags_and_cat(r, REGIMEN_COLS_CLIN)
        w2, a2, c2 = flags_and_cat(r, REGIMEN_COLS_AI)
        clin_wr.append(w1); clin_acc.append(a1); clin_cat.append(c1)
        ai_wr.append(w2); ai_acc.append(a2); ai_cat.append(c2)

    out["clin_WR_any"] = np.array(clin_wr, dtype=int)
    out["ai_WR_any"] = np.array(ai_wr, dtype=int)
    out["clin_access_only"] = np.array(clin_acc, dtype=int)
    out["ai_access_only"] = np.array(ai_acc, dtype=int)
    out["clin_aware_regimen_cat"] = pd.Series(clin_cat, dtype="string")
    out["ai_aware_regimen_cat"] = pd.Series(ai_cat, dtype="string")

    def count_unmapped(cols: List[str]) -> Tuple[int, int]:
        total = 0; unmapped = 0
        for c in cols:
            for v in out[c].tolist():
                cv = _clean_code(v)
                if not cv or cv == "NO_ANTIBIOTIC":
                    continue
                total += 1
                if cv not in mapping:
                    unmapped += 1
        return total, unmapped

    t1, u1 = count_unmapped(REGIMEN_COLS_CLIN)
    t2, u2 = count_unmapped(REGIMEN_COLS_AI)
    meta.update({
        "total_codes_clin_n": int(t1), "unmapped_codes_clin_n": int(u1),
        "total_codes_ai_n": int(t2), "unmapped_codes_ai_n": int(u2),
        "unmapped_rate_clin": float(u1/t1) if t1 else np.nan,
        "unmapped_rate_ai": float(u2/t2) if t2 else np.nan,
    })

    return out, meta, mapping_keys


def tableS_aware_binaries(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, c1, c2 in [
        ("WR_any (Watch/Reserve in regimen)", "clin_WR_any", "ai_WR_any"),
        ("Access-only regimen", "clin_access_only", "ai_access_only"),
    ]:
        if c1 in df.columns and c2 in df.columns:
            res = mcnemar_exact(to_int(df[c1]).to_numpy(), to_int(df[c2]).to_numpy())
            res["endpoint"] = label
            rows.append(res)
    return pd.DataFrame(rows)


def tableS_aware_transitions(df: pd.DataFrame) -> pd.DataFrame:
    if "clin_aware_regimen_cat" not in df.columns or "ai_aware_regimen_cat" not in df.columns:
        return pd.DataFrame()
    order = ["No antibiotic", "Access-only", "Watch-only", "Access+Watch", "Any Reserve"]
    clin = df["clin_aware_regimen_cat"].astype("string").fillna("NA")
    ai = df["ai_aware_regimen_cat"].astype("string").fillna("NA")
    mat = pd.crosstab(clin, ai, dropna=False)
    for r in order:
        if r not in mat.index:
            mat.loc[r] = 0
    for c in order:
        if c not in mat.columns:
            mat[c] = 0
    mat = mat.loc[order, order]
    return mat.reset_index().rename(columns={"clin_aware_regimen_cat": "clin_category"})


# ----------------------------- QC + rules + manifest -----------------------------

def qc_checks(df: pd.DataFrame, cfg: RunConfig, dup_cols_dropped: List[str]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    out.append({"check": "n_rows_merged", "value": int(len(df))})
    out.append({"check": "duplicate_columns_dropped_from_data_n", "value": int(len(dup_cols_dropped))})

    out.append({"check": "primary_binary_cols", "value": f"{cfg.primary_binary_clin_col} vs {cfg.primary_binary_ai_col}"})
    out.append({"check": "primary_binary_present", "value": bool(cfg.primary_binary_clin_col in df.columns and cfg.primary_binary_ai_col in df.columns)})
    out.append({"check": "key_secondary_delta_col", "value": cfg.key_secondary_delta_col})
    out.append({"check": "mdr_text_excluded_from_logic", "value": cfg.mdr_text_col})

    if cfg.admitting_ward_col in df.columns and cfg.icu_transfer_24h_col in df.columns:
        ward = as_str(df[cfg.admitting_ward_col]).str.upper().str.strip()
        ward_icu = (ward == "ICU").astype(int)
        icu24 = to_int(df[cfg.icu_transfer_24h_col])
        mism = int((ward_icu != icu24).sum())
        out.append({"check": "ward_vs_icu_transfer_24h_mismatch_n", "value": mism})
        if cfg.fail_on_icu_mismatch and mism > 0:
            raise ValueError(f"QC FAIL: {cfg.admitting_ward_col} vs {cfg.icu_transfer_24h_col} mismatch in {mism} rows.")
    else:
        out.append({"check": "ward_vs_icu_transfer_24h_gate_skipped", "value": True})

    # delta QC (for key secondary + hist/models delta)
    for label, col in [
        ("key_secondary_delta", cfg.key_secondary_delta_col),
        ("delta_for_hist", cfg.delta_for_hist_col),
        ("delta_for_models", cfg.delta_for_models_col),
    ]:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            out.append({"check": f"{label}_nonmissing_n", "value": int(x.notna().sum())})
            out.append({"check": f"{label}_improved_n_delta_lt0", "value": int((x < 0).sum())})
            out.append({"check": f"{label}_worsened_n_delta_gt0", "value": int((x > 0).sum())})
            out.append({"check": f"{label}_tied_n_delta_eq0", "value": int((x == 0).sum())})

    return out


def build_rules_json(cfg: RunConfig, aware_meta: Dict[str, object], dup_cols_dropped: List[str]) -> Dict[str, object]:
    return {
        "created_utc": now_iso(),
        "merge_policy": "ENDPOINTS WINS: duplicate columns dropped from data before merge; endpoints kept as truth.",
        "duplicate_columns_dropped_from_data": dup_cols_dropped,
        "primary_endpoint": {
            "name": "Any contextual guardrail violation",
            "type": "binary_paired",
            "clin_col": cfg.primary_binary_clin_col,
            "ai_col": cfg.primary_binary_ai_col,
            "effect_direction": "AI lower is favorable",
        },
        "key_secondary_endpoint": {
            "name": "Δ contextual guardrail penalty score (AI-Clin)",
            "col": cfg.key_secondary_delta_col,
            "effect_direction": "negative favors AI",
        },
        "bootstrap": {"n_boot": cfg.n_boot, "seed": cfg.seed},
        "contextual_guardrails": {
            "SEVERE": [
                f"{cfg.admitting_ward_col} == 'ICU'",
                "septic_shock_documented == 1",
                "vasopressors_any == 1",
                "mechanical_ventilation_any == 1",
                "respiratory_failure_documented == 1",
                "NOTE: icu_transfer_any excluded (may be >24h).",
            ],
            "MDR_RISK": [
                "prior_esbl_cre_vre == 1",
                "abx_last_90d == 1",
                "hospitalization_last_90d == 1",
                "ltc_facility_resident == 1",
                "acquisition_text == 'healthcare-associated'",
                f"EXCLUDED: {cfg.mdr_text_col} (free-text/composite).",
            ],
            "CARB_JUSTIFIED": ["prior_esbl_cre_vre == 1 OR (MDR_RISK==1 AND SEVERE==1) [legacy preserved]"],
            "APS_JUSTIFIED": ["SEVERE==1 OR MDR_RISK==1"],
            "MRSA_JUSTIFIED": ["prior_mrsa_colonization==1"],
            "penalty_weights": {"carbapenem": 2, "antipseudomonal": 1, "anti_mrsa": 1},
            "delta_definition": "AI - Clinician (negative favors AI)",
            "primary_binary_definition": "any_violation = (carb_violation OR aps_violation OR mrsa_violation)",
        },
        "exploratory_models": {
            "enabled_with_flag": "--models",
            "delta_used": cfg.delta_for_models_col,
            "outcomes": [
                "nonzero_delta_guardrail = 1[delta!=0]",
                "ai_improved_guardrail = 1[delta<0]",
            ],
            "policy": [
                "Try multivariable Logit with HC3 robust SE.",
                "If MV fails (non-convergence / separation), fallback to univariable Logit HC3 per predictor.",
                "If UV Logit has perfect separation and predictor is binary, compute Fisher exact OR + CI (Haldane-Anscombe).",
                "Complete-case analysis on outcome + predictors; missingness exported.",
            ],
            "collinearity_hygiene": [
                "If SEVERE is included, omit its component markers in MV.",
                "If MDR_RISK is included, omit its component markers and acquisition_text in MV.",
            ],
        },
        "aware_addons": aware_meta,
    }


def _collect_outputs_for_manifest(out_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in sorted(out_dir.glob("*")):
        if p.is_file() and p.name != "analysis_manifest.json":
            files.append(p)
    return files


def write_manifest(out_dir: Path, xlsx: Path, sheets: Dict[str, pd.DataFrame], cfg: RunConfig) -> None:
    outputs = _collect_outputs_for_manifest(out_dir)
    payload = {
        "created_utc": now_iso(),
        "inputs_sha256": {"xlsx": sha256_file(xlsx)},
        "script_sha256": sha256_file(Path(__file__)),
        "abx_codes_sha256": sha256_df_stable(sheets["abx_codes"]) if "abx_codes" in sheets else None,
        "analysis_manifest_excluded_from_outputs": True,
        "outputs": [{"path": p.name, "sha256": sha256_file(p)} for p in outputs],
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": getattr(np, "__version__", "NA"),
            "pandas": getattr(pd, "__version__", "NA"),
            "scipy": getattr(scipy, "__version__", "NA"),
            "statsmodels": getattr(sm, "__version__", "NA") if sm is not None else "NA",
        },
        "config": {
            "primary_binary_clin_col": cfg.primary_binary_clin_col,
            "primary_binary_ai_col": cfg.primary_binary_ai_col,
            "key_secondary_delta_col": cfg.key_secondary_delta_col,
            "delta_for_hist_col": cfg.delta_for_hist_col,
            "delta_for_models_col": cfg.delta_for_models_col,
            "n_boot": cfg.n_boot,
            "seed": cfg.seed,
            "models": cfg.models,
            "clin_cov_col": cfg.clin_cov_col,
            "ai_cov_col": cfg.ai_cov_col,
            "admitting_ward_col": cfg.admitting_ward_col,
            "icu_transfer_24h_col": cfg.icu_transfer_24h_col,
            "fail_on_icu_mismatch": cfg.fail_on_icu_mismatch,
        },
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ----------------------------- figures -----------------------------

def plot_delta_hist(df: pd.DataFrame, col: str, out_dir: Path, fname: str) -> Optional[Path]:
    if col not in df.columns:
        return None
    x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return None
    fig = plt.figure()
    plt.hist(x, bins=25)
    plt.title(col)
    plt.xlabel("Delta (AI - Clinician)")
    plt.ylabel("Count")
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_primary_binary_2x2(df: pd.DataFrame, cfg: RunConfig, out_dir: Path) -> Optional[Path]:
    c1, c2 = cfg.primary_binary_clin_col, cfg.primary_binary_ai_col
    if c1 not in df.columns or c2 not in df.columns:
        return None
    a = to_int(df[c1]).to_numpy()
    b = to_int(df[c2]).to_numpy()
    n00 = int(((a == 0) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n11 = int(((a == 1) & (b == 1)).sum())

    fig = plt.figure()
    ax = plt.gca()
    mat = np.array([[n00, n01], [n10, n11]], dtype=int)
    im = ax.imshow(mat)  # default colormap
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["AI 0", "AI 1"]); ax.set_yticklabels(["Clin 0", "Clin 1"])
    ax.set_title("Primary binary paired 2x2: Any contextual violation")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path = out_dir / "figure_primary_binary_2x2.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------- main -----------------------------

def main() -> None:
    cfg = parse_args()
    safe_mkdir(cfg.out)

    sheets = load_workbook(cfg.xlsx)
    if "data" not in sheets or "endpoints" not in sheets:
        raise ValueError("Workbook must contain sheets: data, endpoints (optional: abx_codes).")

    merged, dup_cols = merge_endpoints_wins(sheets["data"], sheets["endpoints"])

    (cfg.out / "qc_duplicates_dropped.csv").write_text(
        pd.DataFrame({"duplicate_columns_dropped_from_data": dup_cols}).to_csv(index=False),
        encoding="utf-8",
    )

    merged = add_concordance(merged)
    merged = compute_contextual_guardrails(merged, cfg)
    merged = add_guardrail_composites(merged)
    merged = add_micro_evaluable_pair(merged, cfg.clin_cov_col, cfg.ai_cov_col)
    merged, aware_meta, _ = compute_aware_addons(merged, sheets)

    flow_counts(ensure_crt_id(sheets["data"]), merged).to_csv(cfg.out / "flow_counts.csv", index=False)

    write_json(cfg.out / "qc_checks.json", qc_checks(merged, cfg, dup_cols))

    # Hard stop if the necessary primary binary columns were not created
    if cfg.primary_binary_clin_col not in merged.columns or cfg.primary_binary_ai_col not in merged.columns:
        raise ValueError(f"Primary binary endpoint missing. Expected columns: {cfg.primary_binary_clin_col}, {cfg.primary_binary_ai_col}.")

    # Tables
    table2_primary_key_secondary(merged, cfg).to_csv(cfg.out / "table2_primary_secondary_endpoints.csv", index=False)
    table2_secondary_deltas(merged, cfg).to_csv(cfg.out / "table2_secondary_deltas.csv", index=False)
    table3_guardrails_components(merged).to_csv(cfg.out / "table3_guardrails_components.csv", index=False)

    t = table2b_endpoints_plus(merged, cfg)
    if not t.empty:
        t.to_csv(cfg.out / "table2b_endpoints_plus.csv", index=False)

    t = table3b_guardrails_composites(merged, cfg)
    if not t.empty:
        t.to_csv(cfg.out / "table3b_guardrails_composites.csv", index=False)

    t = tableS_micro_tradeoff_success(merged, cfg)
    if not t.empty:
        t.to_csv(cfg.out / "tableS_micro_tradeoff_success.csv", index=False)

    t = tableS_subgroup_lowrisk_guardrails(merged, cfg)
    if not t.empty:
        t.to_csv(cfg.out / "tableS_subgroup_lowrisk_guardrails.csv", index=False)

    t = tableS_guardrail_context_violations(merged)
    if not t.empty:
        t.to_csv(cfg.out / "tableS_guardrail_context_violations.csv", index=False)

    t = table4_coverage_paired(merged, cfg)
    if not t.empty:
        t.to_csv(cfg.out / "table4_coverage_paired.csv", index=False)

    t = tableS_concordance_summary(merged)
    if not t.empty:
        t.to_csv(cfg.out / "tableS_concordance_summary.csv", index=False)

    t = tableS_micro_evaluable_bias_qc(merged, cfg)
    if not t.empty:
        t.to_csv(cfg.out / "tableS_micro_evaluable_bias_qc.csv", index=False)

    # Exploratory models
    if cfg.models:
        status = run_exploratory_predictors(merged, cfg.out, cfg)
        (cfg.out / "models_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    # AWaRe outputs
    if aware_meta.get("aware_ran"):
        t = tableS_aware_binaries(merged)
        if not t.empty:
            t.to_csv(cfg.out / "tableS_aware_binaries.csv", index=False)
        t = tableS_aware_transitions(merged)
        if not t.empty:
            t.to_csv(cfg.out / "tableS_aware_transitions.csv", index=False)

        qc_rows = [
            {"arm": "clinician",
             "total_codes_nonempty_n": int(aware_meta.get("total_codes_clin_n", 0)),
             "unmapped_codes_nonempty_n": int(aware_meta.get("unmapped_codes_clin_n", 0)),
             "unmapped_rate_nonempty": aware_meta.get("unmapped_rate_clin", np.nan),
             "mapping_size": int(aware_meta.get("mapping_size", 0)),
             "aware_col": str(aware_meta.get("aware_col", ""))},
            {"arm": "ai",
             "total_codes_nonempty_n": int(aware_meta.get("total_codes_ai_n", 0)),
             "unmapped_codes_nonempty_n": int(aware_meta.get("unmapped_codes_ai_n", 0)),
             "unmapped_rate_nonempty": aware_meta.get("unmapped_rate_ai", np.nan),
             "mapping_size": int(aware_meta.get("mapping_size", 0)),
             "aware_col": str(aware_meta.get("aware_col", ""))},
        ]
        pd.DataFrame(qc_rows).to_csv(cfg.out / "tableS_aware_unmapped_qc.csv", index=False)

    # Rules JSON
    write_json(cfg.out / "addons_readme_rules.json", build_rules_json(cfg, aware_meta, dup_cols))

    # Figures
    plot_primary_binary_2x2(merged, cfg, cfg.out)
    plot_delta_hist(merged, cfg.delta_for_hist_col, cfg.out, "figure_delta_for_hist.png")
    # keep also histogram for the key secondary, even if different
    if cfg.key_secondary_delta_col != cfg.delta_for_hist_col:
        plot_delta_hist(merged, cfg.key_secondary_delta_col, cfg.out, "figure_key_secondary_hist.png")

    # Manifest
    write_manifest(cfg.out, cfg.xlsx, sheets, cfg)

    print(f"[OK] Outputs written to: {cfg.out.resolve()}")


if __name__ == "__main__":
    main()
