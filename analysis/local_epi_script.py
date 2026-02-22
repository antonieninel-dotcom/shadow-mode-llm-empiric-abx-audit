#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Epidemiology / WISCA-like Prior Builder (2020–2024)

Purpose (research only):
- Parse legacy, aggregated LIS antibiogram reports (often HTML->Excel) where "Denumire" contains block headers:
    <Specimen/label> - <Organism> (N antibiograme)
  followed by rows with antibiotics and S/I/R values.
- Produce audit-ready outputs:
    Bronze (long), Gold (aggregated), mandatory QC logs
    Priors with Jeffreys shrinkage + pooled 2020–2024 backstop
    Google-Sheets friendly lookup with prompt-ready epi_snippet per Ward×Year×Syndrome×Proxy specimen

Key design principles:
- Do NOT drop antibiotics: unmapped -> UNMAPPED::<cleaned>
- Robust S/I/R parsing for count / fraction / percent / binary encodings
- Do NOT force S+I+R=N: interpret p_sum<1 as "not tested"; p_sum>1 -> renormalize (logged)
- Block header must include "antibiogram" marker or "(N antibiograme)"
- Header anomalies (organism missing or organism maps to antibiotic) are logged and do NOT start a valid block
- Hard year bounds: ONLY keep data in [year_min, year_max] (default 2020–2024)

Author: rewritten for auditability and correctness (based on scriptfinal.py audit).
"""

from __future__ import annotations

import argparse
import hashlib
import datetime
import platform
import sys
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Reproducibility: manifest.json
# -----------------------------
def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def write_manifest_json(
    out_dir: Path,
    raw_dir: Path,
    script_path: Optional[Path] = None,
    args_dict: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a lightweight manifest.json for audit/reproducibility.

    Contents:
    - script hash (if provided)
    - CLI args (if provided)
    - output file list + sha256 + size
    - row/col counts for CSVs (best-effort)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"),
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "args": args_dict or {},
        "script": None,
        "outputs": [],
    }

    if script_path and script_path.exists():
        manifest["script"] = {
            "path": str(script_path),
            "sha256": _sha256_file(script_path),
            "bytes": script_path.stat().st_size,
        }

    # Hash all files in out_dir (non-recursive by default; recursive is safer if you create subfolders)
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file():
            continue
        entry: Dict[str, Any] = {
            "name": str(p.relative_to(out_dir)),
            "bytes": p.stat().st_size,
            "sha256": _sha256_file(p),
        }
        # Best-effort: add row/col counts for CSV
        if p.suffix.lower() == ".csv":
            try:
                df_head = pd.read_csv(p, nrows=5)
                entry["columns"] = df_head.columns.tolist()
                # cheap row count without loading full file
                with open(p, "rb") as f:
                    entry["rows"] = max(0, sum(1 for _ in f) - 1)
                entry["cols"] = int(df_head.shape[1])
            except Exception as e:
                entry["csv_meta_error"] = str(e)
        manifest["outputs"].append(entry)

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote {manifest_path}")
    return manifest_path

import numpy as np
import pandas as pd
from scipy.stats import beta

# ----------------------------- CONFIG ---------------------------------

DEFAULT_WARDS = ("ATI", "MI")
DEFAULT_SPECIMEN_GROUP = "other"

DEFAULT_YEAR_MIN = 2020
DEFAULT_YEAR_MAX = 2024

# Confidence labels (by-year cells)
THRESH_HIGH = 30
THRESH_MOD  = 10
THRESH_LOW  = 5
WIDE_CI_WIDTH = 0.30

# Jeffreys prior (robust for small n)
BETA_A = 0.5
BETA_B = 0.5

# In the Sheets lookup snippet: show at most K organisms per context
DEFAULT_MAX_ORGANISMS_IN_SNIPPET = 12

LIS_BAD_TOKENS = {
    "sensibil", "rezistent", "intermediar",
    "susceptible", "resistant", "intermediate",
    "s", "i", "r", "senzitiv", "sensitiv",
}

# ------------------------ Syndrome -> proxy ---------------------------

def _norm_key(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s

SYNDROME_PROXY_MAP: Dict[str, str] = {
    _norm_key("Community-acquired pneumonia"): "respiratory",
    _norm_key("COPD_IE"): "respiratory",
    _norm_key("Bloodstream infection / sepsis"): "blood",
    _norm_key("Urinary tract infection"): "urine",
    _norm_key("Urinary tract infection – pyelonephritis"): "urine",
    _norm_key("Urinary tract infection - pyelonephritis"): "urine",
    _norm_key("Skin and soft-tissue infection"): "wound",
    _norm_key("Intra-abdominal infection"): "sterile_fluid",
    _norm_key("Acute diarrheal disease"): "stool",
    _norm_key("Other infection"): "other",
}

SYNDROME_LIST_ORDERED: List[str] = [
    "Community-acquired pneumonia",
    "Bloodstream infection / sepsis",
    "Urinary tract infection",
    "Urinary tract infection – pyelonephritis",
    "COPD_IE",
    "Skin and soft-tissue infection",
    "Intra-abdominal infection",
    "Acute diarrheal disease",
    "Other infection",
]

def normalize_syndrome_name(x: object) -> str:
    s = "" if x is None else str(x)
    s = s.strip()
    # Keep your canonical list as-is when possible
    for syn in SYNDROME_LIST_ORDERED:
        if _norm_key(s) == _norm_key(syn):
            return syn
    # otherwise return cleaned original
    return re.sub(r"\s+", " ", s)

# ------------------------ Text Normalization --------------------------

def normalize_text_basic(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip().lower()
    s = s.replace("\xa0", " ")  # NBSP
    s = s.replace("\\", " ").replace("_", " ").replace("-", " ").replace("/", " ")
    s = re.sub(r"[^\w\s\+\(\)\.\%]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_float_safe(x: object) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        if np.isnan(v):
            return None
        return v
    s = str(x).strip()
    if s == "":
        return None
    s = s.replace(",", ".")
    try:
        v = float(s)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None

def clean_antibiotic(x: object) -> Optional[str]:
    s = normalize_text_basic(x)
    if not s:
        return None

    s = re.sub(r"^\s*\d+[\)\.\:]*\s*", "", s)  # leading numbering/bullets
    s = re.sub(r"\[[^\]]*\]", " ", s)          # strip bracketed MIC info
    s = re.sub(r"\bmicrograme\b|\bmcg\b|\bmg\b|\bg\b|\bml\b", " ", s)
    s = re.sub(r"\bcm[i|l]\b", " ", s)

    s = s.replace(".", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None

    if s in LIS_BAD_TOKENS:
        return None
    if re.fullmatch(r"(sensibil|rezistent|intermediar|susceptible|resistant|intermediate|s|i|r)\b", s):
        return None
    if re.match(r"^(sensibil|rezistent|intermediar|susceptible|resistant|intermediate)\b", s):
        return None

    return s

# ------------------------ Antibiotic Mapping --------------------------

ANTIBIOTIC_CANON_MAP: List[Tuple[str, str]] = [
    # combos first
    (r"\bceftazidim(e)?\b.*\bavibactam\b", "ceftazidime_avibactam"),
    (r"\bceftazidime\b.*\bavibactam\b", "ceftazidime_avibactam"),
    (r"\b(cza)\b", "ceftazidime_avibactam"),
    (r"\bceftolozan(e)?\b.*\btaz(o|obactam)?\b", "ceftolozane_tazobactam"),
    (r"\b(c/t|ctaz)\b", "ceftolozane_tazobactam"),
    (r"\bcefoperazon(a|ă|e)?\b.*\bsulb(actam)?\b", "cefoperazone_sulbactam"),
    (r"\b(cfp)\b.*\b(sul)\b", "cefoperazone_sulbactam"),
    (r"\bticarcillin(a|ă)?\b.*\bclav(ulan(ic|at)?)?\b", "ticarcillin_clavulanate"),
    (r"\bquinupristin(a|ă)?\b.*\bdalfopristin(a|ă)?\b", "quinupristin_dalfopristin"),

    (r"\bamoxicil+\w*\b.*\b(ac\.?\s*)?clav(ulan(ic|at(e)?)?)\b", "amoxicillin_clavulanate"),
    (r"\bampicillin\b.*\bsulb(actam)?\b", "ampicillin_sulbactam"),
    (r"\bpiperacill?in(a|ă)?\b.*\btaz(o|obactam)?\b", "piperacillin_tazobactam"),
    (r"\b(prl)\b|\bpiperacill?in(a|ă)?\b", "piperacillin"),

    # short codes
    (r"\b(amk)\b", "amikacin"),
    (r"\b(gen)\b", "gentamicin"),
    (r"\b(tob)\b", "tobramycin"),
    (r"\b(amp)\b", "ampicillin"),
    (r"\b(amx)\b", "amoxicillin"),
    (r"\b(amc)\b|\baugmentin\b|\bco\s*amoxiclav(um)?\b", "amoxicillin_clavulanate"),
    (r"\b(sam)\b", "ampicillin_sulbactam"),
    (r"\b(tzp)\b", "piperacillin_tazobactam"),

    (r"\b(czl)\b", "cefazolin"),
    (r"\b(cxm)\b", "cefuroxime"),
    (r"\b(ctx)\b", "cefotaxime"),
    (r"\b(cro)\b", "ceftriaxone"),
    (r"\b(caz)\b", "ceftazidime"),
    (r"\b(fep)\b", "cefepime"),
    (r"\b(cfx)\b", "cefoxitin"),

    (r"\b(etp)\b", "ertapenem"),
    (r"\b(ipm)\b", "imipenem_cilastatin"),
    (r"\b(mem)\b", "meropenem"),

    (r"\b(azt)\b", "aztreonam"),
    (r"\b(cip)\b", "ciprofloxacin"),
    (r"\b(lev)\b", "levofloxacin"),
    (r"\b(mxf)\b", "moxifloxacin"),

    (r"\b(cli)\b", "clindamycin"),
    (r"\b(lzd)\b", "linezolid"),
    (r"\b(van)\b", "vancomycin"),
    (r"\b(tei)\b", "teicoplanin"),
    (r"\b(mtz)\b", "metronidazole"),
    (r"\b(cst)\b", "colistin"),

    # full names
    (r"\bamikacin\b", "amikacin"),
    (r"\bgentamicin(a)?\b", "gentamicin"),
    (r"\btobramycin\b", "tobramycin"),
    (r"\bampicillin(a)?\b", "ampicillin"),
    (r"\bamoxicil(l)?in(a)?\b(?!\s*clav)", "amoxicillin"),

    (r"\bpenicillin(a|ă)?\b|\bpenicilina\s*g\b|\bbenzylpenicillin(a|ă)?\b", "penicillin"),
    (r"\bticarcillin(a|ă)?\b", "ticarcillin"),    
    (r"\bcefazolin\b", "cefazolin"),
    (r"\bcefuroxime\b", "cefuroxime"),
    (r"\bcefotaxime\b", "cefotaxime"),
    (r"\bceftriaxone\b", "ceftriaxone"),
    (r"\bceftazidim(e)?\b|\bceftazidime\b", "ceftazidime"),
    (r"\bcefepime\b", "cefepime"),
    (r"\bcefoxitin\b", "cefoxitin"),
    (r"\bertapenem\b|\binvanz\b", "ertapenem"),
    (r"\bimipenem\b", "imipenem_cilastatin"),
    (r"\bmeropenem\b", "meropenem"),
    (r"\baztreonam\b", "aztreonam"),
    (r"\bciprofloxacin\b", "ciprofloxacin"),
    (r"\blevofloxacin\b", "levofloxacin"),
    (r"\bmoxifloxacin\b", "moxifloxacin"),
    (r"\bclindamicin(a)?\b|\bclindamycin\b", "clindamycin"),
    (r"\blinezolid\b", "linezolid"),
    (r"\bvancomycin\b", "vancomycin"),
    (r"\bteicoplanin\b", "teicoplanin"),
    (r"\bcolistin\b|\bpolymyxin\s*e\b", "colistin"),
    (r"\bmetronidazol(e)?\b", "metronidazole"),
    (r"\bnitrofurantoin(a)?\b|\bfuradantin\b", "nitrofurantoin"),
    (r"\bfosfom(y|i)cin(a|ă)?\b|\bfosfomycin\b|\bfosfomicin\b|\bfosfomycin\s*trom(et)?amol\b", "fosfomycin"),
    (r"\btrimethoprim\b.*\bsulfa(methoxazol(e)?)?\b", "trimethoprim_sulfamethoxazole"),
    (r"\bsulfa(methoxazol(e)?)?\b.*\btrimethoprim\b", "trimethoprim_sulfamethoxazole"),
    (r"\bco[\s-]?trimoxazol(e)?\b|\bbactrim\b|\bsxt\b", "trimethoprim_sulfamethoxazole"),
    (r"\btigecyclin(e)?\b|\btigeciclin(a)?\b", "tigecycline"),
    (r"\boxacillin\b|\boxacilin(a)?\b", "oxacillin"),
    (r"\bery(th)?romycin\b|\beritromicin(a)?\b", "erythromycin"),
    (r"\bdoxiciclin(a|ă)?\b|\bdoxiciclina\b|\bdoxy\b", "doxycycline"),
    (r"\bazithromycin\b|\bazitromicin(a|ă)?\b", "azithromycin"),
    # Cefalosporine alte generații
    (r"\bcefixim(a|ă|e)?\b", "cefixime"),
    (r"\bcefiderocol\b", "cefiderocol"),

    # Fluorochinolone (altele)
    (r"\bofloxacin(a|ă)?\b", "ofloxacin"),
    (r"\bnorfloxacin(a|ă)?\b", "norfloxacin"),
    (r"\bpefloxacin(a|ă)?\b", "pefloxacin"),

    # Aminoglicozide (altele)
    (r"\bnetilmicin(a|ă)?\b", "netilmicin"),

    # Tetracicline (altele)
    (r"\btetraciclin(a|ă|e)?\b|\btetracyclin(e)?\b", "tetracycline"),
    (r"\bminociclin(a|ă|e)?\b|\bminocyclin(e)?\b", "minocycline"),

    # Altele
    (r"\bcloramfenicol\b|\bchloramphenicol\b", "chloramphenicol"),
    (r"\brifampicin(a|ă)?\b", "rifampicin"),
    (r"\bvoriconazol(e)?\b", "voriconazole"),

    # antifungals (kept; can be filtered downstream if desired)
    (r"\bfluconazol(e)?\b|\bfluconazole\b", "fluconazole"),
    (r"\banidulafungin\b", "anidulafungin"),
    (r"\bamfotericin(a)?\s*b\b|\bamphotericin\b", "amphotericin_b"),
    (r"\bmicafungin\b", "micafungin"),
    (r"\bcaspofungin\b", "caspofungin"),
    (r"\bposaconazol(e)?\b|\bposaconazole\b", "posaconazole"),
    (r"\bitraconazol(e)?\b|\bitraconazole\b", "itraconazole"),
    (r"\bflucytosin(e)?\b|\b5\s*fluorocytosin(e)?\b", "flucytosine"),
]

_COMPILED_ABX = [(re.compile(pat, flags=re.IGNORECASE), canon) for pat, canon in ANTIBIOTIC_CANON_MAP]

def map_antibiotic_to_canon(abx_clean: Optional[str]) -> Optional[str]:
    if not abx_clean:
        return None
    s = re.sub(r"\b(iv|po|im|oral)\b", " ", abx_clean, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    for rx, canon in _COMPILED_ABX:
        if rx.search(s):
            return canon
    return None

def normalize_antibiotic_name(x: object) -> Optional[str]:
    """Normalize a clinician code/name to the same canon as LIS mapping, when possible."""
    if x is None:
        return None
    s = clean_antibiotic(x)
    if not s:
        return None
    c = map_antibiotic_to_canon(s)
    return c if c is not None else s  # keep cleaned if not mapped

def pretty_abx(canon: str) -> str:
    s = str(canon).replace("UNMAPPED::", "UNMAPPED: ")
    s = s.replace("_", " ")
    s = s.replace(" tazobactam", "-tazobactam")
    s = s.replace(" avibactam", "-avibactam")
    s = s.replace(" sulfamethoxazole", "-sulfamethoxazole")
    return s.strip()

# ----------------------------- Specimen -------------------------------

SPECIMEN_CONTEXT_RULES: List[Tuple[str, str]] = [
    (r"\bscreening\b|\bportaj\b", "screening"),
    (r"\bcateter\b|\bsonda\b|\bvarf\b|\bvârf\b|\bdren\b|\bpericateter\b|\btub\s*dren\b", "device"),
]

SPECIMEN_REGEX_MAP: List[Tuple[str, str]] = [
    (r"\bhemocultur", "blood"),
    (r"\bsange\b|\bsânge\b", "blood"),
    (r"\burocultur", "urine"),
    (r"\burina\b|\burină\b", "urine"),

    (r"\bsecretie\s*trahe", "respiratory"),
    (r"\baspirat\s*trahe", "respiratory"),
    (r"\baspirat\s*bronsic\b", "respiratory"),
    (r"\bbronhoaspirat\b|\bbronho\s*aspirat\b", "respiratory"),
    (r"\blavaj\b|\bbal\b|\bbronhoalveolar", "respiratory"),
    (r"\bsputa\b|\bspută\b", "respiratory"),
    (r"\bexsudat\b.*\bbron(s|ș)ic\b", "respiratory"),

    (r"\bplag", "wound"),
    (r"\babc(e|e)s\b|\bpus\b", "wound"),
    (r"\bcolectie\s*purulent", "wound"),
    (r"\bfragment\s*tesut\b|\bfragment\s*țesut\b", "wound"),
    (r"\btampon\b.*\btegument\b", "wound"),

    (r"\bcopro\b|\bcoprocultur", "stool"),
    (r"\bscaun\b|\bfecal\b|\bmaterii\s*fecale\b", "stool"),

    (r"\blichid\s*cefalorahidian\b|\blcr\b|\bliquor\b", "sterile_fluid"),
    (r"\blichid\b.*\bpleural\b", "sterile_fluid"),
    (r"\bascit", "sterile_fluid"),
    (r"\bperitoneal\b", "sterile_fluid"),
    (r"\bbila\b|\bbilă\b", "sterile_fluid"),
    (r"\balte\s*lichide\b", "sterile_fluid"),
    (r"\blichid\b.*\bpericard", "sterile_fluid"),
    (r"\blichid\b.*\barticular", "sterile_fluid"),
    (r"\bdializ", "sterile_fluid"),
]

_COMPILED_SPEC = [(re.compile(pat, flags=re.IGNORECASE), grp) for pat, grp in SPECIMEN_REGEX_MAP]
_COMPILED_CTX  = [(re.compile(pat, flags=re.IGNORECASE), ctx) for pat, ctx in SPECIMEN_CONTEXT_RULES]

def specimen_context(specimen_raw: str) -> str:
    s = normalize_text_basic(specimen_raw)
    if not s:
        return "other"
    for rx, tag in _COMPILED_CTX:
        if rx.search(s):
            return tag
    return "diagnostic"

def map_specimen_group(specimen_raw: str) -> str:
    s = normalize_text_basic(specimen_raw)
    if not s:
        return DEFAULT_SPECIMEN_GROUP
    for rx, grp in _COMPILED_SPEC:
        if rx.search(s):
            return grp
    return DEFAULT_SPECIMEN_GROUP

# ----------------------------- Organism -------------------------------

def clean_organism(x: object) -> Optional[str]:
    s = normalize_text_basic(x)
    if not s:
        return None
    s = re.sub(r"\b(germene|germen|organism|nr)\b", " ", s)
    s = re.sub(r"\(\s*\d+\s*antibiogram(?:a|ă|e)?\s*\)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bculturi(\s+aerobe|\s+anaerobe)?\s+cu\s+antibiograma\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None

# ---------------------------- Report Parsing --------------------------

def find_header_row(df: pd.DataFrame) -> Optional[int]:
    dfn = df.copy()
    for c in dfn.columns:
        dfn[c] = dfn[c].map(normalize_text_basic)
    for i in range(len(dfn)):
        row = " ".join([str(x) for x in dfn.iloc[i].tolist() if str(x) != ""])
        if ("denumire" in row) and ("sensibil" in row) and ("intermediar" in row) and ("rezistent" in row):
            return i
    return None

def parse_block_header(denumire: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    raw = str(denumire).replace("\xa0", " ").strip()
    raw2 = raw.replace("–", "-").replace("—", "-")

    m = re.match(
        r"^\s*(.+?)\s*-\s*(.+?)\s*\(\s*(\d+)\s*antibiogram(?:a|ă|e)?\s*\)\s*$",
        raw2,
        flags=re.IGNORECASE
    )
    if m:
        return m.group(1).strip(), m.group(2).strip(), int(m.group(3))

    m = re.search(r"\(\s*(\d+)\s*antibiogram(?:a|ă|e)?\s*\)\s*$", raw2, flags=re.IGNORECASE)
    n = int(m.group(1)) if m else None

    if " - " in raw2:
        parts = raw2.split(" - ", 1)
        specimen = parts[0].strip()
        organism = re.sub(r"\(\s*\d+\s*antibiograme?\s*\)\s*$", "", parts[1], flags=re.IGNORECASE).strip()
        return specimen, organism, n

    return None, None, n

_RE_ANTBIO = re.compile(r"\(\s*\d+\s*antibiogram", re.IGNORECASE)
_RE_HAS_AB = re.compile(r"\bantibiogram", re.IGNORECASE)

def is_block_header_row(den: str, s: object, i: object, r: object) -> bool:
    den_raw = "" if den is None else str(den)
    den_s = normalize_text_basic(den_raw)
    if not den_s:
        return False

    if not ((to_float_safe(s) is None) and (to_float_safe(i) is None) and (to_float_safe(r) is None)):
        return False

    if den_s in {"denumire", "total"}:
        return False

    if not (_RE_ANTBIO.search(den_raw) or _RE_HAS_AB.search(den_raw)):
        return False

    return True

def is_ast_row(den: str, s: object, i: object, r: object) -> bool:
    den_s = normalize_text_basic(den)
    if not den_s:
        return False
    return any(to_float_safe(v) is not None for v in (s, i, r))

def period_from_filename(filename: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    base = Path(filename).stem
    ward = None
    mward = re.match(r"^([A-Za-z]+)[_\- ]", base)
    if mward:
        ward = mward.group(1).upper()

    m = re.search(r"(20\d{2})[_\- ]H([12])", base, flags=re.IGNORECASE)
    if m:
        y = int(m.group(1))
        half = f"H{m.group(2)}"
        return ward, y, half
    m2 = re.search(r"(20\d{2})", base)
    y = int(m2.group(1)) if m2 else None
    return ward, y, None

def extract_period_from_sheet(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    vals = df.astype(str).values.flatten().tolist()
    vals_norm = [v for v in (normalize_text_basic(v) for v in vals) if v]
    text = " | ".join(vals_norm[:600])
    m = re.search(
        r"(\b\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4}\b)\s*[-–]\s*(\b\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4}\b)",
        text,
    )
    if not m:
        return None, None, None
    start, end = m.group(1), m.group(2)
    y = None
    m2 = re.search(r"(\d{4})$", start)
    if m2:
        y = int(m2.group(1))
    return start, end, y

# ----------------------- S/I/R Conversion -----------------------------

def infer_ast_mode(sv: float, iv: float, rv: float, n_header: Optional[int]) -> str:
    vals = [v for v in (sv, iv, rv) if v is not None]
    if not vals:
        return "count"
    mx = max(vals)
    sm = sum(vals)

    if n_header == 1:
        uniq = set(vals)
        if uniq.issubset({0.0, 2.0}) and sm in (0.0, 2.0):
            return "two_coded_binary"

    if 0.0 <= mx <= 1.2 and 0.0 <= sm <= 1.35:
        return "fraction"
    if 0.0 <= mx <= 100.0 and 0.0 <= sm <= 100.5 and mx > 1.2:
        return "percent"
    return "count"

def counts_from_sir(
    s_raw: Optional[float],
    i_raw: Optional[float],
    r_raw: Optional[float],
    n_header: Optional[int],
) -> Tuple[int, int, int, int, Dict[str, Any]]:
    flags: Dict[str, Any] = {}

    sv = 0.0 if s_raw is None else float(s_raw)
    iv = 0.0 if i_raw is None else float(i_raw)
    rv = 0.0 if r_raw is None else float(r_raw)

    mode = infer_ast_mode(sv, iv, rv, n_header)
    flags["ast_mode"] = mode

    if mode == "two_coded_binary":
        sv, iv, rv = sv / 2.0, iv / 2.0, rv / 2.0
        mode = "fraction"
        flags["ast_mode"] = "fraction"
        flags["flag_two_coded_binary"] = True
    else:
        flags["flag_two_coded_binary"] = False

    if mode == "count":
        nS = int(round(sv))
        nI = int(round(iv))
        nR = int(round(rv))
        n_tested = nS + nI + nR
        flags.update({
            "p_S": np.nan, "p_I": np.nan, "p_R": np.nan,
            "p_sum": np.nan, "flag_incomplete_testing": False,
            "n_expected_tested": np.nan,
            "flag_missing_header_for_fraction": False,
            "flag_rounding_drift": 0,
            "flag_psum_gt_1_raw": np.nan,
            "flag_renormalized_psum": False,
        })
        return nS, nI, nR, n_tested, flags

    if n_header is None or n_header <= 0:
        flags["flag_missing_header_for_fraction"] = True
        if mode == "percent":
            flags["p_S"], flags["p_I"], flags["p_R"] = sv/100.0, iv/100.0, rv/100.0
            flags["p_sum"] = (sv+iv+rv)/100.0
        else:
            flags["p_S"], flags["p_I"], flags["p_R"] = sv, iv, rv
            flags["p_sum"] = sv + iv + rv
        flags["flag_incomplete_testing"] = np.nan
        flags["n_expected_tested"] = np.nan
        flags["flag_rounding_drift"] = np.nan
        flags["flag_psum_gt_1_raw"] = np.nan
        flags["flag_renormalized_psum"] = False
        return 0, 0, 0, 0, flags

    flags["flag_missing_header_for_fraction"] = False

    if mode == "percent":
        sv, iv, rv = sv / 100.0, iv / 100.0, rv / 100.0

    p_sum_raw = max(0.0, sv + iv + rv)
    flags["flag_psum_gt_1_raw"] = p_sum_raw if p_sum_raw > 1.0 else np.nan

    if p_sum_raw > 1.000001:
        scale = 1.0 / p_sum_raw
        sv, iv, rv = sv * scale, iv * scale, rv * scale
        flags["flag_renormalized_psum"] = True
        p_sum = 1.0
    else:
        flags["flag_renormalized_psum"] = False
        p_sum = p_sum_raw

    flags["p_S"], flags["p_I"], flags["p_R"] = sv, iv, rv
    flags["p_sum"] = p_sum
    flags["flag_incomplete_testing"] = (p_sum < 0.99)

    n_expected_tested = int(round(p_sum * n_header))
    n_expected_tested = min(max(n_expected_tested, 0), int(n_header))
    flags["n_expected_tested"] = n_expected_tested

    nS = int(round(sv * n_header))
    nI = int(round(iv * n_header))
    nR = int(round(rv * n_header))

    drift = (nS + nI + nR) - n_expected_tested
    if abs(drift) <= 1:
        flags["flag_rounding_drift"] = drift
        buckets = [("S", nS), ("I", nI), ("R", nR)]
        buckets.sort(key=lambda x: x[1], reverse=True)
        if buckets[0][0] == "S":
            nS -= drift
        elif buckets[0][0] == "I":
            nI -= drift
        else:
            nR -= drift
    else:
        flags["flag_rounding_drift"] = 0

    nS = max(nS, 0); nI = max(nI, 0); nR = max(nR, 0)
    n_tested = nS + nI + nR

    return nS, nI, nR, n_tested, flags

# ---------------------------- Workbook Parse ---------------------------

@dataclass
class ParseSummary:
    file: str
    ward: str
    year: Optional[int]
    half: Optional[str]
    sheet: str
    n_rows_emitted: int
    n_blocks_seen: int
    n_ast_rows_seen: int
    n_ast_rows_emitted: int
    n_skipped_no_block: int
    n_skipped_bad_antibiotic: int
    n_skipped_missing_organism: int
    n_skipped_missing_header: int
    n_header_anomalies: int

def parse_one_workbook(path: Path, ward_fallback: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[ParseSummary]]:
    rows: List[Dict[str, Any]] = []
    qc_all: List[Dict[str, Any]] = []
    qc_unmapped: List[Dict[str, Any]] = []
    qc_mapped: List[Dict[str, Any]] = []
    header_anoms: List[Dict[str, Any]] = []
    summaries: List[ParseSummary] = []

    ward_fn, year_fn, half_fn = period_from_filename(path.name)
    ward = (ward_fn or ward_fallback or "").upper() or ward_fallback

    xls = pd.ExcelFile(path)

    for sheet in xls.sheet_names:
        df0 = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object)

        period_start, period_end, year_sheet = extract_period_from_sheet(df0)
        year = year_sheet if year_sheet is not None else year_fn

        header_idx = find_header_row(df0)
        if header_idx is None:
            summaries.append(ParseSummary(
                file=path.name, ward=ward, year=year, half=half_fn, sheet=sheet,
                n_rows_emitted=0, n_blocks_seen=0, n_ast_rows_seen=0, n_ast_rows_emitted=0,
                n_skipped_no_block=0, n_skipped_bad_antibiotic=0, n_skipped_missing_organism=0,
                n_skipped_missing_header=1, n_header_anomalies=0,
            ))
            continue

        hdr = df0.iloc[header_idx].astype(str).tolist()
        hdr_norm = [normalize_text_basic(x) for x in hdr]

        def col_idx(name: str) -> Optional[int]:
            for j, v in enumerate(hdr_norm):
                if v == name:
                    return j
            for j, v in enumerate(hdr_norm):
                if name in v:
                    return j
            return None

        c_den = col_idx("denumire")
        c_s = col_idx("sensibil")
        c_i = col_idx("intermediar")
        c_r = col_idx("rezistent")
        if any(v is None for v in (c_den, c_s, c_i, c_r)):
            summaries.append(ParseSummary(
                file=path.name, ward=ward, year=year, half=half_fn, sheet=sheet,
                n_rows_emitted=0, n_blocks_seen=0, n_ast_rows_seen=0, n_ast_rows_emitted=0,
                n_skipped_no_block=0, n_skipped_bad_antibiotic=0, n_skipped_missing_organism=0,
                n_skipped_missing_header=1, n_header_anomalies=0,
            ))
            continue

        current_specimen: Optional[str] = None
        current_organism: Optional[str] = None
        current_n_header: Optional[int] = None

        n_blocks_seen = 0
        n_ast_rows_seen = 0
        n_ast_rows_emitted = 0
        n_skipped_no_block = 0
        n_skipped_bad_antibiotic = 0
        n_skipped_missing_organism = 0
        n_header_anomalies = 0

        for ridx in range(header_idx + 1, len(df0)):
            den = df0.iat[ridx, c_den]
            s = df0.iat[ridx, c_s]
            i = df0.iat[ridx, c_i]
            r = df0.iat[ridx, c_r]

            if den is None or str(den).strip() == "":
                continue

            if is_block_header_row(str(den), s, i, r):
                sp_raw, org_raw, n_hdr = parse_block_header(str(den))
                specimen = sp_raw if sp_raw else str(den).strip()

                org_clean_candidate = clean_organism(org_raw) if org_raw else None
                org_is_abx = False
                if org_clean_candidate:
                    org_is_abx = map_antibiotic_to_canon(org_clean_candidate) is not None

                if org_is_abx or (org_clean_candidate is None):
                    n_header_anomalies += 1
                    header_anoms.append({
                        "file": path.name,
                        "sheet": sheet,
                        "ward": ward,
                        "year": year,
                        "half": half_fn,
                        "period_start": period_start,
                        "period_end": period_end,
                        "header_text": str(den),
                        "specimen_raw": specimen,
                        "organism_raw": org_raw,
                        "n_antibiograme_header": n_hdr,
                        "issue": "organism_missing_or_is_antibiotic",
                    })
                    current_specimen = specimen
                    current_organism = None
                    current_n_header = n_hdr
                else:
                    current_specimen = specimen
                    current_organism = org_raw
                    current_n_header = n_hdr
                    n_blocks_seen += 1
                continue

            if not is_ast_row(str(den), s, i, r):
                continue

            n_ast_rows_seen += 1
            if current_specimen is None:
                n_skipped_no_block += 1
                continue

            abx_raw = str(den).replace("\xa0", " ").strip()
            abx_clean = clean_antibiotic(abx_raw)
            if not abx_clean:
                n_skipped_bad_antibiotic += 1
                continue

            antibiotic_canon = map_antibiotic_to_canon(abx_clean)

            qc_all.append({
                "file": path.name, "sheet": sheet, "ward": ward,
                "antibiotic_raw": abx_raw,
                "antibiotic_clean": abx_clean,
            })

            if antibiotic_canon is None:
                qc_unmapped.append({
                    "file": path.name, "sheet": sheet, "ward": ward,
                    "antibiotic_raw": abx_raw,
                    "antibiotic_clean": abx_clean,
                })
                antibiotic_canon = f"UNMAPPED::{abx_clean}"
            else:
                qc_mapped.append({
                    "file": path.name, "sheet": sheet, "ward": ward,
                    "antibiotic_raw": abx_raw,
                    "antibiotic_clean": abx_clean,
                    "antibiotic_canon": antibiotic_canon,
                })

            org_clean = clean_organism(current_organism) if current_organism else None
            if org_clean is None:
                n_skipped_missing_organism += 1
                continue

            sp_group = map_specimen_group(current_specimen)
            sp_ctx = specimen_context(current_specimen)

            s_raw = to_float_safe(s)
            i_raw = to_float_safe(i)
            r_raw = to_float_safe(r)

            nS, nI, nR, n_tested, extra = counts_from_sir(
                s_raw=s_raw, i_raw=i_raw, r_raw=r_raw, n_header=current_n_header
            )

            n_not_tested_est = None
            if current_n_header is not None and current_n_header >= 0:
                ne = extra.get("n_expected_tested", np.nan)
                if ne is None or (isinstance(ne, float) and np.isnan(ne)):
                    ne = n_tested
                n_not_tested_est = int(current_n_header - int(ne))

            rows.append({
                "ward": ward,
                "file": path.name,
                "sheet": sheet,
                "period_start": period_start,
                "period_end": period_end,
                "year": year,
                "half": half_fn,

                "specimen_raw": current_specimen,
                "specimen_type": sp_group,
                "specimen_context": sp_ctx,

                "organism_raw": current_organism,
                "organism_clean": org_clean,
                "n_antibiograme_header": current_n_header,

                "antibiotic_raw": abx_raw,
                "antibiotic_clean": abx_clean,
                "antibiotic_canon": antibiotic_canon,

                "s_raw": s_raw,
                "i_raw": i_raw,
                "r_raw": r_raw,

                "p_S": extra.get("p_S", np.nan),
                "p_I": extra.get("p_I", np.nan),
                "p_R": extra.get("p_R", np.nan),
                "p_sum": extra.get("p_sum", np.nan),
                "n_expected_tested": extra.get("n_expected_tested", np.nan),
                "ast_mode": extra.get("ast_mode", "count"),
                "flag_incomplete_testing": extra.get("flag_incomplete_testing", np.nan),
                "flag_missing_header_for_fraction": extra.get("flag_missing_header_for_fraction", False),
                "flag_rounding_drift": extra.get("flag_rounding_drift", 0),
                "flag_psum_gt_1_raw": extra.get("flag_psum_gt_1_raw", np.nan),
                "flag_renormalized_psum": extra.get("flag_renormalized_psum", False),

                "n_S": int(nS),
                "n_I": int(nI),
                "n_R": int(nR),
                "n_tested": int(n_tested),
                "n_not_tested_est": n_not_tested_est,
            })
            n_ast_rows_emitted += 1

        summaries.append(ParseSummary(
            file=path.name, ward=ward, year=year, half=half_fn, sheet=sheet,
            n_rows_emitted=n_ast_rows_emitted,
            n_blocks_seen=n_blocks_seen,
            n_ast_rows_seen=n_ast_rows_seen,
            n_ast_rows_emitted=n_ast_rows_emitted,
            n_skipped_no_block=n_skipped_no_block,
            n_skipped_bad_antibiotic=n_skipped_bad_antibiotic,
            n_skipped_missing_organism=n_skipped_missing_organism,
            n_skipped_missing_header=0,
            n_header_anomalies=n_header_anomalies,
        ))

    return (
        pd.DataFrame(rows),
        pd.DataFrame(qc_all),
        pd.DataFrame(qc_unmapped),
        pd.DataFrame(qc_mapped),
        pd.DataFrame(header_anoms),
        summaries,
    )

# ------------------------- Gold Aggregations ---------------------------

def compute_gold_tables(bronze: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    b = bronze.copy()
    b = b[b["n_tested"] > 0].copy()

    by_org = (
        b.groupby(["ward", "year", "specimen_type", "organism_clean", "antibiotic_canon"], dropna=False)[["n_S", "n_I", "n_R", "n_tested"]]
         .sum()
         .reset_index()
    )
    by_org["pct_S"] = by_org["n_S"] / by_org["n_tested"]
    by_org["pct_I"] = by_org["n_I"] / by_org["n_tested"]
    by_org["pct_R"] = by_org["n_R"] / by_org["n_tested"]
    by_org["pct_nonS"] = (by_org["n_I"] + by_org["n_R"]) / by_org["n_tested"]

    all_org = (
        b.groupby(["ward", "year", "specimen_type", "antibiotic_canon"], dropna=False)[["n_S", "n_I", "n_R", "n_tested"]]
         .sum()
         .reset_index()
    )
    all_org["pct_S"] = all_org["n_S"] / all_org["n_tested"]
    all_org["pct_I"] = all_org["n_I"] / all_org["n_tested"]
    all_org["pct_R"] = all_org["n_R"] / all_org["n_tested"]
    all_org["pct_nonS"] = (all_org["n_I"] + all_org["n_R"]) / all_org["n_tested"]

    return by_org, all_org

# ----------------------- Priors + Lookup (canonical) -------------------

def confidence_label(n_tested: int) -> str:
    if n_tested >= THRESH_HIGH:
        return "HIGH"
    if n_tested >= THRESH_MOD:
        return "MODERATE"
    if n_tested >= THRESH_LOW:
        return "LOW"
    return "VERY_SPARSE"

def add_beta_shrinkage_and_ci(df: pd.DataFrame, a: float = BETA_A, b: float = BETA_B) -> pd.DataFrame:
    out = df.copy()
    out["n_nonS"] = (out["n_tested"] - out["n_S"]).astype(int)

    out["pct_S_raw"] = out["n_S"] / out["n_tested"]
    out["pct_S_shrunk"] = (out["n_S"] + a) / (out["n_tested"] + a + b)

    out["ci95_low"] = beta.ppf(0.025, out["n_S"] + a, out["n_nonS"] + b)
    out["ci95_high"] = beta.ppf(0.975, out["n_S"] + a, out["n_nonS"] + b)
    out["ci95_width"] = out["ci95_high"] - out["ci95_low"]
    out["wide_ci_flag"] = out["ci95_width"] >= WIDE_CI_WIDTH

    out["confidence_label"] = out["n_tested"].map(lambda x: confidence_label(int(x)))
    return out

def build_by_year_priors_with_pooled_backstop(
    gold_by_org: pd.DataFrame,
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    g = gold_by_org.copy()
    g = g[(g["year"] >= year_min) & (g["year"] <= year_max)].copy()
    g = g[g["n_tested"] > 0].copy()

    by_year = add_beta_shrinkage_and_ci(g)

    pooled = (
        g.groupby(["ward", "specimen_type", "organism_clean", "antibiotic_canon"], dropna=False)[["n_S", "n_I", "n_R", "n_tested"]]
         .sum()
         .reset_index()
    )
    pooled = add_beta_shrinkage_and_ci(pooled)
    pooled = pooled.rename(columns={
        "n_tested": "pooled_n_tested_2020_2024",
        "n_S": "pooled_n_S_2020_2024",
        "n_I": "pooled_n_I_2020_2024",
        "n_R": "pooled_n_R_2020_2024",
        "pct_S_shrunk": "pooled_pct_S_shrunk_2020_2024",
        "ci95_low": "pooled_ci95_low_2020_2024",
        "ci95_high": "pooled_ci95_high_2020_2024",
        "confidence_label": "pooled_confidence_label_2020_2024",
        "wide_ci_flag": "pooled_wide_ci_flag_2020_2024",
    })

    keep_cols = [
        "ward","specimen_type","organism_clean","antibiotic_canon",
        "pooled_n_tested_2020_2024","pooled_n_S_2020_2024","pooled_n_I_2020_2024","pooled_n_R_2020_2024",
        "pooled_pct_S_shrunk_2020_2024","pooled_ci95_low_2020_2024","pooled_ci95_high_2020_2024",
        "pooled_confidence_label_2020_2024","pooled_wide_ci_flag_2020_2024",
    ]
    pooled = pooled[keep_cols].copy()

    merged = by_year.merge(
        pooled,
        on=["ward","specimen_type","organism_clean","antibiotic_canon"],
        how="left",
        validate="many_to_one"
    )
    return merged

def fmt_prob(p: float) -> str:
    return f"{100.0*p:.1f}%"

def build_qc_rates_for_snippet(bronze: pd.DataFrame, ps: pd.DataFrame, year_min: int, year_max: int) -> pd.DataFrame:
    b = bronze.copy()
    b = b[(b["year"] >= year_min) & (b["year"] <= year_max)].copy()

    ren = (
        b.groupby(["ward","year"], dropna=False)["flag_renormalized_psum"]
         .mean()
         .reset_index()
         .rename(columns={"flag_renormalized_psum": "qc_renormalized_rate"})
    )

    ps2 = ps.copy()
    ps2 = ps2[(ps2["year"].notna())].copy()
    ps2["year"] = ps2["year"].astype(int)
    ps2 = ps2[(ps2["year"] >= year_min) & (ps2["year"] <= year_max)].copy()

    ha = (
        ps2.groupby(["ward","year"], dropna=False)[["n_header_anomalies","n_blocks_seen"]]
           .sum()
           .reset_index()
    )
    ha["qc_header_anomaly_rate"] = np.where(
        ha["n_blocks_seen"] > 0,
        ha["n_header_anomalies"] / ha["n_blocks_seen"],
        0.0
    )

    # Unmapped rate derived from bronze for this ward-year
    unmap = (
        b.groupby(["ward","year"], dropna=False)["antibiotic_canon"]
         .apply(lambda s: float(np.mean(pd.Series(s).astype(str).str.startswith("UNMAPPED::"))))
         .reset_index()
         .rename(columns={"antibiotic_canon":"qc_unmapped_rate"})
    )

    out = ha.merge(ren, on=["ward","year"], how="left").merge(unmap, on=["ward","year"], how="left")
    out["qc_renormalized_rate"] = out["qc_renormalized_rate"].fillna(0.0)
    out["qc_unmapped_rate"] = out["qc_unmapped_rate"].fillna(0.0)
    return out[["ward","year","qc_header_anomaly_rate","qc_renormalized_rate","qc_unmapped_rate"]]

def build_epi_lookup_by_year(
    priors: pd.DataFrame,
    wards: List[str],
    year_min: int,
    year_max: int,
    max_organisms: int,
    qc_rates: pd.DataFrame,
    include_all_fallback: bool = True,
) -> pd.DataFrame:
    """
    Canonical Google-Sheets/LLM lookup table with prompt-ready epi_snippet.

    Rows:
      ward ∈ wards (+ optional ALL)
      year ∈ [year_min, year_max]
      syndrome ∈ SYNDROME_LIST_ORDERED
      specimen_proxy = mapping(syndrome)
      epi_snippet = multi-line text (bounded by your Sheet cell limit; typically OK)
    """
    pri = priors.copy()
    all_rows: List[Dict[str, Any]] = []

    def make_snippet(df_key: pd.DataFrame, ward: str, year: int, syndrome: str, proxy: str, qc_row: Optional[pd.Series]) -> str:
        qc_header = float(qc_row["qc_header_anomaly_rate"]) if qc_row is not None else 0.0
        qc_reno = float(qc_row["qc_renormalized_rate"]) if qc_row is not None else 0.0
        qc_unmap = float(qc_row["qc_unmapped_rate"]) if qc_row is not None else 0.0

        lines = [
            "LOCAL EPIDEMIOLOGY PRIOR (WISCA-like proxy; research/shadow-mode only)",
            f"- Ward: {ward} ; Year: {year} ; Syndrome: {syndrome} ; Proxy specimen: {proxy}",
            "- Source: aggregate LIS susceptibility reports (no deduplication / breakpoints metadata; panel variability possible).",
            f"- QC (ward-year): header_anomaly_rate={qc_header:.3f} ; renormalized_rate={qc_reno:.3f} ; unmapped_rate={qc_unmap:.3f}",
        ]

        if df_key.empty:
            lines.append("No local antibiogram data available for this ward-year-proxy. Treat as 'no prior'.")
            lines.append("Guardrail: Use guidelines first; do not infer coverage from missing data.")
            return "\n".join(lines)

        org_rank = (
            df_key.groupby("organism_clean", dropna=False)["n_tested"]
                  .sum()
                  .reset_index()
                  .sort_values(["n_tested","organism_clean"], ascending=[False, True])
        )
        top_orgs = org_rank["organism_clean"].head(max_organisms).tolist()
        lines.append("Top organisms by total tested isolates (sum n_tested across antibiotics):")
        lines.append("  " + "; ".join([f"{o} (n={int(org_rank.loc[org_rank['organism_clean']==o,'n_tested'].iloc[0])})" for o in top_orgs]))
        lines.append("Organism-specific susceptibility (S%; Jeffreys-shrunk with CI95; year-specific + pooled 2020–2024 backstop when sparse):")

        for org in top_orgs:
            sub = df_key[df_key["organism_clean"] == org].copy()
            sub = sub.sort_values(["pct_S_shrunk","n_tested","antibiotic_canon"], ascending=[False, False, True])
            org_total = int(sub["n_tested"].sum())
            lines.append(f"* {org} (sum n_tested={org_total})")
            for _, r in sub.iterrows():
                abx = pretty_abx(str(r["antibiotic_canon"]))
                n = int(r["n_tested"])
                lab = str(r["confidence_label"])
                wide = " WIDE_CI" if bool(r["wide_ci_flag"]) else ""
                yr_part = f"{fmt_prob(float(r['pct_S_shrunk']))} ({fmt_prob(float(r['ci95_low']))}–{fmt_prob(float(r['ci95_high']))}), n={n}, {lab}{wide}"

                pooled_txt = ""
                if lab != "HIGH" and pd.notna(r.get("pooled_n_tested_2020_2024", np.nan)):
                    pn = int(r["pooled_n_tested_2020_2024"])
                    if pn > 0:
                        pooled_txt = (
                            f" | pooled 2020–2024: "
                            f"{fmt_prob(float(r['pooled_pct_S_shrunk_2020_2024']))} "
                            f"({fmt_prob(float(r['pooled_ci95_low_2020_2024']))}–{fmt_prob(float(r['pooled_ci95_high_2020_2024']))}), n={pn}"
                        )

                lines.append(f"  - {abx}: S≈{yr_part}{pooled_txt}")

        lines.append("Guardrail: Use this as prior evidence only. Treat LOW/VERY_SPARSE or WIDE_CI as uncertain; prefer guideline-concordant regimens; do not over-weight small differences.")
        return "\n".join(lines)

    for ward in wards:
        for year in range(year_min, year_max + 1):
            qc_row = qc_rates[(qc_rates["ward"] == ward) & (qc_rates["year"] == year)]
            qc_row_s = qc_row.iloc[0] if len(qc_row) else None
            for syndrome in SYNDROME_LIST_ORDERED:
                proxy = SYNDROME_PROXY_MAP.get(_norm_key(syndrome), "other")
                df_key = pri[(pri["ward"] == ward) & (pri["year"] == year) & (pri["specimen_type"] == proxy)].copy()
                snippet = make_snippet(df_key, ward=ward, year=year, syndrome=syndrome, proxy=proxy, qc_row=qc_row_s)
                all_rows.append({
                    "ward": ward,
                    "year": year,
                    "syndrome": syndrome,
                    "syndrome_norm": normalize_syndrome_name(syndrome),
                    "specimen_proxy": proxy,
                    "epi_snippet": snippet,
                    "snippet_char_count": len(snippet),
                })

    if include_all_fallback:
        pooled_year = (
            pri.groupby(["year","specimen_type","organism_clean","antibiotic_canon"], dropna=False)[["n_S","n_I","n_R","n_tested"]]
               .sum()
               .reset_index()
        )
        pooled_year = add_beta_shrinkage_and_ci(pooled_year)
        pooled_year["ward"] = "ALL"

        pooled_backstop_all = (
            pooled_year.groupby(["specimen_type","organism_clean","antibiotic_canon"], dropna=False)[["n_S","n_I","n_R","n_tested"]]
                      .sum().reset_index()
        )
        pooled_backstop_all = add_beta_shrinkage_and_ci(pooled_backstop_all)
        pooled_backstop_all = pooled_backstop_all.rename(columns={
            "n_tested": "pooled_n_tested_2020_2024",
            "n_S": "pooled_n_S_2020_2024",
            "n_I": "pooled_n_I_2020_2024",
            "n_R": "pooled_n_R_2020_2024",
            "pct_S_shrunk": "pooled_pct_S_shrunk_2020_2024",
            "ci95_low": "pooled_ci95_low_2020_2024",
            "ci95_high": "pooled_ci95_high_2020_2024",
            "confidence_label": "pooled_confidence_label_2020_2024",
            "wide_ci_flag": "pooled_wide_ci_flag_2020_2024",
        })
        pooled_backstop_all = pooled_backstop_all[["specimen_type","organism_clean","antibiotic_canon",
                                                   "pooled_n_tested_2020_2024","pooled_n_S_2020_2024","pooled_n_I_2020_2024","pooled_n_R_2020_2024",
                                                   "pooled_pct_S_shrunk_2020_2024","pooled_ci95_low_2020_2024","pooled_ci95_high_2020_2024",
                                                   "pooled_confidence_label_2020_2024","pooled_wide_ci_flag_2020_2024"]]
        pooled_year = pooled_year.merge(pooled_backstop_all, on=["specimen_type","organism_clean","antibiotic_canon"], how="left", validate="many_to_one")

        qc_all = qc_rates.groupby(["year"], dropna=False)[["qc_header_anomaly_rate","qc_renormalized_rate","qc_unmapped_rate"]].max().reset_index()
        qc_all["ward"] = "ALL"

        for year in range(year_min, year_max + 1):
            qc_row = qc_all[(qc_all["ward"] == "ALL") & (qc_all["year"] == year)]
            qc_row_s = qc_row.iloc[0] if len(qc_row) else None
            for syndrome in SYNDROME_LIST_ORDERED:
                proxy = SYNDROME_PROXY_MAP.get(_norm_key(syndrome), "other")
                df_key = pooled_year[(pooled_year["year"] == year) & (pooled_year["specimen_type"] == proxy)].copy()
                snippet = make_snippet(df_key, ward="ALL", year=year, syndrome=syndrome, proxy=proxy, qc_row=qc_row_s)
                all_rows.append({
                    "ward": "ALL",
                    "year": year,
                    "syndrome": syndrome,
                    "syndrome_norm": normalize_syndrome_name(syndrome),
                    "specimen_proxy": proxy,
                    "epi_snippet": snippet,
                    "snippet_char_count": len(snippet),
                })

    return pd.DataFrame(all_rows)

# ------------------- Compact lookup for prompts (shortlist) -------------------

def evidence_grade_ab(total_tests: int, panel_cov: float) -> str:
    """
    Context-level evidence grade for an antibiotic in a ward-year-proxy context.
    Defaults are conservative and reviewer-friendly (tunable).
    """
    if total_tests >= 100 and panel_cov >= 0.70:
        return "HIGH"
    if total_tests >= 50 and panel_cov >= 0.60:
        return "MODERATE"
    if total_tests >= 20 and panel_cov >= 0.50:
        return "LOW"
    return "VERY_SPARSE"

def build_org_weights(
    gold_by_org: pd.DataFrame,
    ward: str,
    year: int,
    specimen_proxy: str,
    top_n: int = 8,
) -> Dict[str, float]:
    """
    Weight organisms by sum(n_tested) within ward-year-proxy.
    Returns dict of top_n organisms + an 'oth' bucket so weights sum to 1.
    """
    sub = gold_by_org[(gold_by_org["ward"] == ward) & (gold_by_org["year"] == year) & (gold_by_org["specimen_type"] == specimen_proxy)].copy()
    sub = sub[sub["n_tested"] > 0].copy()
    if sub.empty:
        return {"oth": 1.0}
    w = sub.groupby("organism_clean")["n_tested"].sum().reset_index(name="w_raw")
    w = w.sort_values(["w_raw","organism_clean"], ascending=[False, True])
    total = float(w["w_raw"].sum())
    if total <= 0:
        return {"oth": 1.0}
    w["w"] = w["w_raw"] / total
    top = w.head(top_n)
    rest = float(w["w"].iloc[top_n:].sum()) if len(w) > top_n else 0.0
    out = {str(r["organism_clean"]): float(r["w"]) for _, r in top.iterrows()}
    if rest > 1e-9:
        out["oth"] = rest
    else:
        out["oth"] = max(0.0, 1.0 - float(sum(out.values())))
    s = float(sum(out.values()))
    if s > 0:
        out = {k: v/s for k, v in out.items()}
    return out

def build_epi_compact_from_shortlist_rowset(
    ward: str,
    year: int,
    syndrome: str,
    specimen_proxy: str,
    org_w: Dict[str, float],
    shortlist_ctx: pd.DataFrame,
    qc_row: Optional[pd.Series],
    max_abx: int = 12,
) -> str:
    """
    Produce a minimal, high-signal EPI block for LLM prompts.
    Uses shortlist (top-per-class) with WISCA conservative stats, plus organism weights.
    """
    qc_header = float(qc_row["qc_header_anomaly_rate"]) if qc_row is not None else 0.0
    qc_reno = float(qc_row["qc_renormalized_rate"]) if qc_row is not None else 0.0
    qc_unmap = float(qc_row["qc_unmapped_rate"]) if qc_row is not None else 0.0

    org_parts = []
    for k, v in org_w.items():
        kk = re.sub(r"\s+", "_", str(k))[:24]
        org_parts.append(f"{kk}:{v:.2f}")
    org_str = ",".join(org_parts)

    lines = []
    lines.append(f"EPI{{w={ward},y={int(year)},s={syndrome},px={specimen_proxy}}}")
    lines.append(f"QC{{hdr={qc_header:.3f},reno={qc_reno:.3f},unmap={qc_unmap:.3f}}}")
    lines.append(f"ORG{{{org_str}}}")

    if shortlist_ctx is None or shortlist_ctx.empty:
        lines.append("ABX{}")
        lines.append("RULE{prior=P(S); mode=prior-only; no_local_shortlist; rely_on_guidelines; do_not_infer_from_missing}")
        return "\n".join(lines)

    df = shortlist_ctx.copy()
    df["evidence_grade"] = df.apply(lambda r: evidence_grade_ab(int(r["total_tests"]), float(r["coverage_frac"])), axis=1)
    df = df.sort_values(["score_conservative","coverage_frac","total_tests"], ascending=[False, False, False]).head(max_abx)

    abx_parts = []
    for _, r in df.iterrows():
        abx = str(r["antibiotic_canon"])
        abx_tok = abx.replace("UNMAPPED::", "UNMAPPED_").replace("_", "-")
        cls = str(r.get("abx_class", "other"))
        abx_parts.append(
            f"{abx_tok}:{float(r['wisca_mean']):.2f}|p05={float(r['wisca_p05']):.2f}|cov={float(r['coverage_frac']):.2f}|n={int(r['total_tests'])}|G={str(r['evidence_grade'])[0]}|C={cls}"
        )
    lines.append("ABX{" + ";".join(abx_parts) + "}")
    lines.append("RULE{prior=P(S); mode=prior-only; rank=p05_then_cov; if(cov<0.60 or G=V) treat_unreliable; avoid_overweighting_small_diffs}")
    return "\n".join(lines)

def build_epi_lookup_shortlist_by_year(
    gold_by_org: pd.DataFrame,
    shortlist_df: pd.DataFrame,
    qc_rates: pd.DataFrame,
    year_min: int,
    year_max: int,
    include_all_fallback: bool = True,
    top_org: int = 8,
    max_abx: int = 12,
) -> pd.DataFrame:
    """
    Google-Sheets import table: ward×year×syndrome×proxy -> epi_compact block.
    Requires WISCA shortlist to be available.
    """
    rows = []
    qc_rates = qc_rates.copy()
    for (ward, year, syndrome, proxy), g in shortlist_df.groupby(["ward","year","syndrome_norm","specimen_proxy"]):
        year = int(year)
        if year < year_min or year > year_max:
            continue
        qc_row = qc_rates[(qc_rates["ward"] == ward) & (qc_rates["year"] == year)]
        qc_row_s = qc_row.iloc[0] if len(qc_row) else None

        org_w = build_org_weights(gold_by_org, ward=ward, year=year, specimen_proxy=proxy, top_n=top_org)
        epi_compact = build_epi_compact_from_shortlist_rowset(
            ward=ward, year=year, syndrome=syndrome, specimen_proxy=proxy,
            org_w=org_w, shortlist_ctx=g, qc_row=qc_row_s, max_abx=max_abx
        )
        rows.append({
            "ward": ward,
            "year": year,
            "syndrome_norm": syndrome,
            "specimen_proxy": proxy,
            "epi_compact": epi_compact,
            "compact_char_count": len(epi_compact),
        })

    wards = sorted(shortlist_df["ward"].dropna().astype(str).unique().tolist())
    for ward in wards:
        for year in range(year_min, year_max + 1):
            qc_row = qc_rates[(qc_rates["ward"] == ward) & (qc_rates["year"] == year)]
            qc_row_s = qc_row.iloc[0] if len(qc_row) else None
            for syndrome in SYNDROME_LIST_ORDERED:
                proxy = SYNDROME_PROXY_MAP.get(_norm_key(syndrome), "other")
                exists = any((r["ward"] == ward and r["year"] == year and r["syndrome_norm"] == syndrome and r["specimen_proxy"] == proxy) for r in rows)
                if exists:
                    continue
                org_w = build_org_weights(gold_by_org, ward=ward, year=year, specimen_proxy=proxy, top_n=top_org)
                epi_compact = build_epi_compact_from_shortlist_rowset(
                    ward=ward, year=year, syndrome=syndrome, specimen_proxy=proxy,
                    org_w=org_w, shortlist_ctx=None, qc_row=qc_row_s, max_abx=max_abx
                )
                rows.append({
                    "ward": ward,
                    "year": year,
                    "syndrome_norm": syndrome,
                    "specimen_proxy": proxy,
                    "epi_compact": epi_compact,
                    "compact_char_count": len(epi_compact),
                })

    out = pd.DataFrame(rows)
    out["syndrome"] = out["syndrome_norm"]
    out = out[["ward","year","syndrome","syndrome_norm","specimen_proxy","epi_compact","compact_char_count"]]
    out = out.sort_values(["ward","year","syndrome_norm","specimen_proxy"], ascending=[True, True, True, True]).reset_index(drop=True)
    return out

# ----------------------------- WISCA scoring ---------------------------
# Note: optional; kept correct and bounded to 2020–2024.

ABX_CLASS_MAP: Dict[str, str] = {
    "ampicillin": "penicillin",
    "amoxicillin": "penicillin",
    "amoxicillin_clavulanate": "bl_bli",
    "ampicillin_sulbactam": "bl_bli",
    "piperacillin": "antipseudomonal_penicillin",
    "piperacillin_tazobactam": "bl_bli_antipseudomonal",
    "cefazolin": "1gc",
    "cefuroxime": "2gc",
    "ceftriaxone": "3gc",
    "cefotaxime": "3gc",
    "ceftazidime": "3gc_antipseudomonal",
    "cefepime": "4gc",
    "aztreonam": "monobactam",
    "ertapenem": "carbapenem",
    "imipenem_cilastatin": "carbapenem",
    "meropenem": "carbapenem",
    "amikacin": "aminoglycoside",
    "gentamicin": "aminoglycoside",
    "tobramycin": "aminoglycoside",
    "ciprofloxacin": "fluoroquinolone",
    "levofloxacin": "fluoroquinolone",
    "moxifloxacin": "fluoroquinolone",
    "vancomycin": "glycopeptide",
    "teicoplanin": "glycopeptide",
    "linezolid": "oxazolidinone",
    "clindamycin": "lincosamide",
    "erythromycin": "macrolide",
    "doxycycline": "tetracycline",
    "tigecycline": "glycylcycline",
    "trimethoprim_sulfamethoxazole": "folate_inhibitor",
    "metronidazole": "nitroimidazole",
    "ceftazidime_avibactam": "bl_bli_reserve",
    "ceftolozane_tazobactam": "bl_bli_reserve",
    "cefoperazone_sulbactam": "bl_bli_antipseudomonal",
          "ticarcillin_clavulanate": "bl_bli_antipseudomonal",
          "ticarcillin": "antipseudomonal_penicillin",
          "penicillin": "penicillin",
          "cefixime": "3gc",
          "cefiderocol": "siderophore_cephalosporin",
          "ofloxacin": "fluoroquinolone",
          "norfloxacin": "fluoroquinolone",
          "pefloxacin": "fluoroquinolone",
          "netilmicin": "aminoglycoside",
          "tetracycline": "tetracycline",
          "minocycline": "tetracycline",
          "chloramphenicol": "phenicol",
          "rifampicin": "rifamycin",
          "voriconazole": "azole_antifungal",
           "quinupristin_dalfopristin": "streptogramin",
}

def get_abx_class(antibiotic_canon: str) -> str:
    abx = str(antibiotic_canon).strip().lower()
    return ABX_CLASS_MAP.get(abx, "other")

def load_clinical_antibiotics(
    clinical_xlsx_path: str,
    syndrome_col: str = "syndrome_text",
    abx_cols: Tuple[str, str, str] = ("clin_abx_code_1", "clin_abx_code_2", "clin_abx_code_3"),
) -> pd.DataFrame:
    df = pd.read_excel(clinical_xlsx_path)
    missing = [c for c in [syndrome_col, *abx_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Clinical cohort missing columns: {missing}. Found: {list(df.columns)}")
    long = (
        df[[syndrome_col, *abx_cols]]
        .melt(id_vars=[syndrome_col], value_vars=list(abx_cols), var_name="abx_slot", value_name="abx_raw")
        .dropna(subset=["abx_raw"])
    )
    long["abx_raw"] = long["abx_raw"].astype(str).str.strip()
    long = long[long["abx_raw"] != ""]
    long["antibiotic_canon"] = long["abx_raw"].map(normalize_antibiotic_name)
    long["syndrome_norm"] = long[syndrome_col].map(normalize_syndrome_name)
    return long

def clinician_abx_frequency(clin_long: pd.DataFrame) -> pd.DataFrame:
    freq = (
        clin_long.groupby(["syndrome_norm", "antibiotic_canon"], dropna=False)
        .size()
        .reset_index(name="clin_n")
    )
    freq["abx_class"] = freq["antibiotic_canon"].map(get_abx_class)
    return freq

def wisca_scores_for_context(
    gold_by_org: pd.DataFrame,
    ward: str,
    year: int,
    specimen_proxy: str,
    n_mc: int = 400,
    seed: int = 1337,
    min_total_tests: int = 20,
) -> pd.DataFrame:
    """
    WISCA-like success score for each antibiotic in a ward-year-specimen context.
    Uses organism weights proportional to sum(n_tested) (proxy frequency).
    Score draw = Σ w_org * p(S|org,abx), where p drawn from Beta(nS+0.5, nNonS+0.5).
    """
    rng = np.random.default_rng(seed)
    sub = gold_by_org[(gold_by_org["ward"] == ward) & (gold_by_org["year"] == year) & (gold_by_org["specimen_type"] == specimen_proxy)].copy()
    sub = sub[sub["n_tested"] > 0].copy()
    if sub.empty:
        return pd.DataFrame(columns=["ward","year","specimen_proxy","antibiotic_canon","abx_class","total_tests","coverage_frac","wisca_mean","wisca_p05","wisca_p95"])

    w = sub.groupby("organism_clean")["n_tested"].sum().reset_index(name="org_weight_raw")
    totw = float(w["org_weight_raw"].sum())
    w["org_weight"] = np.where(totw > 0, w["org_weight_raw"]/totw, 0.0)
    w_map = dict(zip(w["organism_clean"], w["org_weight"]))

    tot = sub.groupby("antibiotic_canon")["n_tested"].sum().reset_index(name="total_tests")
    tot = tot[tot["total_tests"] >= min_total_tests].copy()
    if tot.empty:
        return pd.DataFrame(columns=["ward","year","specimen_proxy","antibiotic_canon","abx_class","total_tests","coverage_frac","wisca_mean","wisca_p05","wisca_p95"])

    out_rows: List[Dict[str, Any]] = []
    for abx in tot["antibiotic_canon"].astype(str).tolist():
        abx_sub = sub[sub["antibiotic_canon"].astype(str) == abx].copy()
        abx_sub["org_weight"] = abx_sub["organism_clean"].map(w_map).fillna(0.0)
        coverage_frac = float(abx_sub["org_weight"].sum())
        if coverage_frac <= 0:
            continue

        alpha = (abx_sub["n_S"].astype(float) + BETA_A).to_numpy()
        beta_b = ((abx_sub["n_tested"].astype(float) - abx_sub["n_S"].astype(float)) + BETA_B).to_numpy()
        weights = abx_sub["org_weight"].to_numpy()

        draws = rng.beta(alpha, beta_b, size=(n_mc, len(alpha)))
        score_draws = draws.dot(weights)

        out_rows.append({
            "ward": ward,
            "year": int(year),
            "specimen_proxy": specimen_proxy,
            "antibiotic_canon": abx,
            "abx_class": get_abx_class(abx),
            "total_tests": int(tot.loc[tot["antibiotic_canon"].astype(str)==abx, "total_tests"].iloc[0]),
            "coverage_frac": coverage_frac,
            "wisca_mean": float(np.mean(score_draws)),
            "wisca_p05": float(np.quantile(score_draws, 0.05)),
            "wisca_p95": float(np.quantile(score_draws, 0.95)),
        })

    return pd.DataFrame(out_rows).sort_values(["wisca_p05","coverage_frac","total_tests"], ascending=False)

def build_wisca_score_table(
    gold_by_org: pd.DataFrame,
    epi_lookup_df: pd.DataFrame,
    min_total_tests: int = 20,
    n_mc: int = 400,
    seed: int = 1337,
) -> pd.DataFrame:
    rows = []
    ctx_cols = ["ward","year","syndrome_norm","specimen_proxy"]
    for _, r in epi_lookup_df[ctx_cols].drop_duplicates().iterrows():
        ward = str(r["ward"])
        year = int(r["year"])
        proxy = str(r["specimen_proxy"])
        sc = wisca_scores_for_context(
            gold_by_org=gold_by_org,
            ward=ward,
            year=year,
            specimen_proxy=proxy,
            n_mc=n_mc,
            seed=seed,
            min_total_tests=min_total_tests,
        )
        if sc.empty:
            continue
        sc = sc.copy()
        sc["syndrome_norm"] = str(r["syndrome_norm"])
        rows.append(sc)
    if not rows:
        return pd.DataFrame(columns=["ward","year","syndrome_norm","specimen_proxy","antibiotic_canon","abx_class","total_tests","coverage_frac","wisca_mean","wisca_p05","wisca_p95"])
    return pd.concat(rows, ignore_index=True)

def shortlist_antibiotics_by_class(
    wisca_long: pd.DataFrame,
    clin_freq: Optional[pd.DataFrame] = None,
    top_per_class: int = 2,
    max_total: int = 12,
    min_total_tests: int = 30,
    min_clin_uses: int = 3,
) -> pd.DataFrame:
    df = wisca_long.copy()
    df["score_conservative"] = df["wisca_p05"] * df["coverage_frac"]

    if clin_freq is not None and not clin_freq.empty:
        cf = clin_freq.rename(columns={"antibiotic_canon":"antibiotic_canon"})[["syndrome_norm","antibiotic_canon","clin_n"]].rename(columns={"antibiotic_canon":"antibiotic_canon"})
        df = df.merge(cf, on=["syndrome_norm","antibiotic_canon"], how="left")
        df["clin_n"] = df["clin_n"].fillna(0).astype(int)
        df["is_candidate"] = (df["total_tests"] >= min_total_tests) | (df["clin_n"] >= min_clin_uses)
    else:
        df["clin_n"] = 0
        df["is_candidate"] = df["total_tests"] >= min_total_tests

    df = df[df["is_candidate"]].copy()

    out_rows = []
    grp_cols = ["ward","year","syndrome_norm","specimen_proxy"]
    for ctx, g in df.groupby(grp_cols):
        g = g.sort_values(["score_conservative","total_tests"], ascending=False)
        selected = []
        for cls, gg in g.groupby("abx_class"):
            gg = gg.sort_values(["score_conservative","total_tests"], ascending=False)
            selected.extend(gg.head(top_per_class).to_dict("records"))

        best = {}
        for rec in selected:
            k = rec["antibiotic_canon"]
            if (k not in best) or (rec["score_conservative"] > best[k]["score_conservative"]):
                best[k] = rec
        final = list(best.values())
        final.sort(key=lambda x: (x["score_conservative"], x["total_tests"]), reverse=True)
        final = final[:max_total]

        for rec in final:
            out_rows.append({
                **{c:v for c,v in zip(grp_cols, ctx)},
                "antibiotic_canon": rec["antibiotic_canon"],
                "abx_class": rec["abx_class"],
                "total_tests": int(rec["total_tests"]),
                "coverage_frac": float(rec["coverage_frac"]),
                "wisca_mean": float(rec["wisca_mean"]),
                "wisca_p05": float(rec["wisca_p05"]),
                "wisca_p95": float(rec["wisca_p95"]),
                "score_conservative": float(rec["score_conservative"]),
                "clin_n": int(rec.get("clin_n", 0)),
            })
    return pd.DataFrame(out_rows)

def export_shortlist_json(shortlist_df: pd.DataFrame, out_path: str) -> None:
    payload: Dict[str, Any] = {}
    for (ward,year,syndrome), g in shortlist_df.groupby(["ward","year","syndrome_norm"]):
        key = f"{ward}|{int(year)}|{syndrome}"
        payload[key] = {
            "ward": ward,
            "year": int(year),
            "syndrome": syndrome,
            "specimen_proxy": str(g["specimen_proxy"].iloc[0]),
            "antibiotics": [
                {
                    "name": str(r["antibiotic_canon"]),
                    "class": str(r["abx_class"]),
                    "wisca_mean": float(r["wisca_mean"]),
                    "wisca_p05": float(r["wisca_p05"]),
                    "wisca_p95": float(r["wisca_p95"]),
                    "coverage_frac": float(r["coverage_frac"]),
                    "total_tests": int(r["total_tests"]),
                    "clin_n": int(r.get("clin_n", 0)),
                }
                for _, r in g.sort_values(["score_conservative"], ascending=False).iterrows()
            ]
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# ----------------------------- Main -----------------------------------

def discover_files(raw_dir: Path, wards: Iterable[str], exclude_basenames: Optional[set] = None) -> List[Tuple[str, Path]]:
    """Discover LIS .xlsx inputs. If ward subfolders exist, use them; else recurse.
    Excludes any file whose basename is in exclude_basenames (e.g., clinical cohort file).
    """
    exclude_basenames = exclude_basenames or set()
    out: List[Tuple[str, Path]] = []
    for ward in wards:
        wdir = raw_dir / ward
        if wdir.exists():
            for fp in sorted(wdir.glob("*.xlsx")):
                if fp.name in exclude_basenames:
                    continue
                out.append((ward, fp))
    if out:
        return out
    for fp in sorted(raw_dir.rglob("*.xlsx")):
        if fp.name in exclude_basenames:
            continue
        out.append(("", fp))
    return out

def run_pipeline(
    raw_dir: Path,
    out_dir: Path,
    wards: Iterable[str],
    year_min: int,
    year_max: int,
    max_organisms_snippet: int,
    include_all_fallback: bool,
    enable_wisca: bool,
    wisca_min_total_tests: int,
    wisca_mc: int,
    shortlist_top_per_class: int,
    shortlist_max_total: int,
    shortlist_min_total_tests: int,
    shortlist_min_clin_uses: int,
    clinical_cohort: Optional[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[pd.DataFrame] = []
    qca_all: List[pd.DataFrame] = []
    qcu_all: List[pd.DataFrame] = []
    qcm_all: List[pd.DataFrame] = []
    han_all: List[pd.DataFrame] = []
    summaries: List[ParseSummary] = []

    exclude = set()
    if clinical_cohort is not None:
        exclude.add(Path(clinical_cohort).name)
    files = discover_files(raw_dir, wards, exclude_basenames=exclude)
    print(f"[INFO] Discovered {len(files)} .xlsx files under {raw_dir}")

    for ward, fp in files:
        print(f"[INFO] Parsing: {fp.name} (ward={ward or 'AUTO'})")
        df, qca, qcu, qcm, han, sums = parse_one_workbook(fp, ward_fallback=ward)
        summaries.extend(sums)

        if not df.empty:
            all_rows.append(df)
        qca_all.append(qca if not qca.empty else pd.DataFrame(columns=["file","sheet","ward","antibiotic_raw","antibiotic_clean"]))
        qcu_all.append(qcu if not qcu.empty else pd.DataFrame(columns=["file","sheet","ward","antibiotic_raw","antibiotic_clean"]))
        qcm_all.append(qcm if not qcm.empty else pd.DataFrame(columns=["file","sheet","ward","antibiotic_raw","antibiotic_clean","antibiotic_canon"]))
        han_all.append(han if not han.empty else pd.DataFrame(columns=[
            "file","sheet","ward","year","half","period_start","period_end","header_text","specimen_raw","organism_raw","n_antibiograme_header","issue"
        ]))

    if not all_rows:
        raise SystemExit("No data parsed. Check folder structure and report format.")

    bronze = pd.concat(all_rows, ignore_index=True)
    bronze = bronze.dropna(subset=["year"]).copy()
    bronze["year"] = bronze["year"].astype(int)

    # HARD YEAR BOUNDS (only 2020–2024 by default)
    before = len(bronze)
    bronze = bronze[(bronze["year"] >= year_min) & (bronze["year"] <= year_max)].copy()
    after = len(bronze)
    print(f"[INFO] Year filter [{year_min},{year_max}]: kept {after}/{before} rows (dropped {before-after})")

    # QC: organism_is_antibiotic_rate (per file/sheet) - should be ~0 after header anomaly fix
    canon_set = set([c for c in bronze["antibiotic_canon"].dropna().unique()
                     if isinstance(c, str) and not c.startswith("UNMAPPED::")])
    bronze["flag_organism_is_antibiotic"] = bronze["organism_clean"].isin(canon_set)
    (
        bronze.groupby(["ward","year","file","sheet"], dropna=False)["flag_organism_is_antibiotic"]
              .mean()
              .reset_index()
              .rename(columns={"flag_organism_is_antibiotic":"rate_organism_is_antibiotic"})
    ).to_csv(out_dir / "qc_organism_is_antibiotic_rate.csv", index=False)

    bronze.to_csv(out_dir / "bronze_long_all_abx.csv", index=False)
    print(f"[OK] Wrote {out_dir / 'bronze_long_all_abx.csv'} ({len(bronze)} rows)")

    pd.concat(qca_all, ignore_index=True).to_csv(out_dir / "qc_antibiotics_all_cleaned.csv", index=False)
    pd.concat(qcu_all, ignore_index=True).to_csv(out_dir / "qc_antibiotics_unmapped.csv", index=False)
    pd.concat(qcm_all, ignore_index=True).to_csv(out_dir / "qc_antibiotics_mapped.csv", index=False)
    pd.concat(han_all, ignore_index=True).to_csv(out_dir / "qc_header_anomalies.csv", index=False)

    ps = pd.DataFrame([s.__dict__ for s in summaries])
    # Keep only bounded years for parsing summary exports (makes QC internally consistent)
    ps2 = ps.copy()
    ps2 = ps2[ps2["year"].notna()].copy()
    ps2["year"] = ps2["year"].astype(int)
    ps2 = ps2[(ps2["year"] >= year_min) & (ps2["year"] <= year_max)].copy()
    ps2.to_csv(out_dir / "qc_parsing_summary_by_file.csv", index=False)

    # GOLD
    gold_by_org, gold_all = compute_gold_tables(bronze)

    # Optional: create pooled ward='ALL' rows for contexts (used as fallback in lookups/WISCA).
    # This is NOT patient-level deduplicated; it simply pools MI+ATI sample-based aggregates.
    if include_all_fallback:
        try:
            base_wards = [str(w) for w in wards]
            gb = gold_by_org[gold_by_org["ward"].astype(str).isin(base_wards)].copy()
            if not gb.empty:
                group_cols = [c for c in ["year","specimen_type","organism_clean","antibiotic_canon"] if c in gb.columns]
                agg_cols = [c for c in ["n_S","n_I","n_R","n_tested"] if c in gb.columns]
                gb_all = gb.groupby(group_cols, dropna=False)[agg_cols].sum().reset_index()
                gb_all["ward"] = "ALL"
                # Recompute pct columns for ALL
                if "n_tested" in gb_all.columns:
                    gb_all["pct_S"] = gb_all["n_S"] / gb_all["n_tested"]
                    gb_all["pct_I"] = gb_all["n_I"] / gb_all["n_tested"]
                    gb_all["pct_R"] = gb_all["n_R"] / gb_all["n_tested"]
                    gb_all["pct_nonS"] = (gb_all["n_I"] + gb_all["n_R"]) / gb_all["n_tested"]
                # Ensure canonical column order compatibility
                for c in gold_by_org.columns:
                    if c not in gb_all.columns:
                        gb_all[c] = pd.NA
                gold_by_org = pd.concat([gold_by_org, gb_all[gold_by_org.columns]], ignore_index=True)
                print(f"[OK] Added pooled ward='ALL' rows to gold_by_org ({len(gb_all)} rows)")
        except Exception as e:
            print("[WARN] Could not build ward='ALL' pooled gold_by_org:", e)
    gold_by_org.to_csv(out_dir / "gold_antibiogram_by_organism.csv", index=False)
    gold_all.to_csv(out_dir / "gold_antibiogram_all_organisms.csv", index=False)

    xlsx_path = out_dir / "gold_antibiogram.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        gold_by_org.to_excel(xw, sheet_name="by_organism", index=False)
        gold_all.to_excel(xw, sheet_name="all_organisms", index=False)
    print(f"[OK] Wrote {xlsx_path}")
    print("[DONE] Bronze + Gold + QC created (bounded to years).")

    # PRIORS + LOOKUP (Google Sheets import target)
    priors = build_by_year_priors_with_pooled_backstop(gold_by_org, year_min=year_min, year_max=year_max)
    priors_path = out_dir / f"epi_prior_rows_by_organism_{year_min}_{year_max}_by_year.csv"
    priors.to_csv(priors_path, index=False)
    print(f"[OK] Wrote {priors_path} ({len(priors)} rows)")

    qc_rates = build_qc_rates_for_snippet(bronze=bronze, ps=ps2, year_min=year_min, year_max=year_max)
    wards_list = [w.strip().upper() for w in wards if w.strip()]
    epi_lookup = build_epi_lookup_by_year(
        priors=priors,
        wards=wards_list,
        year_min=year_min,
        year_max=year_max,
        max_organisms=max_organisms_snippet,
        qc_rates=qc_rates,
        include_all_fallback=include_all_fallback,
    )
    lookup_path = out_dir / "EPI_LOOKUP_BY_ORGANISM_BY_YEAR.csv"
    epi_lookup.to_csv(lookup_path, index=False, encoding="utf-8")
    print(f"[OK] Wrote {lookup_path} ({len(epi_lookup)} rows)")

    # OPTIONAL WISCA/shortlists (also bounded by year because epi_lookup is bounded)
    if enable_wisca:
        try:
            wisca_long = build_wisca_score_table(
                gold_by_org=gold_by_org,
                epi_lookup_df=epi_lookup,
                min_total_tests=wisca_min_total_tests,
                n_mc=wisca_mc,
                seed=1337,
            )
            wisca_path = out_dir / "epi_antibiotic_scores_long.csv"
            wisca_long.to_csv(wisca_path, index=False)
            print(f"[OK] Wrote {wisca_path} ({len(wisca_long)} rows)")

            clin_freq = None
            if clinical_cohort:
                clin_long = load_clinical_antibiotics(clinical_cohort)
                clin_freq = clinician_abx_frequency(clin_long)
                clin_path = out_dir / "clinical_antibiotic_frequency_by_syndrome.csv"
                clin_freq.to_csv(clin_path, index=False)
                print(f"[OK] Wrote {clin_path} ({len(clin_freq)} rows)")

            shortlist = shortlist_antibiotics_by_class(
                wisca_long=wisca_long,
                clin_freq=clin_freq,
                top_per_class=shortlist_top_per_class,
                max_total=shortlist_max_total,
                min_total_tests=shortlist_min_total_tests,
                min_clin_uses=shortlist_min_clin_uses,
            )
            shortlist_path = out_dir / "epi_shortlist_by_class.csv"
            shortlist.to_csv(shortlist_path, index=False)
            print(f"[OK] Wrote {shortlist_path} ({len(shortlist)} rows)")

            json_path = out_dir / "local_epi_wisca_shortlist.json"
            export_shortlist_json(shortlist, str(json_path))

            # Compact prompt lookup (best signal-per-character), for Google Sheets
            try:
                epi_compact_lookup = build_epi_lookup_shortlist_by_year(
                    gold_by_org=gold_by_org,
                    shortlist_df=shortlist,
                    qc_rates=qc_rates,
                    year_min=year_min,
                    year_max=year_max,
                    include_all_fallback=include_all_fallback,
                    top_org=8,
                    max_abx=12,
                )
                compact_path = out_dir / "EPI_LOOKUP_SHORTLIST_BY_YEAR.csv"
                epi_compact_lookup.to_csv(compact_path, index=False, encoding="utf-8")
                print(f"[OK] Wrote {compact_path} ({len(epi_compact_lookup)} rows)")
            except Exception as ee:
                print("[WARN] Could not write compact shortlist lookup:", ee)
            print(f"[OK] Wrote {json_path}")

        except Exception as e:
            print("[WARN] Could not compute WISCA/shortlists:", e)

    # Always write a reproducibility manifest (best-effort)
    try:
        args_dict = {
            "wards": list(wards),
            "year_min": int(year_min),
            "year_max": int(year_max),
            "max_organisms_snippet": int(max_organisms_snippet),
            "include_all_fallback": bool(include_all_fallback),
            "enable_wisca": bool(enable_wisca),
            "wisca_min_total_tests": int(wisca_min_total_tests),
            "wisca_mc": int(wisca_mc),
            "shortlist_top_per_class": int(shortlist_top_per_class),
            "shortlist_max_total": int(shortlist_max_total),
            "shortlist_min_total_tests": int(shortlist_min_total_tests),
            "shortlist_min_clin_uses": int(shortlist_min_clin_uses),
            "clinical_cohort": str(clinical_cohort) if clinical_cohort else None,
        }
        write_manifest_json(out_dir=out_dir, raw_dir=raw_dir, script_path=Path(__file__), args_dict=args_dict)
    except Exception as me:
        print("[WARN] Could not write manifest.json:", me)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Parse legacy hospital antibiogram reports into bronze/gold tables + by-year priors + Google Sheets lookup (2020–2024)."
    )
    ap.add_argument("--raw", type=str, default="raw", help="Input root folder containing ward subfolders (ATI/MI) with .xlsx files.")
    ap.add_argument("--out", type=str, default="outputs_local_epi", help="Output folder.")

    ap.add_argument("--wards", type=str, default=",".join(DEFAULT_WARDS), help="Comma-separated ward folder names.")
    ap.add_argument("--year-min", type=int, default=DEFAULT_YEAR_MIN, help="Min year (hard filter).")
    ap.add_argument("--year-max", type=int, default=DEFAULT_YEAR_MAX, help="Max year (hard filter).")

    ap.add_argument("--max-organisms-snippet", type=int, default=DEFAULT_MAX_ORGANISMS_IN_SNIPPET, help="Top organisms to print per epi_snippet.")
    ap.add_argument("--include-all-fallback", action="store_true", help="Also write ward=ALL fallback rows in lookup.")

    # Optional WISCA outputs
    ap.add_argument("--enable-wisca", action="store_true", help="Also compute WISCA-like antibiotic scores and shortlists.")
    ap.add_argument("--wisca-min-total-tests", type=int, default=20, help="Min total AST tests (across organisms) to score an antibiotic.")
    ap.add_argument("--wisca-mc", type=int, default=400, help="Monte Carlo draws per antibiotic score.")
    ap.add_argument("--shortlist-top-per-class", type=int, default=2, help="Shortlist: how many antibiotics per class.")
    ap.add_argument("--shortlist-max-total", type=int, default=12, help="Shortlist: cap total antibiotics per context.")
    ap.add_argument("--shortlist-min-total-tests", type=int, default=30, help="Shortlist: min total tests for AST-based inclusion.")
    ap.add_argument("--shortlist-min-clin-uses", type=int, default=3, help="Shortlist: if clinical cohort provided, min clinician uses to force-in an antibiotic.")

    ap.add_argument("--clinical-cohort", default=None, help="Optional: clinical cohort XLSX (syndrome_text + clin_abx_code_1/2/3).")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    wards = [w.strip() for w in args.wards.split(",") if w.strip()]

    run_pipeline(
        raw_dir=raw_dir,
        out_dir=out_dir,
        wards=wards,
        year_min=int(args.year_min),
        year_max=int(args.year_max),
        max_organisms_snippet=int(args.max_organisms_snippet),
        include_all_fallback=bool(args.include_all_fallback),
        enable_wisca=bool(args.enable_wisca),
        wisca_min_total_tests=int(args.wisca_min_total_tests),
        wisca_mc=int(args.wisca_mc),
        shortlist_top_per_class=int(args.shortlist_top_per_class),
        shortlist_max_total=int(args.shortlist_max_total),
        shortlist_min_total_tests=int(args.shortlist_min_total_tests),
        shortlist_min_clin_uses=int(args.shortlist_min_clin_uses),
        clinical_cohort=args.clinical_cohort,
    )

if __name__ == "__main__":
    main()