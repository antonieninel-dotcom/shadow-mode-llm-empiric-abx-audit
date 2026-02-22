#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_openai_api.py

Batch runner for retrospective LLM evaluation (offline/shadow mode).
- Reads inputs/case_input.csv with columns: case_id, ai_user_prompt_text, ai_system_prompt
- Reads inputs/abx_codes.csv with columns: abx_code, generic_name_en, (optional) abx_canon
  * abx_code should include route-aware codes (e.g., CRO_IV, AMC1G_PO)
  * NO_ANTIBIOTIC must be present as an abx_code and is handled as a sentinel
- Calls OpenAI Responses API with STRICT JSON schema output (Structured Outputs)
- Writes outputs/ai_results_for_sheets.csv plus append-only logs in logs/

Windows-friendly. Python >= 3.8.
"""

import argparse
import asyncio
import csv
import json
import os
import time
import random
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

# -------------------------
# Structured Outputs schema (STRICT)
# -------------------------
JSON_SCHEMA = {
    "name": "abx_reco",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ai_recommended_regimen_text": {"type": "string"},
            "ai_abx_code_1": {"type": "string"},
            "ai_abx_code_2": {"type": "string"},
            "ai_abx_code_3": {"type": "string"},
        },
        "required": [
            "ai_recommended_regimen_text",
            "ai_abx_code_1",
            "ai_abx_code_2",
            "ai_abx_code_3",
        ],
    },
    "strict": True,
}

# -------------------------
# Stewardship flags (deterministic by generic)
# NOTE: Normalize generics to underscore form (/, - -> _).
# Adjust sets if you want a different operational definition.
# -------------------------
ANTIPSEUDOMONAL_GENERIC = {
    "piperacillin_tazobactam", "piperacillin", "cefepime", "ceftazidime",
    "ceftolozane_tazobactam", "ceftazidime_avibactam",
    "meropenem", "imipenem","imipenem_cilastatin", "doripenem",
    "aztreonam", "ciprofloxacin", "levofloxacin", "cefiderocol",
    "amikacin", "gentamicin", "tobramycin",
}
CARBAPENEM_GENERIC = {"meropenem", "imipenem", "imipenem_cilastatin", "doripenem", "ertapenem"}
ANTI_MRSA_GENERIC = {"vancomycin", "linezolid", "daptomycin", "ceftaroline", "tedizolid", "teicoplanin", "doxycycline", "minocycline", "clindamycin", "quinupristin_dalfopristin", "rifampin"}

TRANSIENT_HTTP = {429, 500, 502, 503, 504}

# -------------------------
# Helpers
# -------------------------
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def read_manifest_created_date(manifest_path: str) -> str:
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    created = m.get("created_utc") or m.get("created_at") or m.get("created")
    if not created:
        raise ValueError("manifest.json missing created_utc (or created_at/created).")
    return str(created)[:10]  # YYYY-MM-DD

def _read_csv_dict(path: str, encodings=("utf-8-sig", "utf-8", "latin1")) -> List[Dict[str, str]]:
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                r = csv.DictReader(f)
                if not r.fieldnames:
                    raise ValueError(f"{os.path.basename(path)} has no header.")
                return list(r)
        except Exception as e:
            last_err = e
    raise last_err  # type: ignore

def normalize_case_id(cid_raw: str) -> str:
    cid_raw = (cid_raw or "").strip()
    if not cid_raw:
        return ""
    try:
        if "." in cid_raw:
            return str(int(float(cid_raw)))
    except Exception:
        pass
    return cid_raw

def normalize_generic(g: str) -> str:
    g = (g or "").strip().lower()
    g = g.replace("/", "_").replace("-", "_").replace(" ", "_")
    while "__" in g:
        g = g.replace("__", "_")
    return g

def load_cases_from_case_input(case_input_csv: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    rows = _read_csv_dict(case_input_csv)
    required = {"case_id", "ai_user_prompt_text", "ai_system_prompt"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"case_input.csv missing columns: {sorted(missing)}")

    out: List[Dict[str, str]] = []
    for row in rows:
        cid = normalize_case_id(row.get("case_id", ""))
        if not cid:
            continue

        up = row.get("ai_user_prompt_text") or ""
        sp = row.get("ai_system_prompt") or ""

        if not up.strip():
            raise ValueError(f"Empty ai_user_prompt_text for case_id={cid}")
        if not sp.strip():
            raise ValueError(f"Empty ai_system_prompt for case_id={cid}")

        out.append({"case_id": cid, "user_prompt": up, "system_prompt": sp})
        if limit is not None and len(out) >= limit:
            break

    if not out:
        raise ValueError("No valid cases found (after filtering empty case_id rows).")
    return out

def load_abx_codes(abx_csv: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    rows = _read_csv_dict(abx_csv)
    required = {"abx_code", "generic_name_en"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"abx_codes.csv missing columns: {sorted(missing)}")

    has_abx_canon = "abx_canon" in rows[0].keys()

    code2gen: Dict[str, str] = {}
    canon2codes: Dict[str, List[str]] = {}

    for row in rows:
        code = (row.get("abx_code") or "").strip()
        gen = (row.get("generic_name_en") or "").strip()
        if not code:
            continue

        code_u = code.upper()
        gen_norm = normalize_generic(gen)
        code2gen[code_u] = gen_norm

        if has_abx_canon:
            canon = normalize_generic(row.get("abx_canon") or "")
            if canon:
                canon2codes.setdefault(canon, []).append(code_u)

    if not code2gen:
        raise ValueError("abx_codes.csv mapping is empty (after filtering blanks).")

    if "NO_ANTIBIOTIC" not in code2gen:
        print("[WARN] NO_ANTIBIOTIC not present in abx_codes.csv. Add it for a fully data-driven vocabulary.")

    return code2gen, canon2codes

def resolve_to_abx_code(raw: str, code2gen: Dict[str, str], canon2codes: Dict[str, List[str]]) -> str:
    s = (raw or "").strip()
    if not s:
        return ""

    s_up = s.upper()
    if s_up in {"NA", "N/A", "NONE", "NULL"}:
        return ""

    if s_up == "NO_ANTIBIOTIC":
        return "NO_ANTIBIOTIC"

    if s_up in code2gen:
        return s_up

    s_canon = normalize_generic(s)
    if s_canon in canon2codes:
        candidates = canon2codes[s_canon]
        for c in candidates:
            if c.endswith("_IV"):
                return c
        return candidates[0]

    return s_up  # unresolved (kept for debugging)

def compute_flags(codes: List[str], code2gen: Dict[str, str]) -> Tuple[int, int, int]:
    gens = []
    for c in codes:
        c = (c or "").strip().upper()
        if not c or c == "NO_ANTIBIOTIC":
            continue
        g = code2gen.get(c, "")
        if g:
            gens.append(g)

    anti_pseudo = int(any(g in ANTIPSEUDOMONAL_GENERIC for g in gens))
    carb = int(any(g in CARBAPENEM_GENERIC for g in gens))
    anti_mrsa = int(any(g in ANTI_MRSA_GENERIC for g in gens))
    return anti_pseudo, carb, anti_mrsa

def already_ok(out_csv: str) -> set:
    if not os.path.exists(out_csv):
        return set()
    done = set()
    with open(out_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("final_status") or "") == "OK":
                done.add((row.get("case_id") or "").strip())
    return done

def is_transient(status: Optional[int], errtype: str) -> bool:
    if status in TRANSIENT_HTTP:
        return True
    if errtype in {"TimeoutError", "APIConnectionError"}:
        return True
    return False

async def call_one(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    case_id: str,
    max_output_tokens: int,
    temperature: float,
) -> Tuple[dict, dict]:
    prompt_hash = sha256_text(system_prompt + "\n\n" + user_prompt)

    for attempt in range(1, 4):
        t0 = time.time()
        try:
            resp = await client.responses.create(
                model=model,
                instructions=system_prompt,
                input=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                store=False,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": JSON_SCHEMA["name"],
                        "schema": JSON_SCHEMA["schema"],
                        "strict": True,
                    }
                },
            )

            latency_ms = int((time.time() - t0) * 1000)
            raw = getattr(resp, "output_text", "") or ""
            if not raw:
                raw = json.dumps(resp.model_dump(), ensure_ascii=False)

            parsed = json.loads(raw)

            ledger = {
                "timestamp": utc_iso(),
                "case_id": case_id,
                "attempt": attempt,
                "model": model,
                "prompt_hash": prompt_hash,
                "http_status": 200,
                "error_type": None,
                "latency_ms": latency_ms,
                "response_id": getattr(resp, "id", None),
            }
            rawrec = {
                "timestamp": utc_iso(),
                "case_id": case_id,
                "attempt": attempt,
                "model": model,
                "response_sha256": sha256_text(raw),
                "response_text": raw,
                "parsed": parsed,
            }
            return ledger, rawrec

        except Exception as e:
            latency_ms = int((time.time() - t0) * 1000)
            errtype = type(e).__name__
            status = getattr(e, "status_code", None) or getattr(e, "status", None)

            ledger = {
                "timestamp": utc_iso(),
                "case_id": case_id,
                "attempt": attempt,
                "model": model,
                "prompt_hash": prompt_hash,
                "http_status": status,
                "error_type": errtype,
                "latency_ms": latency_ms,
                "error_message": str(e)[:1200],
            }

            if attempt < 3 and is_transient(status, errtype):
                sleep = min(60.0, (1.7 ** attempt)) * (0.5 + random.random())
                await asyncio.sleep(sleep)
                continue

            rawrec = {
                "timestamp": utc_iso(),
                "case_id": case_id,
                "attempt": attempt,
                "model": model,
                "response_sha256": None,
                "response_text": "",
                "parsed": None,
            }
            return ledger, rawrec

    return {"timestamp": utc_iso(), "case_id": case_id}, {"parsed": None}

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to run folder containing inputs/")
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-output-tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    base = os.path.abspath(args.run_dir)
    inputs = os.path.join(base, "inputs")
    logs = os.path.join(base, "logs")
    outputs = os.path.join(base, "outputs")
    os.makedirs(inputs, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    cases_csv = os.path.join(inputs, "case_input.csv")
    abx_csv = os.path.join(inputs, "abx_codes.csv")
    manifest_json = os.path.join(inputs, "manifest.json")

    if not os.path.exists(cases_csv):
        raise FileNotFoundError(f"Missing: {cases_csv}")
    if not os.path.exists(abx_csv):
        raise FileNotFoundError(f"Missing: {abx_csv}")
    if not os.path.exists(manifest_json):
        raise FileNotFoundError(f"Missing: {manifest_json}")

    run_date = read_manifest_created_date(manifest_json)
    local_epi_version_date = run_date

    cases = load_cases_from_case_input(cases_csv, limit=args.limit)
    code2gen, canon2codes = load_abx_codes(abx_csv)

    out_csv = os.path.join(outputs, "ai_results_for_sheets.csv")
    ledger_path = os.path.join(logs, "requests_ledger.jsonl")
    raw_path = os.path.join(logs, "responses_raw.jsonl")

    done = already_ok(out_csv) if args.resume else set()

    fieldnames = [
        "case_id",
        "ai_model_name",
        "ai_model_version_date",
        "local_epi_version_date",
        "ai_recommended_regimen_text",
        "ai_abx_code_1",
        "ai_abx_code_2",
        "ai_abx_code_3",
        "ai_antipseudomonal_any",
        "ai_carbapenem_any",
        "ai_anti_mrsa_any",
        "final_status",
        "schema_valid",
        "canon_valid",
        "prompt_hash",
        "attempt_used",
        "canon_mode",  # direct_code | resolved_from_canon | unresolved
    ]

    if (not os.path.exists(out_csv)) or (not args.resume):
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")

    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)

    counters = {}

    async def handle(case: Dict[str, str]) -> None:
        cid = case["case_id"]
        if args.resume and cid in done:
            return

        system_prompt = case["system_prompt"]
        user_prompt = case["user_prompt"]
        prompt_hash = sha256_text(system_prompt + "\n\n" + user_prompt)

        async with sem:
            ledger, rawrec = await call_one(
                client, args.model, system_prompt, user_prompt,
                cid, args.max_output_tokens, args.temperature
            )

        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ledger, ensure_ascii=False) + "\n")

        raw_out = {k: v for k, v in rawrec.items() if k != "parsed"}
        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(raw_out, ensure_ascii=False) + "\n")

        parsed = rawrec.get("parsed")
        schema_valid = 0
        canon_valid = 0
        regimen_text = ""
        code1 = code2 = code3 = ""
        final_status = "FAILED_API"
        canon_mode = "unresolved"
        attempt_used = ledger.get("attempt", None)

        if ledger.get("http_status") == 200 and isinstance(parsed, dict):
            need = {"ai_recommended_regimen_text", "ai_abx_code_1", "ai_abx_code_2", "ai_abx_code_3"}
            schema_valid = int(need.issubset(set(parsed.keys())))

            if schema_valid:
                regimen_text = str(parsed.get("ai_recommended_regimen_text", "")).strip()

                raw1 = str(parsed.get("ai_abx_code_1", "")).strip()
                raw2 = str(parsed.get("ai_abx_code_2", "")).strip()
                raw3 = str(parsed.get("ai_abx_code_3", "")).strip()

                res1 = resolve_to_abx_code(raw1, code2gen, canon2codes)
                res2 = resolve_to_abx_code(raw2, code2gen, canon2codes)
                res3 = resolve_to_abx_code(raw3, code2gen, canon2codes)

                code1, code2, code3 = res1, res2, res3

                def slot_mode(raw: str, res: str) -> str:
                    if not raw.strip():
                        return "direct_code"
                    if raw.strip().upper() == res:
                        return "direct_code"
                    if normalize_generic(raw) in canon2codes:
                        return "resolved_from_canon"
                    if res in code2gen or res in {"", "NO_ANTIBIOTIC"}:
                        return "direct_code"
                    return "unresolved"

                modes = [slot_mode(raw1, res1), slot_mode(raw2, res2), slot_mode(raw3, res3)]
                canon_mode = "resolved_from_canon" if "resolved_from_canon" in modes else ("unresolved" if "unresolved" in modes else "direct_code")

                def ok_slot(c: str) -> bool:
                    if c == "":
                        return True
                    if c == "NO_ANTIBIOTIC":
                        return True
                    return c in code2gen

                slots_ok = ok_slot(code1) and ok_slot(code2) and ok_slot(code3)

                noabx_ok = True
                if code1 == "NO_ANTIBIOTIC":
                    noabx_ok = (code2 == "" and code3 == "")
                else:
                    noabx_ok = (code2 != "NO_ANTIBIOTIC" and code3 != "NO_ANTIBIOTIC")

                canon_valid = int(slots_ok and noabx_ok)

                if canon_valid:
                    final_status = "OK"
                else:
                    if slots_ok and (not noabx_ok):
                        final_status = "FAILED_NOABX_RULE"
                    else:
                        final_status = "FAILED_CANON"
            else:
                final_status = "FAILED_SCHEMA"
        else:
            hs = ledger.get("http_status", None)
            if hs in TRANSIENT_HTTP:
                final_status = "FAILED_TRANSIENT"

        anti_pseudo = carb = anti_mrsa = 0
        if canon_valid and code1 != "NO_ANTIBIOTIC":
            anti_pseudo, carb, anti_mrsa = compute_flags([code1, code2, code3], code2gen)

        out_row = {
            "case_id": cid,
            "ai_model_name": args.model,
            "ai_model_version_date": run_date,
            "local_epi_version_date": local_epi_version_date,
            "ai_recommended_regimen_text": regimen_text,
            "ai_abx_code_1": code1,
            "ai_abx_code_2": code2,
            "ai_abx_code_3": code3,
            "ai_antipseudomonal_any": anti_pseudo,
            "ai_carbapenem_any": carb,
            "ai_anti_mrsa_any": anti_mrsa,
            "final_status": final_status,
            "schema_valid": schema_valid,
            "canon_valid": canon_valid,
            "prompt_hash": prompt_hash,
            "attempt_used": attempt_used,
            "canon_mode": canon_mode,
        }

        with open(out_csv, "a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(out_row)

        counters[final_status] = counters.get(final_status, 0) + 1

    await asyncio.gather(*(handle(c) for c in cases))

    print("DONE:", out_csv)
    print("SUMMARY:", json.dumps(counters, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
