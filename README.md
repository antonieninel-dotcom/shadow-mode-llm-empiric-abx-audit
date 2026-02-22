# Shadow-mode LLM vs clinician empiric antibiotics — Public audit bundle (non-PHI)

This repository contains **derived, non-identifiable** artifacts supporting reproducibility of a retrospective, single-center **paired** (per admission) study comparing **clinician empiric antibiotic regimens** versus **LLM-recommended regimens** within the first **24 hours** of admission. The LLM arm was generated in **shadow-mode/offline** (no clinical intervention).

## Scope and intent
This repo is designed to let readers/reviewers audit:
- **Study specification and scoring logic** (structured output schema, allowed regimen codes, guardrail rules/weights, costing assumptions).
- **Traceability** of publicly released artifacts (file manifests and SHA-256 checksums).
- **Derived, aggregate outputs** used in the manuscript (tables/figure inputs), without exposing patient-level records.

## What is NOT included
- Patient-level EHR extracts (row-level admissions/patient data).
- Raw per-admission prompts and full free-text model responses (may contain potentially identifiable clinical context).
- Any linkage keys or identifiers.

A restricted bundle may be shared under institutional approvals and data-use agreements.

## Repository structure (public-only)
> Note: this GitHub repository contains the **public** folder only (non-PHI).
public/
analysis/ # Analysis + plotting scripts (no PHI)
docs/ # Methods notes and reproducibility documentation
figures/ # Final manuscript figures (export-ready)
manifests/ # analysis_manifest.json + sha256_report.csv
mappings/ # Regimen dictionaries, guardrail rules, costing tables (drug acquisition only)
outputs/ # Derived, aggregate outputs used in the manuscript (no row-level EHR)
prompts/ # De-identified prompt template (placeholders only) + system/output rules
schemas/ # Structured output schema (JSON)
supplementary/ # Supplementary tables/data files/appendices referenced in the manuscript
tables/ # Exported main tables (e.g., Table 1/2/3) when applicable
manuscripts/ # Manuscript versions provided for review (optional)


## Supplementary naming convention
- **Supplementary Appendix 1**: Prompt template and structured-output specification (placeholders only; de-identified).
- **Supplementary Data File S2**: Costing assumptions and unit-price mapping (**drug acquisition only**).

## How to cite
See `CITATION.cff`.

## License
Code and text in this repository are provided under the MIT License (see `LICENSE`).

_Last updated: 2026-02-22_