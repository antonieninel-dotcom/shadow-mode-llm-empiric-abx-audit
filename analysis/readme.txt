Figure4.py requires tableS_guardrail_context_violations.csv 
Figure5.py requires table2_secondary_deltas.csv, table2b_endpoints_plus.csv, tableS_guardrail_context_violations.csv
------------------------------------------------------------------------------------------------------------------------------------------
Expected files for sap_runner.py: --xlsx data.xlsx with sheets:
  - data
  - endpoints
  - optional: abx_codes (for AWaRe mapping)
------------------------------------------------------------------------------------------------------------------------------------------
run_openai_api.py

Batch runner for retrospective LLM evaluation (offline/shadow mode).
- Reads inputs/case_input.csv with columns: case_id, ai_user_prompt_text, ai_system_prompt
- Reads inputs/abx_codes.csv with columns: abx_code, generic_name_en, (optional) abx_canon
  * abx_code should include route-aware codes (e.g., CRO_IV, AMC1G_PO)
  * NO_ANTIBIOTIC must be present as an abx_code and is handled as a sentinel
- Calls OpenAI Responses API with STRICT JSON schema output (Structured Outputs)
- Writes outputs/ai_results_for_sheets.csv plus append-only logs in logs/

------------------------------------------------------------------------------------------------------------------------------------------
local_epi_script.py requires local epidemiological data files (not to be made public)