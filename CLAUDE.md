# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Streamlit web application that fetches and visualizes Google Trends "interest over time" data across multiple search terms and countries simultaneously. Uses the unofficial `pytrends` library. All UI text is in Spanish; code comments and docs are in English.

## Running the App

```bash
# Preferred: use the launch script (creates venv, installs deps, starts Streamlit)
./run.sh

# Manual alternative:
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

There are no tests or linting configured in this project.

## Architecture

The app follows a layered architecture with four backend modules consumed by the Streamlit UI:

- **app.py** — Streamlit entry point. Sidebar collects inputs (terms, countries, timeframe, search type, category, throttle settings). Main panel runs queries, displays charts/tables, and provides download buttons. Stores results in `st.session_state` for persistence across reruns.
- **trends_client.py** — `TrendsClient` class wrapping `pytrends`. Fetches interest-over-time for individual (term, country) pairs sequentially with configurable retry (via `tenacity`) and inter-request throttle delay. Returns a list of result dicts with keys: `term`, `country_code`, `status` ("success"/"empty"/"failed"), `data` (DataFrame or None), `error_message`.
- **data_model.py** — Transforms raw results into pandas DataFrames. `build_long_dataframe()` produces the canonical long format with per-pair normalization (max=100). `build_wide_dataframe()` pivots to wide format (date index, one column per pair labeled `"term :: country_code"`).
- **export.py** — CSV/JSON/ZIP export utilities: wide CSV, long CSV, per-pair ZIP, config JSON save/load, run log JSON.
- **countries.py** — Country list helpers using `pycountry`. Provides name↔code mappings and a `MAX_COUNTRIES=30` cap.

### Data Flow

```
User inputs (sidebar) → TrendsClient.fetch_all_pairs() → list[result_dict]
  → build_long_dataframe() → long DataFrame (with normalization)
  → build_wide_dataframe() → wide DataFrame (for table/CSV)
  → Plotly chart + export functions
```

### Key Constraints

- Google Trends has no official API; `pytrends` is subject to rate limiting (~5-10 req/min). The app uses configurable throttle delay (default 10s between requests) and exponential backoff retries.
- Each (term, country) pair is fetched independently and normalized independently — values are relative within each pair, not absolute search volumes.
- Hard limits: MAX_TERMS=20 (in app.py), MAX_COUNTRIES=30 (in countries.py).

## Dependencies

Python 3.11+ with: `streamlit`, `pytrends`, `pandas`, `plotly`, `tenacity`, `pycountry` (see `requirements.txt`).
