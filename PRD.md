# PRD — Multi-Country Google Trends Explorer (Python)

## 1) Overview
Build a Python application that lets a user:
- Enter one or more Google Trends search terms (or “topics” where feasible),
- Select multiple countries and a timeframe,
- Download the underlying time series data as CSV (per term–country pair and/or combined),
- Plot the evolution of interest over time where each **term–country pair is normalized to 0–100** (100 = peak interest for that pair over the selected timeframe).

The product is explicitly **not** intended to provide absolute search volume. It provides **relative, within-pair normalized** time series for comparative visualization of shapes and timing across countries and terms.

---

## 2) Goals & Non-Goals

### Goals
1. Provide an intuitive UI for selecting:
   - Terms (1..N),
   - Countries (1..M),
   - Timeframe (start/end or presets),
   - Search type (Web, YouTube, News, etc.),
   - Optional category.
2. Fetch Google Trends “interest over time” series for every selected **term × country** pair.
3. Normalize and plot each pair so the peak value in that pair’s series is 100.
4. Allow export:
   - Raw series (as returned),
   - Normalized series,
   - A combined “wide” CSV (Date + one column per pair),
   - Optional “long” CSV (Date, term, country, value, normalized_value).
5. Provide reproducible runs (save configuration, re-run later).

### Non-Goals
- Absolute volume estimation.
- Keyword research features (related queries/topics) beyond basic exploration (future enhancement).
- Guaranteed compliance with an official Google Trends API (there isn’t one publicly supported for this use case); we will use a best-effort approach and treat availability as a constraint.

---

## 3) Target Users & Use Cases

### Target Users
- Analysts, marketers, researchers, journalists, students.
- Users who understand Google Trends outputs are normalized.

### Primary Use Cases
1. “Compare how interest in `term` evolved in ES/FR/US between 2021–2025.”
2. “Compare multiple terms across multiple countries and visually inspect timing and relative spikes.”
3. “Export normalized series for reporting and charting in other tools.”

---

## 4) User Experience (UX)

### UI Approach (recommended)
**Streamlit** (fast to build, runs locally, simple deployment).
Alternative options: PySide6/Qt (desktop), Tkinter (basic), web framework (Flask + frontend). Streamlit offers best dev speed for this scope.

### Primary Screen Layout
1. **Sidebar: Inputs**
   - Terms input: multi-line text area (one term per line).
   - Countries selector: multi-select dropdown (with search). Uses ISO-3166-1 alpha-2 codes under the hood.
   - Timeframe:
     - Presets: “Past 12 months”, “Past 5 years”, “2004–present”
     - Custom: start date + end date
   - Granularity controls (optional):
     - Allow user to request weekly vs daily (note: Trends granularity depends on timeframe).
   - Search type: Web / Image / News / Google Shopping / YouTube.
   - Category: optional numeric category selector (with label list).
   - Advanced:
     - Language/locale
     - Retry/backoff config
     - Rate limiting toggle
     - Proxy support toggle (optional)

2. **Main Panel: Results**
   - Run button (“Fetch trends”)
   - Status & progress (term-country queue)
   - Chart area:
     - Multi-line chart with a legend.
     - Controls:
       - Facet mode: “All lines in one plot” vs “Small multiples per term” vs “Small multiples per country”.
       - Toggle: show raw vs normalized.
       - Smoothing (optional): moving average window.
   - Data table preview:
     - Wide format and/or long format toggle.
   - Download buttons:
     - Combined CSV (raw)
     - Combined CSV (normalized)
     - ZIP of pair CSVs (optional)
   - Save/Load config:
     - Export settings JSON
     - Import settings JSON

---

## 5) Functional Requirements

### FR1 — Terms Input
- User can enter 1..N terms.
- Validation:
  - Remove empty lines.
  - Deduplicate terms (case-insensitive) but preserve original display.
  - Hard cap (configurable) to prevent excessive requests (e.g., N ≤ 20).

### FR2 — Country Selection
- Multi-select from a built-in country list (name + ISO code).
- Validation:
  - Require at least 1 country.
  - Hard cap (configurable), e.g., M ≤ 30.

### FR3 — Timeframe Selection
- Presets + custom start/end.
- Validation:
  - Start < End.
  - Enforce acceptable ranges if required by backend behavior.

### FR4 — Query Execution
- For each (term, country) pair:
  - Fetch Google Trends “interest over time”.
  - Capture metadata:
    - term, country code, country name, timeframe, search type, category, retrieval timestamp.
- Provide progress indicator and partial results (don’t block entire run on one failure).

### FR5 — Normalization (“Relative terms”)
- Goal: Each term–country pair is scaled so the **max value in that series = 100**.
- Compute:
  - If raw series max is 0 or series is empty: normalized series is all 0 (and flag as “no data”).
  - Otherwise: `normalized = round(raw / max(raw) * 100, 2)` (precision configurable).
- Important nuance: Google Trends often already returns 0–100. Still normalize to guarantee the invariant, especially when series includes missing values or partial zeros.

### FR6 — Plotting
- Multi-line time series chart:
  - X axis: date/time
  - Y axis: normalized interest (0–100)
  - Line label: `{term} — {country}` (or user-selectable label format).
- Handle:
  - Many lines: provide filtering and faceting.
  - Tooltip with date, raw, normalized, term, country.

### FR7 — CSV Export
- Combined **wide** CSV:
  - Column 1: date
  - Columns 2..: one column per pair, named e.g. `term|ES` (sanitize term for filenames/headers).
- Combined **long** CSV:
  - `date, term, country_code, country_name, raw_value, normalized_value`
- Optional per-pair CSVs:
  - File per (term, country), stored in a folder and optionally zipped.

### FR8 — Configuration Save/Load
- Export a JSON config containing:
  - terms list
  - countries list
  - timeframe
  - search type, category
  - normalization precision
  - chart preferences
- Import config to restore state.

### FR9 — Error Handling & Resilience
- If request fails:
  - Retry with exponential backoff (configurable: retries, base delay, jitter).
  - If still failing, mark pair as failed and continue.
- Provide user-facing summary:
  - Success count, failure count, empty-data count, elapsed time.
- Log all failures with reason (HTTP errors, parsing errors, rate limits).

---

## 6) Non-Functional Requirements

### NFR1 — Performance
- Must support typical runs like:
  - up to 10 terms × 10 countries = 100 pairs within reasonable runtime on consumer hardware.
- Introduce concurrency carefully:
  - Prefer limited parallelism (e.g., 2–5 workers) to reduce rate-limit risk.

### NFR2 — Reliability
- Avoid total failure if a single pair fails.
- Deterministic output for same inputs (subject to Trends’ own volatility).

### NFR3 — Usability
- Clear messaging about normalization:
  - “Each term–country line is normalized independently; values aren’t absolute volume.”
- Provide sensible defaults.

### NFR4 — Portability
- Runs on Windows/macOS/Linux.
- Provide `requirements.txt` or `pyproject.toml`.

### NFR5 — Observability
- Local logs:
  - Info: run start, inputs, counts
  - Warn: empty series
  - Error: request failures
- Optional debug mode to store raw responses (if obtainable).

---

## 7) Technical Approach

### Data Source / Client
- Use the `pytrends` library (unofficial Google Trends client).
- Primary function: “interest over time”.
- Inputs: `kw_list`, `geo`, `timeframe`, `cat`, `gprop` (search type).
- Known constraints:
  - Rate limiting and occasional 429s.
  - Granularity changes with timeframe.

### Architecture
1. **UI Layer**
   - Streamlit app: collects inputs, shows progress, renders charts/tables, download buttons.
2. **Service Layer**
   - TrendsClient: responsible for request orchestration, retries, throttling.
   - DataModel: structures results, merges time indexes, handles missing values.
3. **Normalization Layer**
   - Applies per-pair normalization and produces raw + normalized dataframes.
4. **Export Layer**
   - Writes combined CSVs, per-pair CSVs, ZIP packaging.
5. **Visualization Layer**
   - Plotly preferred for interactive multi-line charts in Streamlit.

### Recommended Stack
- Python 3.11+
- streamlit
- pytrends
- pandas
- plotly
- tenacity (retry/backoff) or custom backoff utility
- pycountry (country list/ISO codes) or a bundled JSON mapping

---

## 8) Data Model

### Internal “Long” Table (canonical)
Columns:
- `date` (datetime)
- `term` (string)
- `country_code` (string, ISO2)
- `country_name` (string)
- `raw_value` (int/float)
- `normalized_value` (float)
- `timeframe_start`, `timeframe_end` (optional metadata)
- `search_type`, `category` (optional metadata)
- `status` (success/empty/failed)
- `error_message` (nullable)

### Derived “Wide” Table
- Index: `date`
- Columns: one per `{term}|{country_code}` containing raw or normalized.

---

## 9) Edge Cases & Handling

1. **Empty data / low volume**
   - Trends may return all zeros or an empty dataframe.
   - Mark as `empty` and display a warning.
2. **Different date indexes across queries**
   - Align on union of dates; fill missing values with 0 or NaN (configurable).
   - Default: use NaN for missing (more honest), but offer “fill 0”.
3. **Large number of lines**
   - Add filtering and faceting:
     - Filter by term(s) or country(ies).
     - “Small multiples” mode.
4. **Term ambiguity**
   - If possible, provide a UI hint: use “Topic” vs “Search term”.
   - (Optional future) Offer topic resolution UI; for MVP, accept raw terms.
5. **Rate limit / CAPTCHA / 429**
   - Backoff retries.
   - Suggest user reduce concurrency or pair count if persistent.
6. **Timeframe too short/long**
   - Trends granularity varies; warn user that daily vs weekly changes are expected.

---

## 10) Privacy & Compliance Considerations
- Runs locally by default; no user data leaves the machine beyond requests to Google Trends endpoints performed by pytrends.
- Store configs/logs locally only.
- Be transparent in UI: “Uses an unofficial client; availability may vary.”

---

## 11) Milestones

### MVP (v0.1)
- Streamlit UI: terms, countries, timeframe, search type.
- Fetch term×country interest over time.
- Normalize per pair.
- Plot normalized lines.
- Export combined normalized CSV.

### v0.2
- Raw vs normalized toggle.
- Long + wide exports.
- Save/load config JSON.
- Better error summaries and logs.

### v0.3
- Faceting / filtering for large line counts.
- Optional per-pair CSV ZIP export.
- Concurrency control UI and advanced retry settings.

### v1.0
- Polished UX, robust edge handling, packaging (pip/installer), documentation.

---

## 12) Acceptance Criteria (MVP)
1. User can input ≥1 term, select ≥1 country, set timeframe, and run.
2. App fetches data for each term–country pair and does not stop if one pair fails.
3. App produces a chart where each pair’s maximum value equals 100 (or 0 if no data).
4. App exports a normalized CSV with:
   - Date column
   - One column per pair
5. App clearly states the normalization rule in the UI/help text.

---

## 13) Open Questions / Future Enhancements
- Topic resolution (choose between “search term” and “topic” entities).
- Related queries/topics panels.
- Built-in report generation (PDF/HTML).
- Caching results to avoid repeat requests.
- Dockerized distribution and/or lightweight desktop packaging.