"""
Main Streamlit application for the Multi-Country Google Trends Explorer.

Entry point that ties together the backend modules (countries, trends_client,
data_model, export) into an interactive UI.
"""

import hashlib

import streamlit as st
import plotly.express as px
from datetime import date, timedelta

from countries import get_country_list, get_country_map, get_code_to_name, MAX_COUNTRIES
from trends_client import TrendsClient
from data_model import build_long_dataframe, build_wide_dataframe, get_run_summary
from export import (
    export_wide_csv,
    export_long_csv,
    export_config_json,
    import_config_json,
    export_run_log_json,
    export_per_pair_zip,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Google Trends Explorer", layout="wide")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TERMS = 20

TIMEFRAME_PRESETS = {
    "Last month": "today 1-m",
    "Last 12 months": "today 12-m",
    "Year to date": "year_to_date",
    "Last 5 years": "today 5-y",
    "Since 2004": "all",
    "Custom": "custom",
}

SEARCH_TYPE_OPTIONS = {
    "Web": "",
    "YouTube": "youtube",
    "News": "news",
    "Images": "images",
    "Google Shopping": "froogle",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def parse_terms(raw_text):
    """
    Parse, deduplicate, and validate search terms from a multiline text input.

    Splits the raw text by newlines, strips whitespace, removes empty lines,
    and deduplicates case-insensitively (keeps the first occurrence).

    Args:
        raw_text (str): Raw text from the text area, one term per line.

    Returns:
        list[str]: Deduplicated, non-empty search terms.
    """
    seen = set()
    terms = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        key = stripped.lower()
        if key not in seen:
            seen.add(key)
            terms.append(stripped)
    # End of the loop that deduplicates terms
    return terms
# End of function parse_terms()


def restore_config_to_session(config, country_name_map):
    """
    Write imported configuration values into st.session_state so widgets
    pick them up on the next rerun.

    Args:
        config (dict): Configuration dict with keys terms, country_codes,
            timeframe, search_type, category.
        country_name_map (dict): Mapping {code: name} to convert country
            codes back to display names.
    """
    # Terms
    st.session_state["terms_input"] = "\n".join(config.get("terms", []))

    # Countries: convert codes to names for the multiselect widget
    code_to_name = country_name_map
    country_names = [
        code_to_name[code]
        for code in config.get("country_codes", [])
        if code in code_to_name
    ]
    st.session_state["countries_input"] = country_names

    # Timeframe
    timeframe = config.get("timeframe", "today 12-m")
    preset_match = None
    for label, value in TIMEFRAME_PRESETS.items():
        if value == timeframe:
            preset_match = label
            break
    # End of the loop that searches for a matching preset

    if preset_match:
        st.session_state["timeframe_preset"] = preset_match
    else:
        # Custom timeframe: try to parse start/end dates
        st.session_state["timeframe_preset"] = "Custom"
        parts = timeframe.split()
        if len(parts) == 2:
            try:
                st.session_state["custom_start"] = date.fromisoformat(parts[0])
                st.session_state["custom_end"] = date.fromisoformat(parts[1])
            except ValueError:
                pass
    # End of timeframe restoration

    # Search type
    search_type_val = config.get("search_type", "")
    for label, value in SEARCH_TYPE_OPTIONS.items():
        if value == search_type_val:
            st.session_state["search_type_input"] = label
            break
    # End of the loop that matches search type

    # Category
    st.session_state["category_input"] = config.get("category", 0)
# End of function restore_config_to_session()


def get_available_series(long_df):
    """
    Extract the list of unique series labels from the long DataFrame.

    Only includes rows with status "success" and a valid date.

    Args:
        long_df (pd.DataFrame): Long-format DataFrame from build_long_dataframe.

    Returns:
        list[str]: Sorted list of unique "term — country_name" labels.
    """
    success_df = long_df[
        (long_df["status"] == "success") & (long_df["date"].notna())
    ]
    if success_df.empty:
        return []

    series = (success_df["term"] + " \u2014 " + success_df["country_name"]).unique()
    return sorted(series.tolist())
# End of function get_available_series()


def build_chart(long_df, value_column, y_label, visible_series=None, facet_mode="all"):
    """
    Build a Plotly line chart from the long DataFrame.

    Filters the DataFrame to only include rows with status "success" and
    a valid date, then creates a line chart. Supports filtering by series
    visibility and optional faceting by term or country.

    Args:
        long_df (pd.DataFrame): Long-format DataFrame from build_long_dataframe.
        value_column (str): Column to use for the Y axis ("normalized_value"
            or "raw_value").
        y_label (str): Display label for the Y axis.
        visible_series (list[str] or None): List of "term — country_name" labels
            to include. If None, all successful series are shown.
        facet_mode (str): One of "all" (single chart), "by_term" (one facet per
            term), or "by_country" (one facet per country).

    Returns:
        plotly.graph_objects.Figure or None: The configured Plotly figure,
            or None if there is no data to plot.
    """
    chart_df = long_df[
        (long_df["status"] == "success") & (long_df["date"].notna())
    ].copy()

    if chart_df.empty:
        return None

    # Build a readable legend label
    chart_df["serie"] = chart_df["term"] + " \u2014 " + chart_df["country_name"]

    # Filter to only visible series
    if visible_series is not None:
        chart_df = chart_df[chart_df["serie"].isin(visible_series)]
        if chart_df.empty:
            return None
    # End of visible-series filtering

    # Determine faceting column
    facet_col = None
    if facet_mode == "by_term":
        facet_col = "term"
    elif facet_mode == "by_country":
        facet_col = "country_name"

    fig = px.line(
        chart_df,
        x="date",
        y=value_column,
        color="serie",
        facet_col=facet_col,
        facet_col_wrap=2 if facet_col else None,
        labels={
            "date": "Date",
            value_column: y_label,
            "serie": "Term \u2014 Country",
        },
    )

    fig.update_layout(
        legend_title_text="Term \u2014 Country",
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode="x unified",
    )

    # When faceting, clean up subplot titles (remove "term=" / "country_name=" prefixes)
    if facet_col:
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    return fig
# End of function build_chart()


# ---------------------------------------------------------------------------
# Preload country data (cached)
# ---------------------------------------------------------------------------

country_list = get_country_list()
country_names = [c["name"] for c in country_list]
country_name_to_code = get_country_map()
code_to_name_map = get_code_to_name()


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("Google Trends Explorer")


# ---------------------------------------------------------------------------
# Sidebar — Input controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Search parameters")

    # 1. Terms input
    terms_raw = st.text_area(
        "Search terms (one per line)",
        height=150,
        key="terms_input",
    )

    # 2. Countries selector
    selected_country_names = st.multiselect(
        "Countries",
        options=country_names,
        max_selections=MAX_COUNTRIES,
        key="countries_input",
    )

    # 3. Timeframe
    timeframe_label = st.selectbox(
        "Time period",
        options=list(TIMEFRAME_PRESETS.keys()),
        key="timeframe_preset",
    )

    timeframe_value = TIMEFRAME_PRESETS[timeframe_label]

    # Resolve "year_to_date" to a custom date range (Jan 1 → today)
    if timeframe_value == "year_to_date":
        timeframe_value = f"{date.today().year}-01-01 {date.today().isoformat()}"
        timeframe_valid = True
    elif timeframe_value == "custom":
        col_start, col_end = st.columns(2)
        with col_start:
            custom_start = st.date_input(
                "Start date",
                value=date.today() - timedelta(days=365),
                key="custom_start",
            )
        with col_end:
            custom_end = st.date_input(
                "End date",
                value=date.today(),
                key="custom_end",
            )
        timeframe_valid = custom_start < custom_end
        if not timeframe_valid:
            st.error("Start date must be before end date.")
        timeframe_value = f"{custom_start.isoformat()} {custom_end.isoformat()}"
    else:
        timeframe_valid = True
    # End of custom timeframe block

    # 4. Search type
    search_type_label = st.selectbox(
        "Search type",
        options=list(SEARCH_TYPE_OPTIONS.keys()),
        key="search_type_input",
    )
    search_type_value = SEARCH_TYPE_OPTIONS[search_type_label]

    # 5. Category
    category_value = st.number_input(
        "Category (0 = all)",
        min_value=0,
        value=0,
        step=1,
        key="category_input",
    )

    # 6. Advanced throttle/retry settings
    with st.expander("Advanced (speed and retries)", expanded=False):
        st.caption(
            "Google Trends does not have an official API and enforces rate "
            "limits (~5-10 requests/min). If you get 429 errors, "
            "increase the delay between requests."
        )
        request_delay = st.slider(
            "Delay between requests (seconds)",
            min_value=1,
            max_value=60,
            value=10,
            step=1,
            key="request_delay",
            help="Seconds to wait between each Google Trends query.",
        )
        retries = st.slider(
            "Retries per failed pair",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key="retries",
            help="How many times to retry if a request fails (with exponential backoff).",
        )
        backoff_base = st.slider(
            "Base retry delay (seconds)",
            min_value=5,
            max_value=60,
            value=10,
            step=5,
            key="backoff_base",
            help="Initial wait before the first retry. Doubles with each subsequent attempt.",
        )

        total_pairs_est = len(parse_terms(terms_raw)) * len(selected_country_names)
        if total_pairs_est > 0:
            est_minutes = (total_pairs_est * request_delay) / 60
            st.info(f"Estimate: {total_pairs_est} pairs, ~{est_minutes:.1f} min without retries.")
    # End of advanced settings expander

    st.divider()

    # 7. Run button
    run_clicked = st.button("Fetch trends", type="primary", use_container_width=True)

    st.divider()

    # 7. Config save/load
    st.subheader("Configuration")

    # --- Save config ---
    terms_for_config = parse_terms(terms_raw)
    codes_for_config = [country_name_to_code[n] for n in selected_country_names if n in country_name_to_code]

    config_json_str = export_config_json(
        terms=terms_for_config,
        country_codes=codes_for_config,
        timeframe=timeframe_value,
        search_type=search_type_value,
        category=int(category_value),
    )

    st.download_button(
        label="Save configuration",
        data=config_json_str,
        file_name="trends_config.json",
        mime="application/json",
        use_container_width=True,
    )

    # --- Load config ---
    uploaded_config = st.file_uploader(
        "Load configuration",
        type=["json"],
        key="config_uploader",
    )

    if uploaded_config is not None:
        # Guard against infinite rerun loop: only process if content differs
        config_hash = hashlib.md5(uploaded_config.getvalue()).hexdigest()
        if st.session_state.get("_last_imported_config_hash") != config_hash:
            try:
                raw_json = uploaded_config.read().decode("utf-8")
                imported_config = import_config_json(raw_json)
                restore_config_to_session(imported_config, code_to_name_map)
                st.session_state["_last_imported_config_hash"] = config_hash
                st.success("Configuration loaded successfully.")
                st.rerun()
            except ValueError as exc:
                st.error(f"Error loading configuration: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error loading configuration: {exc}")
        # End of config-already-imported guard
    # End of config upload handler
# End of sidebar block


# ---------------------------------------------------------------------------
# Main panel — Fetch & display results
# ---------------------------------------------------------------------------

if run_clicked:
    # Parse and validate inputs
    terms = parse_terms(terms_raw)
    selected_codes = [
        country_name_to_code[name]
        for name in selected_country_names
        if name in country_name_to_code
    ]

    # Validate: at least 1 term
    if not terms:
        st.error("You must enter at least one search term.")
    elif len(terms) > MAX_TERMS:
        st.warning(
            f"{len(terms)} terms entered. The maximum is {MAX_TERMS}. "
            f"Only the first {MAX_TERMS} will be used."
        )
        terms = terms[:MAX_TERMS]
    # End of terms validation

    # Validate: at least 1 country
    if not selected_codes:
        st.error("You must select at least one country.")
    # End of country validation

    # Proceed only if inputs are valid
    if terms and selected_codes and timeframe_valid:
        total_pairs = len(terms) * len(selected_codes)

        progress_bar = st.progress(0, text="Starting search...")
        status_container = st.status("Fetching data from Google Trends...", expanded=True)

        def progress_callback(current_index, total, term, country_code):
            """
            Callback invoked by TrendsClient for each pair being fetched.
            Updates the Streamlit progress bar and status message.

            Args:
                current_index (int): Zero-based index of the current pair.
                total (int): Total number of pairs to fetch.
                term (str): The search term being fetched.
                country_code (str): The country code being fetched.
            """
            fraction = (current_index + 1) / total
            country_display = code_to_name_map.get(country_code, country_code)
            message = f"Querying: \"{term}\" in {country_display} ({current_index + 1}/{total})"
            progress_bar.progress(fraction, text=message)
            status_container.write(message)
        # End of function progress_callback()

        try:
            client = TrendsClient(
                retries=retries,
                base_delay=float(backoff_base),
                request_delay=float(request_delay),
            )
            results = client.fetch_all_pairs(
                terms=terms,
                country_codes=selected_codes,
                timeframe=timeframe_value,
                search_type=search_type_value,
                category=int(category_value),
                progress_callback=progress_callback,
            )

            # Update progress to 100%
            progress_bar.progress(1.0, text="Search completed.")
            status_container.update(label="Search completed.", state="complete", expanded=False)

            # Process results
            long_df = build_long_dataframe(results, code_to_name_map)
            wide_norm_df = build_wide_dataframe(long_df, value_column="normalized_value")
            wide_raw_df = build_wide_dataframe(long_df, value_column="raw_value")

            # Store in session state for persistence
            st.session_state["results"] = results
            st.session_state["long_df"] = long_df
            st.session_state["wide_norm_df"] = wide_norm_df
            st.session_state["wide_raw_df"] = wide_raw_df

        except Exception as exc:
            progress_bar.empty()
            status_container.update(label="Error during search.", state="error", expanded=True)
            st.error(f"Unexpected error during search: {exc}")
        # End of try/except for fetching
    # End of valid-inputs block
# End of run_clicked block


# ---------------------------------------------------------------------------
# Display results (from session state, persists across reruns)
# ---------------------------------------------------------------------------

if "results" in st.session_state and st.session_state["results"] is not None:
    results = st.session_state["results"]
    long_df = st.session_state["long_df"]
    wide_norm_df = st.session_state["wide_norm_df"]
    wide_raw_df = st.session_state["wide_raw_df"]

    # --- Run summary ---
    summary = get_run_summary(results)

    col_ok, col_empty, col_fail = st.columns(3)
    col_ok.metric("Successful", summary["success"])
    col_empty.metric("No data", summary["empty"])
    col_fail.metric("Failed", summary["failed"])

    if summary["failed_pairs"]:
        failed_lines = "\n".join(
            f"- **{term}** in {code_to_name_map.get(code, code)}"
            for term, code in summary["failed_pairs"]
        )
        st.warning(
            f"The following combinations failed:\n\n{failed_lines}"
        )
    # End of failed-pairs warning

    # --- Detailed run log (collapsible) ---
    with st.expander("Run details", expanded=False):
        import pandas as _pd
        log_rows = []
        for r in results:
            log_rows.append({
                "Term": r["term"],
                "Country": code_to_name_map.get(r["country_code"], r["country_code"]),
                "Status": {"success": "Successful", "empty": "No data", "failed": "Failed"}.get(r["status"], r["status"]),
                "Points": len(r["data"]) if r["data"] is not None and not r["data"].empty else 0,
                "Error": r.get("error_message") or "",
            })
        # End of the loop that builds log table rows
        st.dataframe(_pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

        st.download_button(
            label="Download run log (JSON)",
            data=export_run_log_json(results, code_to_name_map),
            file_name="trends_run_log.json",
            mime="application/json",
            use_container_width=True,
        )
    # End of run-log expander

    # --- Normalization info ---
    st.info(
        "Each term-country line is normalized independently; "
        "values do not represent absolute search volume."
    )

    # --- Chart ---
    st.subheader("Trends chart")

    # Chart controls row
    ctrl_col1, ctrl_col2 = st.columns(2)

    with ctrl_col1:
        chart_mode = st.radio(
            "Values to display",
            options=["Normalized", "Raw"],
            horizontal=True,
            key="chart_mode",
        )

    with ctrl_col2:
        facet_mode_label = st.radio(
            "Chart mode",
            options=["All together", "Split by term", "Split by country"],
            horizontal=True,
            key="facet_mode",
        )
    # End of chart controls row

    facet_mode_map = {
        "All together": "all",
        "Split by term": "by_term",
        "Split by country": "by_country",
    }
    facet_mode = facet_mode_map[facet_mode_label]

    if chart_mode == "Normalized":
        value_col = "normalized_value"
        y_label = "Normalized value"
    else:
        value_col = "raw_value"
        y_label = "Raw value"

    # Series visibility filter
    all_series = get_available_series(long_df)

    visible_series = st.multiselect(
        "Visible series (deselect to hide)",
        options=all_series,
        default=all_series,
        key="visible_series",
    )

    fig = build_chart(long_df, value_col, y_label, visible_series=visible_series, facet_mode=facet_mode)

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No successful data to plot.")

    # --- Data table ---
    st.subheader("Data preview (wide format)")

    if not wide_norm_df.empty:
        preview_df = wide_norm_df if chart_mode == "Normalized" else wide_raw_df
        st.dataframe(preview_df.head(100), use_container_width=True)
    else:
        st.info("No data available to display in table.")
    # End of data-table block

    # --- Download buttons ---
    st.subheader("Downloads")

    dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)

    with dl_col1:
        if not wide_norm_df.empty:
            st.download_button(
                label="Normalized CSV (wide)",
                data=export_wide_csv(wide_norm_df),
                file_name="trends_wide_normalized.csv",
                mime="text/csv",
                use_container_width=True,
            )
        # End of normalized wide CSV download
    # End of dl_col1

    with dl_col2:
        if not wide_raw_df.empty:
            st.download_button(
                label="Raw CSV (wide)",
                data=export_wide_csv(wide_raw_df),
                file_name="trends_wide_raw.csv",
                mime="text/csv",
                use_container_width=True,
            )
        # End of raw wide CSV download
    # End of dl_col2

    with dl_col3:
        if not long_df.empty:
            st.download_button(
                label="Long format CSV",
                data=export_long_csv(long_df),
                file_name="trends_long.csv",
                mime="text/csv",
                use_container_width=True,
            )
        # End of long CSV download
    # End of dl_col3

    with dl_col4:
        zip_data = export_per_pair_zip(results)
        if zip_data:
            st.download_button(
                label="ZIP per individual pair",
                data=zip_data,
                file_name="trends_per_pair.zip",
                mime="application/zip",
                use_container_width=True,
            )
        # End of per-pair ZIP download
    # End of dl_col4
# End of results display block
