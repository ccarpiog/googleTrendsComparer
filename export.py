"""
CSV and JSON export/import utilities for the Google Trends explorer.

Provides functions to convert pandas DataFrames (wide and long format) into
CSV strings suitable for Streamlit download buttons, as well as configuration
save/load via JSON.
"""

import io
import json
import zipfile

import pandas as pd


def export_wide_csv(wide_df):
    """
    Exports a wide-format DataFrame to CSV string (for Streamlit download).

    The wide DataFrame is expected to have a date index and one column per
    term|country pair. The date index is included in the output.

    Args:
        wide_df: pd.DataFrame with date index and one column per term|country pair.

    Returns:
        str: CSV content as string with the date index included.
    """
    return wide_df.to_csv(index=True)


def export_long_csv(long_df):
    """
    Exports a long-format DataFrame to CSV string (for Streamlit download).

    The long DataFrame is expected to contain the following columns:
    date, term, country_code, country_name, raw_value, normalized_value,
    status, error_message. The index is not included in the output since
    it carries no meaningful information in long format.

    Args:
        long_df: pd.DataFrame with columns: date, term, country_code,
                 country_name, raw_value, normalized_value, status,
                 error_message.

    Returns:
        str: CSV content as string without the DataFrame index.
    """
    return long_df.to_csv(index=False)


def export_config_json(terms, country_codes, timeframe, search_type, category):
    """
    Exports the current configuration as a JSON string for save/load.

    Serialises the exploration parameters into a JSON structure that can
    later be restored with import_config_json.

    Args:
        terms: list of search term strings.
        country_codes: list of ISO-3166-1 alpha-2 codes.
        timeframe: timeframe string (e.g. "today 12-m").
        search_type: search type string (e.g. "web", "news", "images").
        category: category number (0 for all categories).

    Returns:
        str: Pretty-printed JSON string.
    """
    config = {
        "terms": terms,
        "country_codes": country_codes,
        "timeframe": timeframe,
        "search_type": search_type,
        "category": category,
    }
    return json.dumps(config, indent=2, ensure_ascii=False)
# End of function export_config_json()


def export_run_log_json(results, code_to_name_map):
    """
    Exports a structured run log as a JSON string for download.

    Each entry in the log corresponds to a (term, country) pair and includes
    the status, error message (if any), and the number of data points retrieved.

    Args:
        results (list[dict]): List of result dicts from TrendsClient.
        code_to_name_map (dict): Mapping of country_code -> country_name.

    Returns:
        str: Pretty-printed JSON string with the run log.
    """
    log_entries = []
    for r in results:
        entry = {
            "term": r["term"],
            "country_code": r["country_code"],
            "country_name": code_to_name_map.get(r["country_code"], r["country_code"]),
            "status": r["status"],
            "data_points": len(r["data"]) if r["data"] is not None and not r["data"].empty else 0,
            "error_message": r.get("error_message"),
        }
        log_entries.append(entry)
    # End of the loop that builds log entries

    return json.dumps(log_entries, indent=2, ensure_ascii=False)
# End of function export_run_log_json()


def _sanitise_filename(name):
    """
    Sanitise a string for use as a safe cross-platform file name component.

    Replaces characters that are invalid on Windows or common file systems
    with underscores.

    Args:
        name (str): The raw string to sanitise.

    Returns:
        str: A file-system-safe version of the name.
    """
    import re
    return re.sub(r'[/\\:*?"<>| ]+', "_", name).strip("_")
# End of function _sanitise_filename()


def export_per_pair_zip(results):
    """
    Exports individual CSV files for each successful (term, country) pair
    packaged in a ZIP archive.

    Each CSV file contains columns: date, raw_value, normalized_value.
    File names follow the pattern: {term}__{country_code}.csv (sanitised
    for cross-platform safety). Duplicate names are disambiguated with a
    numeric suffix.

    Args:
        results (list[dict]): List of result dicts from TrendsClient.

    Returns:
        bytes: ZIP archive content as bytes, ready for Streamlit download.
    """
    buf = io.BytesIO()
    used_names = set()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            if r["status"] != "success" or r["data"] is None or r["data"].empty:
                continue

            df = r["data"].copy()
            df.rename(columns={"value": "raw_value"}, inplace=True)

            # Fill NaN before normalization (matches data_model.py logic)
            df["raw_value"] = df["raw_value"].fillna(0)

            max_val = df["raw_value"].max()
            if max_val > 0:
                df["normalized_value"] = (df["raw_value"] / max_val * 100).round(2)
            else:
                df["normalized_value"] = 0.0

            # Build a safe, deduplicated filename
            safe_term = _sanitise_filename(r["term"])
            base_name = f"{safe_term}__{r['country_code']}"
            filename = f"{base_name}.csv"
            counter = 1
            while filename in used_names:
                filename = f"{base_name}_{counter}.csv"
                counter += 1
            used_names.add(filename)

            zf.writestr(filename, df.to_csv(index=False))
        # End of the loop that writes per-pair CSVs into the ZIP
    # End of zipfile context manager

    buf.seek(0)
    return buf.getvalue()
# End of function export_per_pair_zip()


_REQUIRED_CONFIG_FIELDS = {"terms", "country_codes", "timeframe", "search_type", "category"}


def import_config_json(json_string):
    """
    Parses a configuration JSON string.

    Validates that the JSON is well-formed and contains every required field
    before returning the configuration dictionary.

    Args:
        json_string: JSON string previously produced by export_config_json.

    Returns:
        dict with keys: terms, country_codes, timeframe, search_type, category.

    Raises:
        ValueError: if the JSON is invalid or missing required fields.
    """
    try:
        config = json.loads(json_string)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected a JSON object at the top level, got {type(config).__name__}"
        )

    missing_fields = _REQUIRED_CONFIG_FIELDS - set(config.keys())
    if missing_fields:
        raise ValueError(
            f"Missing required configuration fields: {', '.join(sorted(missing_fields))}"
        )

    # Validate types of required fields
    if not isinstance(config["terms"], list) or not all(isinstance(t, str) for t in config["terms"]):
        raise ValueError("'terms' must be a list of strings")
    if not isinstance(config["country_codes"], list) or not all(isinstance(c, str) for c in config["country_codes"]):
        raise ValueError("'country_codes' must be a list of strings")
    if not isinstance(config["timeframe"], str):
        raise ValueError("'timeframe' must be a string")
    if not isinstance(config["search_type"], str):
        raise ValueError("'search_type' must be a string")
    if not isinstance(config["category"], (int, float)):
        raise ValueError("'category' must be a number")
    # End of the type-validation block

    return {field: config[field] for field in _REQUIRED_CONFIG_FIELDS}
# End of function import_config_json()
