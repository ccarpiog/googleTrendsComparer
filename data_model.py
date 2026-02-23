"""
data_model.py - Data processing and normalization for Google Trends data.

Receives results from a trends client (list of dicts) and produces
normalized dataframes for visualization and export.
"""

import pandas as pd


def build_long_dataframe(results, country_name_map):
    """
    Builds a canonical long-format DataFrame from results.

    Takes the raw result dicts returned by the trends client and assembles
    them into a single long-format DataFrame with both raw and normalized
    values. Normalization is performed per (term, country_code) pair:
    normalized_value = raw_value / max(raw_value for that pair) * 100,
    rounded to 2 decimal places. If the max is 0 or the status is not
    "success", normalized_value is set to 0.

    Args:
        results (list[dict]): List of result dicts from TrendsClient.
            Each dict has keys: term, country_code, status, data, error_message.
            data is a pd.DataFrame with columns [date, value] or None.
        country_name_map (dict): Mapping of country_code -> country_name,
            e.g. {"ES": "Spain", "US": "United States"}.

    Returns:
        pd.DataFrame: Long-format DataFrame with columns:
            date, term, country_code, country_name, raw_value,
            normalized_value, status, error_message.
    """
    rows = []

    for result in results:
        term = result["term"]
        country_code = result["country_code"]
        status = result["status"]
        error_message = result.get("error_message")
        country_name = country_name_map.get(country_code, country_code)
        data = result.get("data")

        if status == "success" and data is not None and not data.empty:
            # Extract a copy of the data to avoid mutating the original
            pair_df = data.copy()
            pair_df.rename(columns={"value": "raw_value"}, inplace=True)

            # Replace NaN with 0 in raw_value before normalization
            pair_df["raw_value"] = pair_df["raw_value"].fillna(0)

            max_val = pair_df["raw_value"].max()

            if max_val > 0:
                pair_df["normalized_value"] = (
                    (pair_df["raw_value"] / max_val * 100).round(2)
                )
            else:
                pair_df["normalized_value"] = 0.0

            pair_df["term"] = term
            pair_df["country_code"] = country_code
            pair_df["country_name"] = country_name
            pair_df["status"] = status
            pair_df["error_message"] = error_message

            rows.append(pair_df)
        else:
            # For empty or failed results, create a single placeholder row
            placeholder = pd.DataFrame([{
                "date": pd.NaT,
                "term": term,
                "country_code": country_code,
                "country_name": country_name,
                "raw_value": 0,
                "normalized_value": 0.0,
                "status": status,
                "error_message": error_message,
            }])
            rows.append(placeholder)
        # End of the loop that processes each result dict

    if not rows:
        return pd.DataFrame(
            columns=[
                "date", "term", "country_code", "country_name",
                "raw_value", "normalized_value", "status", "error_message",
            ]
        )

    long_df = pd.concat(rows, ignore_index=True)

    # Ensure consistent column order
    column_order = [
        "date", "term", "country_code", "country_name",
        "raw_value", "normalized_value", "status", "error_message",
    ]
    long_df = long_df[column_order]

    return long_df
# End of function build_long_dataframe()


def build_wide_dataframe(long_df, value_column="normalized_value"):
    """
    Pivots the long DataFrame to wide format.

    Filters to only include rows with status "success", then pivots so that
    each (term, country_code) pair becomes a column. The index is the date.

    Args:
        long_df (pd.DataFrame): DataFrame from build_long_dataframe, with
            columns: date, term, country_code, country_name, raw_value,
            normalized_value, status, error_message.
        value_column (str): Which column to use as cell values in the wide
            format. Defaults to "normalized_value". Can also be "raw_value".

    Returns:
        pd.DataFrame: Wide-format DataFrame with:
            index: date
            columns: "{term} :: {country_code}" for each pair
            Returns an empty DataFrame if there is no successful data.
    """
    if long_df.empty:
        return pd.DataFrame()

    # Only include rows with status "success" and a valid date
    success_df = long_df[
        (long_df["status"] == "success") & (long_df["date"].notna())
    ].copy()

    if success_df.empty:
        return pd.DataFrame()

    # Build a composite column label for the pivot.
    # Use " :: " as separator to avoid collision with terms containing "|".
    success_df["pair_label"] = (
        success_df["term"] + " :: " + success_df["country_code"]
    )

    wide_df = success_df.pivot_table(
        index="date",
        columns="pair_label",
        values=value_column,
        aggfunc="first",
    )

    # Remove the columns name attribute for a cleaner output
    wide_df.columns.name = None

    # Sort the index chronologically
    wide_df.sort_index(inplace=True)

    return wide_df
# End of function build_wide_dataframe()


def get_run_summary(results):
    """
    Returns a summary dict of the fetch run.

    Counts the number of results by status and collects the (term, country_code)
    pairs that failed.

    Args:
        results (list[dict]): List of result dicts from TrendsClient.
            Each dict has keys: term, country_code, status.

    Returns:
        dict: Summary with keys:
            "total" (int): Total number of result entries.
            "success" (int): Count of results with status "success".
            "empty" (int): Count of results with status "empty".
            "failed" (int): Count of results with status "failed".
            "failed_pairs" (list[tuple]): List of (term, country_code) tuples
                for results whose status is "failed".
    """
    total = len(results)
    success = 0
    empty = 0
    failed = 0
    failed_pairs = []

    for result in results:
        status = result["status"]
        if status == "success":
            success += 1
        elif status == "empty":
            empty += 1
        elif status == "failed":
            failed += 1
            failed_pairs.append((result["term"], result["country_code"]))
    # End of the loop that counts statuses

    return {
        "total": total,
        "success": success,
        "empty": empty,
        "failed": failed,
        "failed_pairs": failed_pairs,
    }
# End of function get_run_summary()
