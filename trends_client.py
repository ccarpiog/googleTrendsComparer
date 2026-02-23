"""
TrendsClient — a wrapper around pytrends with retry logic and throttling.

Fetches Google Trends "interest over time" for each (term, country) pair
independently. Designed for sequential execution with configurable backoff
and throttle delays to minimize rate-limit issues.
"""

import logging
import time

import pandas as pd
from pytrends.request import TrendReq
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


class TrendsClient:
    """
    Wrapper around pytrends that fetches interest-over-time data for
    individual (term, country) pairs with retry logic and throttling.

    Attributes:
        retries (int): Maximum retry attempts per request.
        base_delay (float): Base seconds for exponential backoff.
        max_delay (float): Maximum seconds for backoff cap.
        request_delay (float): Seconds to wait between consecutive requests (throttling).
    """

    def __init__(self, retries=5, base_delay=10.0, max_delay=120.0, request_delay=10.0):
        """
        Initialise the TrendsClient.

        Args:
            retries (int): Max retry attempts per request.
            base_delay (float): Base seconds for exponential backoff.
            max_delay (float): Max seconds for backoff cap.
            request_delay (float): Seconds to wait between requests (throttling).
        """
        self.retries = retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.request_delay = request_delay
    # End of __init__

    def _build_retry_decorator(self):
        """
        Build a tenacity retry decorator configured with the instance's
        retry parameters.

        Returns:
            A tenacity retry decorator ready to wrap a callable.
        """
        return retry(
            stop=stop_after_attempt(self.retries),
            wait=wait_exponential_jitter(
                initial=self.base_delay,
                max=self.max_delay,
                jitter=2,
            ),
            retry=retry_if_exception_type(Exception),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
    # End of _build_retry_decorator

    def fetch_interest_over_time(
        self,
        term,
        country_code,
        timeframe="today 5-y",
        search_type="",
        category=0,
    ):
        """
        Fetch Google Trends interest over time for a single term-country pair.

        Uses pytrends under the hood. Retries with exponential backoff on
        failure. Returns a result dict regardless of success or failure.

        Args:
            term (str): Search term string.
            country_code (str): ISO-3166-1 alpha-2 code (e.g. "ES").
            timeframe (str): pytrends timeframe string
                (e.g. "today 5-y", "2020-01-01 2024-12-31").
            search_type (str): gprop value — "" for web, "youtube", "news",
                "images", "froogle".
            category (int): Google Trends category number (0 = all).

        Returns:
            dict: A result dictionary with keys:
                - "term" (str): The search term.
                - "country_code" (str): The ISO country code.
                - "status" (str): "success", "empty", or "failed".
                - "data" (pandas.DataFrame or None): DataFrame with columns
                  [date, value] on success/empty, or None on failure.
                - "error_message" (str or None): Error description on failure,
                  None otherwise.
        """
        result = {
            "term": term,
            "country_code": country_code,
            "status": "failed",
            "data": None,
            "error_message": None,
        }

        retry_decorator = self._build_retry_decorator()

        @retry_decorator
        def _do_request():
            """
            Perform the actual pytrends request inside the retry wrapper.

            Returns:
                pandas.DataFrame: The raw interest-over-time dataframe from pytrends.

            Raises:
                Exception: Any exception from pytrends is propagated so
                    tenacity can decide whether to retry.
            """
            pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 30))
            pytrends.build_payload(
                kw_list=[term],
                cat=category,
                timeframe=timeframe,
                geo=country_code,
                gprop=search_type,
            )
            return pytrends.interest_over_time()
        # End of _do_request

        try:
            logger.info(
                "Fetching interest over time for term='%s', geo='%s', "
                "timeframe='%s', search_type='%s', category=%d",
                term,
                country_code,
                timeframe,
                search_type,
                category,
            )
            raw_df = _do_request()

            # pytrends returns an empty DataFrame when there is no data
            if raw_df is None or raw_df.empty or term not in raw_df.columns:
                logger.warning(
                    "Empty data returned for term='%s', geo='%s'",
                    term,
                    country_code,
                )
                result["status"] = "empty"
                result["data"] = pd.DataFrame(columns=["date", "value"])
                return result
            # End of empty-data check

            # Build a clean DataFrame with only the columns we need
            df = raw_df[[term]].copy()
            df = df.reset_index()
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])

            # Note: pytrends may include an "isPartial" column in the raw
            # dataframe. By selecting only [term] above, that column is
            # excluded. Partial data points are still kept in the series.

            result["status"] = "success"
            result["data"] = df

            logger.info(
                "Successfully fetched %d data points for term='%s', geo='%s'",
                len(df),
                term,
                country_code,
            )
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Failed to fetch data for term='%s', geo='%s': %s",
                term,
                country_code,
                error_msg,
            )
            result["status"] = "failed"
            result["error_message"] = error_msg
        # End of try/except block for fetch_interest_over_time

        return result
    # End of fetch_interest_over_time

    def fetch_all_pairs(
        self,
        terms,
        country_codes,
        timeframe="today 5-y",
        search_type="",
        category=0,
        progress_callback=None,
    ):
        """
        Fetch data for all term x country combinations sequentially with throttling.

        Iterates over every (term, country_code) pair, calls
        fetch_interest_over_time for each, and applies a sleep-based throttle
        between requests. A failed pair does NOT stop execution — the loop
        continues to the next pair.

        Args:
            terms (list[str]): List of search term strings.
            country_codes (list[str]): List of ISO-3166-1 alpha-2 codes.
            timeframe (str): pytrends timeframe string
                (e.g. "today 5-y", "2020-01-01 2024-12-31").
            search_type (str): gprop value — "" for web, "youtube", "news",
                "images", "froogle".
            category (int): Google Trends category number (0 = all).
            progress_callback (callable or None): Optional callback invoked as
                progress_callback(current_index, total, term, country_code)
                before each request, useful for progress bars / UI updates.

        Returns:
            list[dict]: A list of result dicts, one per (term, country_code) pair.
                Each dict has the same format as fetch_interest_over_time's return
                value.
        """
        results = []
        pairs = [
            (term, country_code)
            for term in terms
            for country_code in country_codes
        ]
        total = len(pairs)

        logger.info(
            "Starting fetch_all_pairs: %d terms x %d countries = %d pairs",
            len(terms),
            len(country_codes),
            total,
        )

        for idx, (term, country_code) in enumerate(pairs):
            # Notify progress listener if provided
            if progress_callback is not None:
                try:
                    progress_callback(idx, total, term, country_code)
                except Exception as cb_exc:
                    logger.warning(
                        "progress_callback raised an exception: %s", cb_exc
                    )
                # End of try/except for progress_callback
            # End of progress_callback invocation

            result = self.fetch_interest_over_time(
                term=term,
                country_code=country_code,
                timeframe=timeframe,
                search_type=search_type,
                category=category,
            )
            results.append(result)

            # Throttle: sleep between requests, but skip the sleep after the
            # last request to avoid an unnecessary wait.
            if idx < total - 1:
                logger.debug(
                    "Throttling: sleeping %.1f seconds before next request",
                    self.request_delay,
                )
                time.sleep(self.request_delay)
            # End of throttle sleep
        # End of loop through all (term, country_code) pairs

        # Log summary
        success_count = sum(1 for r in results if r["status"] == "success")
        empty_count = sum(1 for r in results if r["status"] == "empty")
        failed_count = sum(1 for r in results if r["status"] == "failed")
        logger.info(
            "fetch_all_pairs complete: %d success, %d empty, %d failed out of %d total",
            success_count,
            empty_count,
            failed_count,
            total,
        )

        return results
    # End of fetch_all_pairs
# End of class TrendsClient
