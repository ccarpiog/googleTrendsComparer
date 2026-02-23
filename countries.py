"""
Country utilities for Google Trends Comparer.

Provides functions to retrieve ISO-3166-1 alpha-2 country codes and names
using the pycountry library. Designed for use with Google Trends geo parameters.
"""

import pycountry

# Hard cap on the number of countries that can be compared simultaneously
MAX_COUNTRIES = 30


def get_country_list():
    """
    Return a sorted list of country dictionaries.

    Each dictionary contains:
        - "name" (str): The country's common name.
        - "code" (str): The ISO-3166-1 alpha-2 code.

    Returns:
        list[dict]: A list of dicts like [{"name": "Spain", "code": "ES"}, ...]
            sorted alphabetically by country name.

    Example:
        >>> countries = get_country_list()
        >>> countries[0]
        {'name': 'Afghanistan', 'code': 'AF'}
    """
    country_list = [
        {"name": country.name, "code": country.alpha_2}
        for country in pycountry.countries
        if hasattr(country, "alpha_2")
    ]

    country_list.sort(key=lambda c: c["name"])

    return country_list
# End of function get_country_list()


def get_country_map():
    """
    Return a mapping of country names to their ISO-3166-1 alpha-2 codes.

    Returns:
        dict[str, str]: A dictionary mapping country names to codes,
            e.g. {"Spain": "ES", "France": "FR", ...}.

    Example:
        >>> mapping = get_country_map()
        >>> mapping["Spain"]
        'ES'
    """
    return {
        entry["name"]: entry["code"]
        for entry in get_country_list()
    }
# End of function get_country_map()


def get_code_to_name():
    """
    Return a mapping of ISO-3166-1 alpha-2 codes to country names.

    Returns:
        dict[str, str]: A dictionary mapping codes to country names,
            e.g. {"ES": "Spain", "FR": "France", ...}.

    Example:
        >>> mapping = get_code_to_name()
        >>> mapping["ES"]
        'Spain'
    """
    return {
        entry["code"]: entry["name"]
        for entry in get_country_list()
    }
# End of function get_code_to_name()


if __name__ == "__main__":
    # Quick self-test when run directly
    countries = get_country_list()
    print(f"Total countries: {len(countries)}")
    print(f"First 5: {countries[:5]}")
    print(f"MAX_COUNTRIES: {MAX_COUNTRIES}")

    name_map = get_country_map()
    print(f"Spain -> {name_map.get('Spain', 'NOT FOUND')}")

    code_map = get_code_to_name()
    print(f"ES -> {code_map.get('ES', 'NOT FOUND')}")
