import pytest
from datetime import date
from data.cache_manager import calculate_delta_dates

def test_calculate_delta_dates_standard_range():
    # Assertion 1: Returns (local_max + 1, current)
    local_max_date = date(2024, 5, 10)
    current_date = date(2024, 5, 15)
    expected = (date(2024, 5, 11), date(2024, 5, 15))
    assert calculate_delta_dates(local_max_date, current_date) == expected

def test_calculate_delta_dates_identical_dates():
    # Assertion 2: Returns None if already up-to-date
    local_max_date = date(2024, 5, 15)
    current_date = date(2024, 5, 15)
    assert calculate_delta_dates(local_max_date, current_date) is None

def test_calculate_delta_dates_future_invalid():
    # Assertion 3: Raises ValueError if local cache is ahead of reality
    local_max_date = date(2024, 5, 20)
    current_date = date(2024, 5, 15)
    with pytest.raises(ValueError):
        calculate_delta_dates(local_max_date, current_date)
