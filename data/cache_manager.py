from datetime import date, timedelta
from typing import Optional, Tuple

def calculate_delta_dates(local_max_date: date, current_date: date) -> Optional[Tuple[date, date]]:
    """
    Calculates the date range needed to synchronize the local cache with the current date.
    
    Returns:
        A tuple of (start_date, end_date) if an update is needed.
        None if the cache is already up-to-date.
        
    Raises:
        ValueError: If local_max_date is in the future relative to current_date.
    """
    if local_max_date > current_date:
        raise ValueError(f"Local cache date ({local_max_date}) cannot be in the future relative to current date ({current_date}).")
    
    if local_max_date == current_date:
        return None
        
    return (local_max_date + timedelta(days=1), current_date)
