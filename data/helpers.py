import numpy as np
from data.fred_client import fred
import pandas as pd

def get_recessions(start_date: str | pd.Timestamp) -> np.ndarray:
    """
    Returns a 2-wide ndarray of recessionary periods starting at the given start date. Each row
    provides the start and end dates of the recession as per NBER standards.
    """
    usrec = fred.get_series("USREC")[start_date:]
    usrec = usrec.asfreq("MS")

    usrec_prev = usrec.shift(1)
    starts = list(usrec[(usrec == 1) & (usrec_prev == 0)].index)
    ends = list(usrec[(usrec == 0) & (usrec_prev == 1)].index)

    if len(ends) < len(starts):
        ends.append(usrec.index[-1])
    elif len(ends) > len(starts):
        starts.append(usrec.index[0])

    return np.vstack([starts, ends]).T