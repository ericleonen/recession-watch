from fred_client import fred
import pandas as pd
import numpy as np


class RecessionDatasetBuilder:
    """
    Provides an interface for creating training data selecting features,
    choosing lags, choosing the window size, and start date.
    """
    def __init__(self):
        """
        Loads potential features.
        """
        self.all_features = self._get_all_features()

    def _get_all_features(self):
        """
        Returns a dictionary mapping names to series for all potentional
        macroeconomic variables. Fills in any interior missing values with
        linear interpolation.
        """
        features = {}

        features["Real GDP"] = fred.get_series("GDPC1").pct_change(1)[1:] * 100
        features["Unemployment Rate"] = fred.get_series("UNRATE").diff(1)[1:]
        features["Nonfarm Payrolls"] = fred.get_series("PAYEMS").pct_change(1)[1:] * 100
        features["Inflation"] = fred.get_series("CORESTICKM159SFRBATL")[1:]
        # features["Industrial Production Index"] = fre  

        return features
    
    def create_data(
        self,
        features_config: dict[str, int | None],
        window=3,
        start_date: str | pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """
        Returns a DataFrame of a dataset made with the given specifications. If
        start_date is None, is set to the latest date when the first entry in
        a selected feature was recorded.
        
        Raises a ValueError if feature_config contains an unknown macroeconomic
        variable or an impossible lag value or window is an impossible value.
        """
