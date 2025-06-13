from data.fred_client import fred
import pandas as pd
import numpy as np
from data.helpers import get_recessions


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
        window=3
    ) -> pd.DataFrame:
        """
        Returns a DataFrame of a dataset made with the given specifications.
        
        Raises a ValueError if feature_config contains an unknown macroeconomic
        variable or an impossible lag value or window is an impossible value.
        """
        # check for invalid inputs
        for feature in features_config.keys():
            lags = features_config[feature]

            if feature not in self.all_features:
                raise ValueError(f"featured_config contains unknown feature: {feature}")
            elif lags is not None and lags <= 0:
                raise ValueError(f"lag for {feature} in features_config is {lags}, but it must \
                                   be postive")

        # start date is the latest first recorded date of all selected features plus window
        start_date = pd.Timestamp(max(
            self.all_features[f].min() for f in self.all_features if f in features_config
        )).normalize() + pd.DateOffset(months=window)
        target = self._get_target(start_date, window)

        data = []
        
        for month in target.index:
            row = []

            for feature, lags in features_config.items():
                series = self.all_features[feature]
                row.extend(list(series[series.index <= month][-lags:]))

            row.append(target[month])

            data.append(row)

        # create column names
        columns = []
        for feature, lags in features_config.items():
            for t in range(lags, 0, -1):
                columns.append(f"{feature} (t-{t})")
        columns.append(target.name)

        return pd.DataFrame(
            data,
            columns=columns,
            index=target.index
        )      

    def _get_target(self, start_date: str | pd.Timestamp, window: int):
        recessions = get_recessions(start_date)

        month_offset = pd.DateOffset(months=window)
        end_date = pd.Timestamp.today().normalize() - month_offset
        months = pd.date_range(start=start_date, end=end_date, freq="MS")

        labels = []

        for start_month in months:
            end_month = start_month + month_offset
            rec_in_window = any(
                (start_month <= recession_start) and (recession_start <= end_month)
                for recession_start, _ in recessions
            )

            labels.append(int(rec_in_window))

        return pd.Series(labels, index=months, name=f"Recession")