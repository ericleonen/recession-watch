from fred_client import fred
import pandas as pd
import numpy as np

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

class RecessionDatasetBuilder:
    """
    Builds training data for recession prediction using selected macroeconomic features,
    their lags, a target window, and a custom start date.
    """
    def __init__(self):
        """
        Initializes and loads available macroeconomic features.
        """
        self.all_features = self._load_all_features()

    def _load_all_features(self) -> dict[str, pd.Series]:
        """
        Fetches and preprocesses macroeconomic time series data. Each series is either reported
        monthly or quarterly. Gaps are imputed with linear interpolation.
        """
        features = {
            "Real GDP": fred.get_series("GDPC1").pct_change(1).mul(100).iloc[1:] \
                .interpolate(method="linear"),
            "Unemployment Rate": fred.get_series("UNRATE").diff(1).iloc[1:] \
                .interpolate(method="linear"),
            "Nonfarm Payrolls": fred.get_series("PAYEMS").pct_change(1).mul(100).iloc[1:] \
                .interpolate(method="linear"),
            "Inflation": fred.get_series("CORESTICKM159SFRBATL").interpolate(method="linear")
        } 

        return features
    
    def create_data(
        self,
        features_config: dict[str, int | None],
        window: int = 3
    ) -> pd.DataFrame:
        """
        Constructs a dataset with specified features and lags. Each row is labeled with a binary
        target indicated if a recession occurs within the future window.

        Args:
            features_config: dict mapping feature names to number of lags
            window: number of months into the future to check for a recession

        Returns:
            pd.DataFrame: Feature matrix with target labels indexed by months

        Raises:
            ValueError: If invalid feature names or lag values are provided
        """
        self._validate_data_config(features_config, window)

        start_date = max(
            self.all_features[feature].index[lags - 1] for feature, lags in features_config.items()
        ) 
        target = self._get_target(start_date, window)
        
        end_date = min(target.index[-1], min(
            self.all_features[feature].index.max() for feature in features_config.keys()
        ))
        target = target[target.index <= end_date]

        data = []
        
        for month in target.index:
            row = []

            for feature, lags in features_config.items():
                series = self.all_features[feature]
                values = series[series.index <= month].tail(lags)

                row.extend(list(values))

            row.append(target[month])
            data.append(row)

        column_names = [
            f"{feature} (t-{lag})"
            for feature, lags in features_config.items()
            for lag in range(lags - 1, -1, -1)
        ] + [target.name]

        return pd.DataFrame(
            data,
            columns=column_names,
            index=target.index
        )      

    def _validate_data_config(self, features_config: dict[str, int | None], window: int):
        """
        Raises a ValueError if feature_config has invalid feature names or lag values or window
        is not positve.
        """
        if window <= 0:
            raise ValueError(f"Window must be a positive integer. Got {window}.")

        for feature, lags in features_config.items():
            if feature not in self.all_features:
                raise ValueError(f"Unknown feature in config: {feature}")
            if lags is not None:
                if lags <= 0:
                    raise ValueError(f"Lag for feature '{feature}' must be positive. Got {lags}.")
                elif lags > len(self.all_features[feature]):
                    raise ValueError(
                        f"Lag for feature '{feature}' can be at most "
                        f"{len(self.all_features[feature])}. Got {lags}"
                    )
            

    def _get_target(self, start_date: str | pd.Timestamp, window: int) -> pd.Series:
        """
        Creates the target series: 1 if a recession starts within the next window months, 0
        otherwise.
        """
        recessions = get_recessions(start_date)

        month_offset = pd.DateOffset(months=window)
        end_date = pd.Timestamp.today().normalize() - month_offset
        months = pd.date_range(start=start_date, end=end_date, freq="MS")

        def recession_in_window(start: pd.Timestamp) -> bool:
            end = start + pd.DateOffset(months=window)

            return any(
                (start <= recession_start) and (recession_start <= end)
                for recession_start, _ in recessions
            )

        labels = [int(recession_in_window(month)) for month in months]
        
        return pd.Series(labels, index=months, name=f"Recession")