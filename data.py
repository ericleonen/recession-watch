from fred_client import fred
import pandas as pd
import numpy as np
from features_config import FEATURES
import streamlit as st

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
    lags, and recession window.
    """
    def __init__(self):
        """
        Initializes a RecessionDatasetBuilder by loading all possible macroeconomic variables.
        """
        self.all_features = {
            feature_name: feature["preprocessing"](fred.get_series(feature["fred_series"])) 
            for feature_name, feature in FEATURES.items()
        }
        self.start_date = None
        self.end_date = None
    
    def build(
        self,
        features: list[str],
        lags: int,
        window: int = 3
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Constructs training (X_train, y_train) and testing (X_test) sets with specified features
        and lags. Sets start_date to the latest date where a feature was first recorded. Sets
        end_date to the earliest date where a feature was last recorded.

        Args:
            selected_features: features to include in the dataset
            lags: number of lags used per feature
            window: number of months into the future to check for a recession

        Returns:
            X_train: DataFrame of features and lags from start_date to end_date
            y_train: Series where 1 indicates a recession happening within the window, 0 otherwise
            X_test: DataFrame of features and lags past end_date
        """
        self.start_date = max(
            self.all_features[feature].index[lags - 1] for feature in features
        ) 
        y_train = self._get_target(self.start_date, window)
        
        self.end_date = min(y_train.index[-1], min(
            self.all_features[feature].index.max() for feature in features
        ))
        y_train = y_train[y_train.index <= self.end_date]

        X = []
        dates = pd.date_range(start=self.start_date, end=pd.Timestamp.today(), freq="MS")
        
        for month in dates:
            row = []

            for feature in features:
                series = self.all_features[feature]
                values = series[series.index <= month].tail(lags)

                row.extend(list(values))

            X.append(row)

        column_names = [
            f"{feature} (t-{lag})"
            for feature in features
            for lag in range(lags-1, -1, -1)
        ]

        X = pd.DataFrame(
            X,
            columns=column_names,
            index=dates
        )

        X_train = X[X.index <= self.end_date]
        X_test = X[X.index > self.end_date]

        return X_train, y_train, X_test
            

    def _get_target(self, start_date: str | pd.Timestamp, window: int) -> pd.Series:
        """
        Creates the target Series: 1 if a recession starts within the next window months, 0
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

@st.cache_resource
def create_dataset_builder():
    """
    Initializes a RecessionDatasetBuilder by loading all possible macroeconomic variables. This
    function is cached.
    """
    return RecessionDatasetBuilder()