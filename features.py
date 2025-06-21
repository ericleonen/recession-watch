import pandas as pd

def growth_preprocessing(series: pd.Series) -> pd.Series:
    return series.pct_change(1).mul(100).iloc[1:].interpolate(method="linear")

def diff_preprocessing(series: pd.Series) -> pd.Series:
    return series.diff(1).iloc[1:].interpolate(method="linear")

def no_preprocessing(series: pd.Series) -> pd.Series:
    return series.interpolate(method="linear")

FEATURES = {
    "Real GDP growth": {
        "fred_series": "GDPC1",
        "description": "",
        "preprocessing": growth_preprocessing
    },
    "Unemployment rate change": {
        "fred_series": "UNRATE",
        "description": "",
        "preprocessing": diff_preprocessing
    },
    "Nonfarm payrolls growth": {
        "fred_series": "PAYEMS",
        "description": "",
        "preprocessing": growth_preprocessing
    },
    "Inflation": {
        "fred_series": "CORESTICKM159SFRBATL",
        "description": "",
        "preprocessing": no_preprocessing
    }
}