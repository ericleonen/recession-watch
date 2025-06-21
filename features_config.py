import pandas as pd

def base_preprocessing(series: pd.Series) -> pd.Series:
    return series.dropna().interpolate(method="linear")

def growth_preprocessing(series: pd.Series) -> pd.Series:
    return base_preprocessing(series.pct_change(1).mul(100))

def diff_preprocessing(series: pd.Series) -> pd.Series:
    return base_preprocessing(series.diff(1))

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
        "preprocessing": base_preprocessing
    }
}