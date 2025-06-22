import pandas as pd

def base_preprocessing(series: pd.Series) -> pd.Series:
    """
    Returns the given series where missing values have been interpolated linearly.
    """
    return series.dropna().interpolate(method="linear")

def growth_preprocessing(series: pd.Series) -> pd.Series:
    """
    Returns the given series with base preprocessing as percent changes.
    """
    return base_preprocessing(series.pct_change(1).mul(100))

def diff_preprocessing(series: pd.Series) -> pd.Series:
    """
    Returns the given series with base preprocessing as differences.
    """
    return base_preprocessing(series.diff(1))

FEATURES = {
    "Real GDP growth": {
        "fred_series": "GDPC1",
        "description": "Quarterly percent change in real gross domestic product",
        "preprocessing": growth_preprocessing
    },
    "Unemployment rate change": {
        "fred_series": "UNRATE",
        "description": "Monthly difference in unemployment rate (U3)",
        "preprocessing": diff_preprocessing
    },
    "Nonfarm payrolls growth": {
        "fred_series": "PAYEMS",
        "description": "Monthly percent change in number of nonfarm workers",
        "preprocessing": growth_preprocessing
    },
    "Inflation": {
        "fred_series": "CORESTICKM159SFRBATL",
        "description": "Yearly percent change in Sticky Consumer Price Index (less food and energy)",
        "preprocessing": base_preprocessing
    },
    "Industrial production growth": {
        "fred_series": "INDPRO",
        "description": "Monthly percent change in the industrial production index",
        "preprocessing": growth_preprocessing
    },
    "Yield spread": {
        "fred_series": "T10Y3M",
        "description": "Difference between 10-year and 3-month Treasury interest rates",
        "preprocessing": base_preprocessing
    },
    "Federal funds rate change": {
        "fred_series": "FEDFUNDS",
        "description": "Monthly difference in overnight lending interest rate for Federal Reserve Banks",
        "preprocessing": base_preprocessing
    },
    "Consumer sentiment change": {
        "fred_series": "UMCSENT",
        "description": "Monthly difference in University of Michigan's consumer sentiment index",
        "preprocessing": diff_preprocessing
    },
    "Consumer credit growth": {
        "fred_series": "TOTALSL",
        "description": "Monthly percent change in total consumer credit owned and securitized",
        "preprocessing": growth_preprocessing
    }
}