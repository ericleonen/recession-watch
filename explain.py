import shap
from predict import RecessionPredictor
import pandas as pd

def get_top_features(
    predictor: RecessionPredictor, 
    X_test: pd.DataFrame,
    features: list[str],
    lags: int,
    pred: int
) -> list[str]:
    """
    Returns a list of the top three (at most) influential features in the latest prediction from
    the given predictor. The baseline is calculated from X_test, features defines the possible
    features, lags is the number of lags per feature, and pred (1 or 0) was whether the latest
    prediction was a recession forecast or not respectively.
    """
    X_recent = X_test.tail(1)
    X_test = X_test.iloc[:-1]

    explainer = shap.Explainer(predictor.predict_proba, X_test)
    shap_values = explainer(X_recent)

    feature_shaps = pd.Series(
        shap_values.values.reshape(-1, lags).sum(axis=1), 
        index=features,
    ).sort_values(ascending=1-pred)

    return list(feature_shaps.index[:3])