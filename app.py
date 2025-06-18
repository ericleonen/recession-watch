import streamlit as st
from data import RecessionDatasetBuilder
from models import models
from predict import RecessionPredictor
from helpers import format_months, format_proba

dataset_builder = RecessionDatasetBuilder()

# --- STATIC ---

st.markdown("# :red[Recession]Watch")

with st.container():
    prediction, trend = st.columns(
        [1, 3], 
        border=True,
    )

with st.container():
    feature1, feature2, feature3 = st.columns([1, 1, 1], border=True)

    with feature1:
        "FEATURE 1"
    
    with feature2:
        "FEATURE 2"

    with feature3:
        "FEATURE 3"

with st.expander(label="Model Settings & Analytics"):
    st.markdown(
        "**:red[Recession]Watch** works by training several models on a set of macroeconomic " \
        "features from historical data to find a model that optimizes a metric. Use the inputs " \
        "below to adjust the modelling process."
    )

    window = st.select_slider(
        label="Forecast window size",
        options=range(3, 49, 3),
        format_func=format_months,
        value=12
    )
    
    selected_models = st.multiselect(
        label="Models to try",
        options=models.keys(),
        default=["Logistic Regression"],
        help="Each model is trained on the same training set. Metrics are computed with a 5-fold" \
             "walk-forward optimization split."
    )

    features = st.multiselect(
        label="Macroeconomic features",
        options=dataset_builder.all_features.keys(),
        default=["Real GDP", "Unemployment Rate"],
        help="Certain features are differenced when sensisble. All features are given 3 lags."
    )

    optimization_metric = st.selectbox(
        label="Optimization metric",
        options=["Average Precision", "Weighted Average Precision", "ROC AUC"]
    )
    
    st.button(
        label="Run & Predict"
    )

    st.divider()

    st.markdown("We chose :red-badge[SVM] as the best model. We provide all metrics for all models below.")

# --- DYNAMIC ---

with prediction:
    with st.spinner("Thinking..."):
        X, y, X_now = dataset_builder.create_data({
            feature: 3 for feature in features
        }, window=window)

        predictor = RecessionPredictor(selected_models=selected_models)
        predictor.fit(X, y)

        proba_now = predictor.predict_proba(X_now)

        st.metric(
            label=f"US Recession in {format_months(window)}",
            value=f"{format_proba(float(proba_now.iloc[0]))}"
        )

with trend:
    with st.spinner("Thinking..."):
        threshold = predictor.best_model["threshold"]
        probas = predictor.predict_proba(X)

        st.line_chart(data=probas)