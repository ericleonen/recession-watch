import streamlit as st
from data import RecessionDatasetBuilder
from models import models
from predict import RecessionPredictor
from helpers import format_months, format_proba, proba_to_phrase
import altair as alt
import numpy as np
import pandas as pd

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

model_config = st.expander(label="Model Settings & Analytics")

with model_config:
    st.markdown(
        "**:red[Recession]Watch** works by training several models on a set of macroeconomic " \
        "features from historical data to find a model that optimizes a metric. Use the inputs " \
        "below to adjust the modelling process."
    )

    window = st.select_slider(
        label="Forecast window size",
        options=range(3, 25, 3),
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
        help="Certain features are differenced when sensisble. All features are given 2 lags."
    )

    optimization_metric = st.selectbox(
        label="Optimization metric",
        options=[
            "Average Precision",
            "Weighted Average Precision",
            "ROC AUC",
            "Accuracy",
            "Weighted Accuracy",
            "Precision",
            "Weighted Precision",
            "Recall",
            "Weighted Recall"
        ]
    )

    st.divider()

# --- DYNAMIC ---

with prediction:
    with st.spinner("Thinking..."):
        X_train, y_train, X_test = dataset_builder.create_data({
            feature: 2 for feature in features
        }, window=window)

        predictor = RecessionPredictor(selected_models=selected_models)
        predictor.fit(X_train, y_train)

        probas_test = predictor.predict_proba(X_test)

        st.metric(
            label=f"US Recession in {format_months(window)}",
            value=format_proba(float(probas_test.iloc[-1]))
        )

        st.markdown(f"A recession is **{proba_to_phrase(probas_test.iloc[-1])}** due to " \
                     "{}, {}, and {}")

with trend:
    with st.spinner("Thinking..."):
        probas_df = np.round(probas_test, 3).reset_index()
        probas_df.columns = ["Date", f"Probability"]

        proba_area_chart = alt.Chart(probas_df).mark_area(opacity=0.4, color="red").encode(
            x="Date:T",
            y=f"{probas_df.columns[1]}:Q"
        ).properties(
            height=300,
            title=f"Probability of U.S. Recession within {format_months(window)}",
        )

        proba_line_chart = alt.Chart(probas_df).mark_line(opacity=0.8, color="red").encode(
            x="Date:T",
            y=f"{probas_df.columns[1]}:Q"
        )

        st.altair_chart(proba_area_chart + proba_line_chart, use_container_width=True)

with model_config:
    best_model_name = predictor.best_model["name"]
    st.markdown(f"We chose **{best_model_name}** as the best model. We provide all metrics, "
                 "measured with 5-fold walk-forward validation, for all models below.")

    metrics_table = predictor.model_table.copy()
    metrics_table.index = [
        f"⭐ {optimization_metric}" if metric == optimization_metric else metric
        for metric in metrics_table.index
    ]
    metrics_table.columns = [
        f"⭐ {best_model_name}" if name == best_model_name else name
        for name in metrics_table.columns
    ]

    st.write(metrics_table)