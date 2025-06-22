import streamlit as st
from data import create_dataset_builder
from models import MODELS
from predict import RecessionPredictor, METRICS
from helpers import format_months, format_proba, format_pred_phrase
import numpy as np
from explain import get_top_features
from features_config import FEATURES
from charts import series_chart

dataset_builder = create_dataset_builder()

# --- STATIC ---

st.set_page_config(
    page_title="RecessionWatch",
    page_icon="🚨",
    layout="wide"
)

st.markdown("# 🤬 :red[Recession]Watch")

with st.container():
    prediction, trend = st.columns(
        [1, 3], 
        border=True,
    )

with st.container():
    feature1, feature2, feature3 = st.columns([1, 1, 1], border=True)

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
        options=MODELS.keys(),
        default=MODELS.keys(),
        help="Each model is trained on the same training set. Metrics are computed with a 3-fold" \
             "walk-forward optimization split."
    )

    features = st.multiselect(
        label="Macroeconomic features",
        options=FEATURES.keys(),
        default=FEATURES.keys()
    )

    lags = st.select_slider(
        label="Feature lags",
        options=range(1, 13),
        value=3
    )

    optimization_metric = st.selectbox(
        label="Optimization metric",
        options=METRICS
    )

    st.divider()

st.markdown("😨 A pessimistic tool built by [Eric Leonen](https://github.com/ericleonen)")

# --- DYNAMIC ---

with prediction:
    with st.spinner("Predicting..."):
        X_train, y_train, X_test = dataset_builder.build(features, lags, window)
        predictor = RecessionPredictor(selected_models)
        predictor.fit(X_train, y_train, optimization_metric)

        probas_test = predictor.predict_proba(X_test)

        st.metric(
            label=f"Recession in {format_months(window)}",
            value=format_proba(float(probas_test.iloc[-1]))
        )

with trend:
    series_chart(
        series=np.round(probas_test, 3),
        name="Probability",
        color="red",
        height=250,
        title=f"Probability of U.S. Recession within {format_months(window)}"
    )

with prediction:
    with st.spinner("Explaining..."):
        top_features = get_top_features(
            predictor, 
            X_test, 
            features, 
            lags, 
            int(probas_test.iloc[-1])
        )

        st.markdown(format_pred_phrase(probas_test.iloc[-1], top_features))

for i, feature_container in enumerate([feature1, feature2, feature3]):
    with feature_container:
        top_feature = top_features[i]
        series_chart(
            series=dataset_builder.all_features[top_feature].tail(len(X_test) + lags - 1),
            name=top_feature,
            color="blue",
            height=200,
            title=top_feature
        )

        feature_description = FEATURES[top_feature]["description"]
        st.markdown(f":gray[{feature_description}]")

with model_config:
    best_model_name = predictor.best_model["name"]
    st.markdown(f"We chose **{best_model_name}** as the best model. We provide all metrics, "
                 "measured with 3-fold walk-forward validation, for all models below.")

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