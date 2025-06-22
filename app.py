import streamlit as st
from data import create_dataset_builder
from models import MODELS
from predict import RecessionPredictor
from helpers import format_months, format_proba, format_pred_phrase
import altair as alt
import numpy as np
from explain import get_top_features
from features_config import FEATURES

dataset_builder = create_dataset_builder()

# --- STATIC ---

st.markdown("# :red[Recession]Watch")

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
        help="Each model is trained on the same training set. Metrics are computed with a 5-fold" \
             "walk-forward optimization split."
    )

    features = st.multiselect(
        label="Macroeconomic features",
        options=dataset_builder.all_features.keys(),
        default=["Real GDP growth", "Unemployment rate change", "Inflation"]
    )

    lags = st.select_slider(
        label="Feature lags",
        options=range(1, 13),
        value=3
    )

    optimization_metric = st.selectbox(
        label="Optimization metric",
        options=[
            "Average precision",
            "Weighted average precision",
            "ROC AUC",
            "Accuracy",
            "Weighted accuracy",
            "Precision",
            "Weighted precision",
            "Recall",
            "Weighted recall"
        ]
    )

    st.divider()

# --- DYNAMIC ---

with prediction:
    with st.spinner("Thinking..."):
        X_train, y_train, X_test = dataset_builder.build(features, lags, window)
        predictor = RecessionPredictor(selected_models)
        predictor.fit(X_train, y_train)

        probas_test = predictor.predict_proba(X_test)

        st.metric(
            label=f"Recession in {format_months(window)}",
            value=format_proba(float(probas_test.iloc[-1]))
        )

with trend:
    with st.spinner("Thinking..."):
        probas_df = np.round(probas_test, 3).reset_index()
        probas_df.columns = ["Date", f"Probability"]

        proba_x_axis = alt.X("Date:T", axis=alt.Axis(title=""))
        proba_y_axis = alt.Y(f"{probas_df.columns[1]}:Q", axis=alt.Axis(title=""))

        proba_area_chart = alt.Chart(probas_df).mark_area(opacity=0.4, color="red").encode(
            x=proba_x_axis,
            y=proba_y_axis
        ).properties(
            height=250,
            title=f"Probability of U.S. Recession within {format_months(window)}",
        )

        proba_line_chart = alt.Chart(probas_df).mark_line(opacity=0.8, color="red").encode(
            x=proba_x_axis,
            y=proba_y_axis
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

top_features = get_top_features(predictor, X_train, X_test.tail(1), features, lags, int(probas_test.iloc[-1]))

with prediction:
    st.markdown(format_pred_phrase(probas_test.iloc[-1], top_features))

top_feature_x_axis = alt.X("Date:T", axis=alt.Axis(title=""))

for i, feature_container in enumerate([feature1, feature2, feature3]):
    with feature_container:
        feature_series = dataset_builder.all_features[top_features[i]].tail(len(X_test) + lags - 1)
        feature_df = feature_series.reset_index()
        feature_df.columns = ["Date", "Value"]
        feature_area_chart = alt.Chart(feature_df).mark_area(opacity=0.4, color="blue").encode(
            x=top_feature_x_axis,
            y=alt.Y("Value:Q", axis=alt.Axis(title=""))
        ).properties(
            height=200,
            title=top_features[i]
        )
        feature_line_chart = alt.Chart(feature_df).mark_line(opacity=0.8, color="blue").encode(
            x=top_feature_x_axis,
            y=alt.Y("Value:Q", axis=alt.Axis(title=""))
        )
        st.altair_chart(feature_area_chart + feature_line_chart, use_container_width=True)
        
        feature_description = FEATURES[top_features[i]]["description"]
        st.markdown(f":gray[{feature_description}]")