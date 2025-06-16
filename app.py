import streamlit as st
from data import RecessionDatasetBuilder, get_recessions
from models import *
from select_model import ModelSelector
import pandas as pd
import plotly.graph_objects as go

st.write("# :red[Recession]Watch")

dataset_builder = RecessionDatasetBuilder()

selected_features = st.multiselect(
    label="Features included in this model:",
    options=dataset_builder.all_features.keys(),
    default=["Real GDP", "Unemployment Rate"]
)

current_data, data = dataset_builder.create_data(features_config={
    feature: 3 for feature in selected_features
}, window=6)

X = data.drop(columns=["Recession"])
y = data["Recession"]

with st.spinner("Training..."):
    model_selector = ModelSelector([lin_reg, svm])
    model_selector.fit(X, y)
    best_model_name = model_selector.select_model(X, y, "Accuracy")
    best_model, threshold = model_selector.trained_models[best_model_name]
    current_proba = float(best_model.predict_proba(current_data)[:, 1])

    st.write(f"## Prediction: {':red[Recession]' if current_proba >= threshold else ':green[No recession]'}")
    st.write(f"Recession probability: {round(current_proba * 100, 2)}%")

    proba = best_model.predict_proba(X)[:, 1] * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.index, y=proba, mode="lines", name="recession proba"))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Predicted Recession Probability (%)",
    )

    recessions = get_recessions(start_date=dataset_builder.start_date)
    for start, end in recessions:
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="red",
            opacity=0.2,
            layer="above",
            line_width=0
        )

    fig.add_shape(
        type="line",
        x0=X.index.min(), x1=X.index.max(),
        y0=threshold*100, y1=threshold*100,
        line=dict(
            color="red",
            width=2,
            dash="dash"
        ),
        name="Threshold"
    )
    
    st.plotly_chart(fig, use_container_width=True)

