import streamlit as st

st.markdown("# :red[Recession]Watch")

with st.container():
    results, explanations = st.columns(
        [1, 3], 
        border=True,
    )

    with results:
        st.metric(
            label="U.S. Recession in 1 year",
            value="23.82%",
            delta="-5.32%",
            delta_color="inverse"
        )

    with explanations:
        feature1, feature2, feature3 = st.columns([1, 1, 1])

with st.expander(label="Model Settings & Analytics"):
    st.markdown(
        "**:red[Recession]Watch** works by training several models on a set of macroeconomic " \
        "features from historical data to find a model that optimizes a metric. Use the inputs " \
        "below to adjust the modelling process."
    )
    
    models = st.multiselect(
        label="Models to try",
        options=["Logistic Regression", "SVM", "XGBoost"],
        default=["Logistic Regression", "SVM", "XGBoost"],
        help="Each model is trained on the same training set. Metrics are computed with a 5-fold" \
        "walk-forward optimization split."
    )

    features = st.multiselect(
        label="Macroeconomic features",
        options=["Real GDP", "Unemployment Rate"],
        default=["Real GDP", "Unemployment Rate"],
        help="Certain features are differenced when sensisble. All features are given 3 lags."
    )

    optimization_metric = st.selectbox(
        label="Optimization metric",
        options=["Average Precision", "Weighted Average Precision", "ROC AUC"]
    )

    st.empty()
    
    st.button(
        label="Run & Predict"
    )

    st.divider()

    st.markdown("We chose :red-badge[SVM] as the best model. We provide all metrics for all models below.")

