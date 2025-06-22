# RecessionWatch

**RecessionWatch** is a web-based application that tracks and visualizes the probability of a U.S. recession. It uses real-time macroeconomic indicators, an automated model selection pipeline, and SHAP to create a summary of a recession prediction and the key features that drove that prediction.

Built with **Streamlit**, this app aims to make macroeconomic forecasting accessible to researchers, policymakers, investors, and the general public.

---

## Features

- Lots of customization. Under `Model Settings & Analytics`, choose:
  - The forecast window timeframe
  - Which ML models to try
  - Which macroeconomic features to use
  - How many lags per feature
  - Which metric to optimize for

- Comprehensive summary
  - Current probability of recession given latest FRED data
  - Trends in recent predicted recession probabilities
  - The top 3 features that influenced the recession prediction ranked and visualized