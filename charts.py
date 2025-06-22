import pandas as pd
import altair as alt
import streamlit as st

def series_chart(
    series: pd.Series, 
    name: str,
    color: str,
    height: float,
    title: str
):
    """
    Plots an Altair chart of the given series with the given series name, color, chart
    height and title.
    """
    df = series.reset_index()
    df.columns = ["Date", name]

    x_axis = alt.X("Date:T", axis=alt.Axis(title=""))
    y_axis = alt.Y(f"{name}:Q", axis=alt.Axis(title=""))

    area_chart = alt.Chart(df).mark_area(opacity=0.4, color=color).encode(
        x=x_axis,
        y=y_axis
    )
    line_chart = alt.Chart(df).mark_line(opacity=0.8, color=color).encode(
        x=x_axis,
        y=y_axis
    )
    chart = (area_chart + line_chart).properties(
        height=height,
        title=title
    )

    return st.altair_chart(chart, use_container_width=True)