import pandas as pd
import plotly.express as px
import streamlit as st


def choose_chart(data, chart_requested=False, title=""):
    """
    Automatically choose a suitable chart based on the data.

    Returns:
        dict if a chart should be displayed
        None if a chart is not appropriate
    """

    if not chart_requested:
        return None

    if not data:
        return None

    # Remove zero values
    clean_data = {
        key: value
        for key, value in data.items()
        if value is not None and value > 0
    }

    # A chart needs at least two meaningful categories
    if len(clean_data) < 2:
        return None

    # RACI distribution is naturally a part-to-whole chart
    raci_keys = {"R", "A", "C", "I"}

    if set(clean_data.keys()).issubset(raci_keys):
        return {
            "type": "pie",
            "title": title or "Distribusi RACI",
            "data": clean_data,
        }

    # Other categorical numerical data -> bar chart
    return {
        "type": "bar",
        "title": title or "Data",
        "data": clean_data,
    }


def render_chart(chart_data):
    """
    Render the chart selected by choose_chart().
    """

    if not chart_data:
        return

    chart_type = chart_data["type"]

    df = pd.DataFrame(
        [
            {"category": key, "value": value}
            for key, value in chart_data["data"].items()
        ]
    )

    if chart_type == "pie":
        fig = px.pie(
            df,
            names="category",
            values="value",
            title=chart_data["title"],
        )

    elif chart_type == "bar":
        fig = px.bar(
            df,
            x="category",
            y="value",
            title=chart_data["title"],
        )

    else:
        return

    st.plotly_chart(
        fig,
        use_container_width=True,
    )