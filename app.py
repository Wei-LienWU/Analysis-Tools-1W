# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from openai import OpenAI

# 移除 emoji 函式
def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

st.set_page_config(page_title="Analysis Tools 1W", layout="centered")
st.title("Analysis Tools 1W - Automated Report Generator")
st.markdown("Upload an Excel or CSV file to automatically generate charts and GPT summary.")

openai_api_key = st.text_input("Please enter your OpenAI API Key", type="password")

uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read the file: {e}")
        st.stop()

    st.subheader("Data Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns found.")
        st.stop()

    selected_col = st.selectbox("Please select the numeric column to analyze", numeric_cols)

    st.subheader("Chart Display")
    fig, ax = plt.subplots()
    df[selected_col].plot(kind="line", ax=ax, title=f"{selected_col} Data Trend")
    st.pyplot(fig)

    if openai_api_key:
        st.subheader("GPT Analysis Summary")
        data_list = df[selected_col].dropna().tolist()[:100]
        prompt = f"""
You are a data analyst. Based on the following values in the column '{selected_col}':
{data_list}
Please write a brief analysis in English, including trend, average, maximum, minimum values, and provide recommendations.
"""

        if st.button("Generate Summary"):
            with st.spinner("Generating, please wait..."):
                try:
                    client = OpenAI(api_key=openai_api_key)
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a data analyst consultant skilled in English. Please do NOT use any emoji or special symbols in your response."
                            },
                            {"role": "user", "content": prompt}
                        ],
                    )
                    summary = response.choices[0].message.content
                    summary_clean = remove_emoji(summary)
                    st.success("Analysis complete")
                    st.markdown(summary_clean)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Please enter API Key to enable GPT analysis.")
