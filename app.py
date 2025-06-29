# -*- coding: utf-8 -*-
import sys
import io

# 強制 stdout/stderr 用 utf-8 編碼，避免 ASCII 編碼錯誤
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

st.set_page_config(page_title="Analysis Tools 1W", layout="centered")
st.title("Analysis Tools 1W - Automated Report Generator")
st.markdown("Upload an Excel or CSV file to automatically generate charts and GPT summary.")

# User input API Key
openai_api_key = st.text_input("Please enter your OpenAI API Key", type="password")

# File upload
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
                            {"role": "system", "content": "You are a data analyst consultant skilled in English. Please do NOT use any emoji or special symbols in your response."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    summary = response.choices[0].message.content
                    # 移除所有非 ASCII 字元，避免編碼錯誤
                    summary_clean = summary.encode('ascii', errors='ignore').decode()
                    st.success("Analysis complete")
                    st.markdown(summary_clean)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Please enter API Key to enable GPT analysis.")
