# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

# 頁面設定
st.set_page_config(page_title="Analysis Tools 1W", layout="centered")
st.title("Analysis Tools 1W - 自動報表產生工具")
st.markdown("上傳 Excel 或 CSV 檔案，自動產生圖表與 GPT 中文摘要。")

# 使用者輸入 API Key
openai_api_key = st.text_input("請輸入你的 OpenAI API Key", type="password")

# 檔案上傳
uploaded_file = st.file_uploader("請上傳 Excel 或 CSV 檔案", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"檔案讀取失敗：{e}")
        st.stop()

    st.subheader("資料預覽")
    st.dataframe(df.head())

    # 選擇數值欄位
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.warning("找不到數值欄位")
        st.stop()

    selected_col = st.selectbox("請選擇要分析的數值欄位", numeric_cols)

    # 顯示圖表
    st.subheader("圖表呈現")
    fig, ax = plt.subplots()
    df[selected_col].plot(kind="line", ax=ax, title=f"{selected_col} 數據趨勢")
    st.pyplot(fig)

    # GPT 分析
    if openai_api_key:
        st.subheader("GPT 分析摘要")
        data_list = df[selected_col].dropna().tolist()[:100]
        prompt = f"""
你是一位數據分析師。根據以下欄位 {selected_col} 的數值：
{data_list}
請用繁體中文撰寫簡要分析，包括趨勢、平均值、最大最小值，並提供建議。
"""

        if st.button("產生摘要"):
            with st.spinner("生成中，請稍候..."):
                try:
                    client = OpenAI(api_key=openai_api_key)
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "你是一位善於分析資料的中文顧問。"},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    summary = response.choices[0].message.content
                    st.success("分析完成")
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"錯誤：{e}")
    else:
        st.info("請輸入 API Key 才能使用 GPT 分析功能")
