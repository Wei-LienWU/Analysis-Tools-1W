import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai

# Streamlit 網頁設定
st.set_page_config(page_title="Analysis Tools 1W", layout="centered")
st.title("📊 Analysis Tools 1W - 自動化報表生成平台")
st.markdown("上傳 Excel 或 CSV 檔案，系統將自動生成圖表與 GPT 中文分析摘要。")

# 輸入你的 OpenAI API Key
openai_api_key = st.text_input("請輸入你的 OpenAI API Key", type="password")

# 上傳檔案
uploaded_file = st.file_uploader("📁 請上傳 Excel 或 CSV 檔案", type=["csv", "xlsx"])

if uploaded_file is not None:
    # 讀取資料
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"檔案讀取失敗：{e}")
        st.stop()

    st.subheader("📋 資料預覽")
    st.dataframe(df.head())

    # 選擇欄位進行分析
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.warning("找不到數值欄位，請確認資料內容。")
        st.stop()

    selected_col = st.selectbox("📌 請選擇一個數值欄位進行圖表與摘要分析", numeric_cols)

    # 繪圖
    st.subheader("📈 自動生成圖表")
    fig, ax = plt.subplots()
    df[selected_col].plot(kind='line', title=f"{selected_col} 數據趨勢", ax=ax)
    st.pyplot(fig)

    # GPT 分析摘要
    if openai_api_key:
        st.subheader("🧠 GPT 中文分析摘要")

        data_list = df[selected_col].dropna().tolist()[:100]  # 限制長度
        prompt = f"""
你是一位數據分析師。根據以下數據欄位 {selected_col} 的數值列表：
{data_list}
請用繁體中文撰寫一段簡潔的分析摘要，包含趨勢變化、平均值、最高與最低值，並給出一項建議。
"""

import openai
from openai import OpenAI

# 建立 OpenAI client
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)

    if st.button("✨ 產生摘要"):
        with st.spinner("GPT 正在撰寫摘要..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "你是一位善於中文資料分析的專業顧問。"},
                        {"role": "user", "content": prompt}
                    ]
                )
                summary = response.choices[0].message.content
                st.success("✅ 分析完成")
                st.markdown(summary)
            except Exception as e:
                st.error(f"錯誤：{e}")
