# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import numpy as np

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
        # 讀取檔案
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("檔案讀取成功！")
        
    except UnicodeDecodeError:
        # 如果 UTF-8 失敗，嘗試其他編碼
        try:
            df = pd.read_csv(uploaded_file, encoding='big5')
            st.success("檔案讀取成功！")
        except:
            try:
                df = pd.read_csv(uploaded_file, encoding='gbk')
                st.success("檔案讀取成功！")
            except Exception as e:
                st.error(f"檔案讀取失敗：{e}")
                st.stop()
    except Exception as e:
        st.error(f"檔案讀取失敗：{e}")
        st.stop()

    # 資料預覽
    st.subheader("資料預覽")
    st.dataframe(df.head())
    
    # 顯示基本資訊
    st.write(f"**資料筆數：** {len(df)}")
    st.write(f"**欄位數量：** {len(df.columns)}")

    # 選擇數值欄位
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("找不到數值欄位，請確認檔案是否包含數值資料")
        st.stop()

    selected_col = st.selectbox("請選擇要分析的數值欄位", numeric_cols)

    # 顯示基本統計
    st.subheader("基本統計資訊")
    stats = df[selected_col].describe()
    st.write(stats)

    # 顯示圖表
    st.subheader("圖表呈現")
    
    # 創建圖表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 線圖
    df[selected_col].plot(kind="line", ax=ax1, title=f"{selected_col} 數據趨勢")
    ax1.set_xlabel("索引")
    ax1.set_ylabel(selected_col)
    
    # 直方圖
    df[selected_col].hist(ax=ax2, bins=20)
    ax2.set_title(f"{selected_col} 分布")
    ax2.set_xlabel(selected_col)
    ax2.set_ylabel("頻率")
    
    plt.tight_layout()
    st.pyplot(fig)

    # GPT 分析
    if openai_api_key:
        st.subheader("GPT 分析摘要")
        
        # 準備數據摘要
        data_summary = {
            "平均值": round(df[selected_col].mean(), 2),
            "中位數": round(df[selected_col].median(), 2),
            "標準差": round(df[selected_col].std(), 2),
            "最小值": round(df[selected_col].min(), 2),
            "最大值": round(df[selected_col].max(), 2),
            "資料筆數": len(df[selected_col].dropna())
        }
        
        # 取樣部分數據
        sample_data = df[selected_col].dropna().head(50).tolist()
        
        prompt = f"""
Analyze the following data for column: {clean_selected_col if 'clean_selected_col' in locals() else selected_col}

Statistics:
- Mean: {data_summary['平均值']}
- Median: {data_summary['中位數']}
- Std Dev: {data_summary['標準差']}
- Min: {data_summary['最小值']}
- Max: {data_summary['最大值']}
- Count: {data_summary['資料筆數']}

Sample data: {sample_data}

Please provide analysis in Traditional Chinese including:
1. Overall trends and characteristics
2. Data distribution
3. Any outliers
4. Practical recommendations

Keep it concise, around 200-300 words.
"""
        
        if st.button("產生 GPT 分析摘要"):
            with st.spinner("正在生成分析摘要，請稍候..."):
                try:
                    # 清理 prompt 中的特殊字符
                    clean_prompt = prompt.encode('utf-8', errors='ignore').decode('utf-8')
                    clean_selected_col = selected_col.encode('ascii', errors='ignore').decode('ascii')
                    if not clean_selected_col:
                        clean_selected_col = "selected_column"
                    
                    # 初始化 OpenAI 客戶端
                    client = OpenAI(api_key=openai_api_key)
                    
                    # 發送請求
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system", 
                                "content": "You are a professional data analyst. Please provide analysis in Traditional Chinese."
                            },
                            {
                                "role": "user", 
                                "content": clean_prompt
                            }
                        ],
                        max_tokens=500,
                        temperature=0.7
                    )
                    
                    # 獲取回應並清理
                    summary = response.choices[0].message.content
                    if summary:
                        summary = summary.encode('utf-8', errors='ignore').decode('utf-8')
                    
                    st.success("分析完成！")
                    st.markdown("### GPT 分析結果")
                    st.markdown(summary)
                    
                except Exception as e:
                    error_msg = str(e).encode('ascii', errors='ignore').decode('ascii')
                    st.error(f"GPT 分析失敗：{error_msg}")
                    
                    # 提供詳細錯誤資訊
                    error_lower = error_msg.lower()
                    if "authentication" in error_lower or "unauthorized" in error_lower:
                        st.error("請檢查 API Key 是否正確")
                    elif "quota" in error_lower or "billing" in error_lower:
                        st.error("API 配額不足，請檢查你的 OpenAI 帳戶")
                    elif "model" in error_lower:
                        st.error("模型不可用，請稍後再試")
                    else:
                        st.error("請檢查網路連線或重新嘗試")
    else:
        st.info("請輸入 OpenAI API Key 才能使用 GPT 分析功能")
        st.markdown("**如何獲取 API Key：**")
        st.markdown("1. 前往 [OpenAI 官網](https://platform.openai.com)")
        st.markdown("2. 註冊/登入帳戶")
        st.markdown("3. 到 API Keys 頁面建立新的 API Key")

# 側邊欄資訊
with st.sidebar:
    st.markdown("### 使用說明")
    st.markdown("1. 輸入 OpenAI API Key")
    st.markdown("2. 上傳 CSV 或 Excel 檔案")
    st.markdown("3. 選擇要分析的數值欄位")
    st.markdown("4. 查看圖表和統計資訊")
    st.markdown("5. 點擊產生 GPT 分析摘要")
    
    st.markdown("### 注意事項")
    st.markdown("- 支援 CSV 和 Excel 格式")
    st.markdown("- 需要包含數值欄位")
    st.markdown("- API Key 不會被儲存")
    st.markdown("- 大檔案可能需要較長處理時間")
