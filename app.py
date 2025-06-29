# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import locale

# 設定編碼
if sys.stdout.encoding != 'utf-8':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

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
    df[selected_col].plot(kind="line", ax=ax1)
    ax1.set_title(f"{selected_col} 數據趨勢")
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
        
        # 準備數據摘要 - 使用簡單的英文鍵值
        try:
            mean_val = float(df[selected_col].mean())
            median_val = float(df[selected_col].median())
            std_val = float(df[selected_col].std())
            min_val = float(df[selected_col].min())
            max_val = float(df[selected_col].max())
            count_val = int(len(df[selected_col].dropna()))
            
            # 取樣部分數據，轉換為基本數值
            sample_data = [float(x) for x in df[selected_col].dropna().head(20).tolist()]
            
        except Exception as e:
            st.error(f"數據處理錯誤：{e}")
            st.stop()
        
        if st.button("產生 GPT 分析摘要"):
            with st.spinner("正在生成分析摘要，請稍候..."):
                try:
                    # 動態導入 OpenAI
                    from openai import OpenAI
                    
                    # 創建簡單的英文提示
                    simple_prompt = f"Analyze this data: mean={mean_val}, median={median_val}, std={std_val}, min={min_val}, max={max_val}, count={count_val}. Sample values: {sample_data}. Provide analysis in Traditional Chinese, about 200 words."
                    
                    # 初始化客戶端
                    client = OpenAI(api_key=openai_api_key)
                    
                    # 發送請求
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a data analyst. Respond in Traditional Chinese."},
                            {"role": "user", "content": simple_prompt}
                        ],
                        max_tokens=400,
                        temperature=0.5
                    )
                    
                    # 獲取並顯示結果
                    summary = response.choices[0].message.content
                    
                    st.success("分析完成！")
                    st.markdown("### GPT 分析結果")
                    
                    # 安全顯示結果
                    try:
                        st.markdown(summary)
                    except:
                        st.text(summary)
                    
                except ImportError:
                    st.error("請安裝 openai 套件：pip install openai")
                except Exception as e:
                    # 簡化錯誤處理
                    error_str = str(e)
                    st.error("GPT 分析失敗")
                    
                    if "401" in error_str or "authentication" in error_str.lower():
                        st.error("API Key 錯誤，請檢查是否正確")
                    elif "429" in error_str or "quota" in error_str.lower():
                        st.error("API 使用額度不足")
                    elif "400" in error_str:
                        st.error("請求格式錯誤")
                    else:
                        st.error("請檢查網路連線和 API Key")
                        st.text(f"詳細錯誤：{error_str[:100]}")
    else:
        st.info("請輸入 OpenAI API Key 才能使用 GPT 分析功能")
        st.markdown("**取得 API Key：**")
        st.markdown("1. 前往 https://platform.openai.com")
        st.markdown("2. 註冊帳戶並建立 API Key")

# 側邊欄
with st.sidebar:
    st.markdown("### 使用說明")
    st.markdown("1. 輸入 API Key")
    st.markdown("2. 上傳資料檔案")
    st.markdown("3. 選擇數值欄位")
    st.markdown("4. 查看分析結果")
    
    st.markdown("### 支援格式")
    st.markdown("- CSV 檔案")
    st.markdown("- Excel 檔案")
    st.markdown("- 需包含數值欄位")
