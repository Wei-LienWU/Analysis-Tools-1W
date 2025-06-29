# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import locale
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 設定編碼
if sys.stdout.encoding != 'utf-8':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 頁面設定
st.set_page_config(page_title="智能數據分析工具", layout="wide")
st.title("🔍 智能數據分析工具")
st.markdown("上傳數據檔案，獲得專業級自動分析報告 - 無需 API Key！")

# 分析函數
def generate_data_insights(df, column):
    """生成數據洞察報告"""
    insights = []
    data = df[column].dropna()
    
    if len(data) == 0:
        return ["所選欄位無有效數據"]
    
    # 基本統計
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    min_val = data.min()
    max_val = data.max()
    
    # 1. 集中趨勢分析
    if abs(mean_val - median_val) / std_val < 0.5:
        insights.append("📊 **集中趨勢**: 數據分布相對對稱，平均值與中位數接近")
    elif mean_val > median_val:
        insights.append("📊 **集中趨勢**: 數據呈現右偏分布，存在較大的極值拉高平均值")
    else:
        insights.append("📊 **集中趨勢**: 數據呈現左偏分布，存在較小的極值拉低平均值")
    
    # 2. 離散程度分析
    cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')
    if cv < 0.1:
        insights.append("📈 **變異性**: 數據變化很小，相當穩定")
    elif cv < 0.3:
        insights.append("📈 **變異性**: 數據變化適中，具有一定穩定性")
    else:
        insights.append("📈 **變異性**: 數據變化較大，波動性高")
    
    # 3. 異常值檢測
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    if len(outliers) > 0:
        outlier_ratio = len(outliers) / len(data) * 100
        insights.append(f"⚠️ **異常值**: 發現 {len(outliers)} 個異常值 ({outlier_ratio:.1f}%)，建議進一步檢查")
    else:
        insights.append("✅ **異常值**: 未發現明顯異常值，數據品質良好")
    
    # 4. 分布特徵
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    if abs(skewness) < 0.5:
        skew_desc = "近似對稱"
    elif skewness > 0.5:
        skew_desc = "右偏（正偏）"
    else:
        skew_desc = "左偏（負偏）"
    
    insights.append(f"📋 **分布形狀**: {skew_desc}，峰度 = {kurtosis:.2f}")
    
    # 5. 數據範圍分析
    data_range = max_val - min_val
    range_ratio = data_range / mean_val if mean_val != 0 else float('inf')
    insights.append(f"📏 **數據範圍**: {data_range:.2f}，約為平均值的 {range_ratio:.1f} 倍")
    
    # 6. 實務建議
    if cv > 0.5:
        insights.append("💡 **建議**: 數據波動較大，建議分組分析或尋找影響因素")
    elif len(outliers) > len(data) * 0.1:
        insights.append("💡 **建議**: 異常值較多，建議檢查數據收集過程")
    else:
        insights.append("💡 **建議**: 數據品質良好，適合進一步統計分析")
    
    return insights

def detect_patterns(df, column):
    """檢測數據模式"""
    patterns = []
    data = df[column].dropna()
    
    # 趨勢分析（如果數據有時間順序）
    if len(data) > 10:
        # 簡單趨勢檢測
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        if abs(r_value) > 0.3 and p_value < 0.05:
            if slope > 0:
                patterns.append(f"📈 **趨勢**: 數據呈現上升趨勢 (相關係數: {r_value:.3f})")
            else:
                patterns.append(f"📉 **趨勢**: 數據呈現下降趨勢 (相關係數: {r_value:.3f})")
        else:
            patterns.append("➡️ **趨勢**: 無明顯線性趨勢")
    
    # 週期性檢測（簡單版本）
    if len(data) > 20:
        # 檢查是否有重複的峰值模式
        rolling_mean = pd.Series(data).rolling(window=5).mean()
        peaks_valleys = np.diff(np.sign(np.diff(rolling_mean)))
        peak_count = np.sum(peaks_valleys < 0)
        
        if peak_count > len(data) // 10:
            patterns.append("🔄 **週期性**: 數據可能存在週期性波動")
    
    return patterns

def advanced_statistics(df, column):
    """進階統計分析"""
    data = df[column].dropna()
    stats_info = {}
    
    # 常用統計量
    stats_info['基本統計'] = {
        '觀測數': len(data),
        '平均值': f"{data.mean():.4f}",
        '中位數': f"{data.median():.4f}",
        '標準差': f"{data.std():.4f}",
        '變異係數': f"{data.std()/abs(data.mean()):.4f}" if data.mean() != 0 else "N/A",
        '最小值': f"{data.min():.4f}",
        '最大值': f"{data.max():.4f}",
        '範圍': f"{data.max() - data.min():.4f}"
    }
    
    # 分位數
    stats_info['分位數分析'] = {
        '第10百分位': f"{data.quantile(0.1):.4f}",
        '第25百分位 (Q1)': f"{data.quantile(0.25):.4f}",
        '第50百分位 (中位數)': f"{data.quantile(0.5):.4f}",
        '第75百分位 (Q3)': f"{data.quantile(0.75):.4f}",
        '第90百分位': f"{data.quantile(0.9):.4f}",
        '四分位距 (IQR)': f"{data.quantile(0.75) - data.quantile(0.25):.4f}"
    }
    
    # 分布特徵
    stats_info['分布特徵'] = {
        '偏度 (Skewness)': f"{stats.skew(data):.4f}",
        '峰度 (Kurtosis)': f"{stats.kurtosis(data):.4f}",
        '變異數': f"{data.var():.4f}",
        '標準誤': f"{data.std()/np.sqrt(len(data)):.4f}"
    }
    
    # 常態性檢定
    try:
        shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(5000, len(data))))
        stats_info['統計檢定'] = {
            'Shapiro-Wilk 統計量': f"{shapiro_stat:.4f}",
            'Shapiro-Wilk p值': f"{shapiro_p:.4f}",
            '常態性': "可能符合常態分布" if shapiro_p > 0.05 else "不符合常態分布"
        }
    except:
        stats_info['統計檢定'] = {'註記': '無法執行常態性檢定'}
    
    return stats_info

# 檔案上傳
uploaded_file = st.file_uploader("請上傳 Excel 或 CSV 檔案", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # 讀取檔案
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("✅ 檔案讀取成功！")
        
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(uploaded_file, encoding='big5')
            st.success("✅ 檔案讀取成功！")
        except:
            try:
                df = pd.read_csv(uploaded_file, encoding='gbk')
                st.success("✅ 檔案讀取成功！")
            except Exception as e:
                st.error(f"❌ 檔案讀取失敗：{e}")
                st.stop()
    except Exception as e:
        st.error(f"❌ 檔案讀取失敗：{e}")
        st.stop()

    # 資料預覽
    st.subheader("📋 資料預覽")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**資料筆數：** {len(df):,}")
        st.write(f"**欄位數量：** {len(df.columns)}")
    
    with col2:
        missing_data = df.isnull().sum().sum()
        st.write(f"**缺失值總數：** {missing_data:,}")
        st.write(f"**記憶體使用：** {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # 顯示前幾行數據
    st.dataframe(df.head(10), use_container_width=True)
    
    # 資料型態摘要
    with st.expander("📊 資料型態摘要"):
        dtype_summary = df.dtypes.value_counts()
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**欄位型態統計：**")
            for dtype, count in dtype_summary.items():
                st.write(f"• {dtype}: {count} 個欄位")
        
        with col2:
            st.write("**缺失值統計：**")
            missing_summary = df.isnull().sum()
            missing_cols = missing_summary[missing_summary > 0]
            if len(missing_cols) > 0:
                for col, missing_count in missing_cols.items():
                    percentage = (missing_count / len(df)) * 100
                    st.write(f"• {col}: {missing_count} ({percentage:.1f}%)")
            else:
                st.write("• 無缺失值")

    # 選擇分析欄位
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("⚠️ 找不到數值欄位，請確認檔案是否包含數值資料")
        st.stop()

    selected_col = st.selectbox("🎯 請選擇要分析的數值欄位", numeric_cols)

    # 建立分頁
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 基本統計", "📈 視覺化分析", "🔍 深度洞察", "📋 進階統計", "📝 完整報告"])
    
    with tab1:
        st.subheader("📊 基本統計資訊")
        col1, col2 = st.columns(2)
        
        with col1:
            stats_summary = df[selected_col].describe()
            st.write("**描述性統計：**")
            st.dataframe(stats_summary.to_frame().T, use_container_width=True)
        
        with col2:
            # 額外統計資訊
            data = df[selected_col].dropna()
            st.write("**額外資訊：**")
            st.metric("有效觀測數", f"{len(data):,}")
            st.metric("缺失值", f"{df[selected_col].isnull().sum():,}")
            st.metric("變異係數", f"{data.std()/abs(data.mean()):.4f}" if data.mean() != 0 else "N/A")
            st.metric("偏度", f"{stats.skew(data):.4f}")

    with tab2:
        st.subheader("📈 視覺化分析")
        
        # 創建互動式圖表
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('數據趨勢', '分布直方圖', '箱型圖', '統計摘要'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )
        
        # 趨勢圖
        fig.add_trace(
            go.Scatter(y=df[selected_col], mode='lines+markers', name='數值', line=dict(width=2)),
            row=1, col=1
        )
        
        # 直方圖
        fig.add_trace(
            go.Histogram(x=df[selected_col], nbinsx=30, name='分布', opacity=0.7),
            row=1, col=2
        )
        
        # 箱型圖
        fig.add_trace(
            go.Box(y=df[selected_col], name='箱型圖', boxpoints='outliers'),
            row=2, col=1
        )
        
        # 統計表格
        stats_data = df[selected_col].describe()
        fig.add_trace(
            go.Table(
                header=dict(values=['統計量', '數值']),
                cells=dict(values=[stats_data.index, [f"{val:.4f}" for val in stats_data.values]])
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text=f"📊 {selected_col} 完整分析")
        st.plotly_chart(fig, use_container_width=True)
        
        # 額外的 Seaborn 圖表
        st.subheader("📊 分布分析")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(data=df, x=selected_col, kde=True, ax=ax)
            ax.set_title(f'{selected_col} 分布圖 (含核密度估計)')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=df, y=selected_col, ax=ax)
            ax.set_title(f'{selected_col} 箱型圖')
            st.pyplot(fig)

    with tab3:
        st.subheader("🔍 智能數據洞察")
        
        # 生成洞察
        insights = generate_data_insights(df, selected_col)
        patterns = detect_patterns(df, selected_col)
        
        st.write("### 🤖 自動分析結果")
        for insight in insights:
            st.markdown(insight)
        
        if patterns:
            st.write("### 🔄 模式識別")
            for pattern in patterns:
                st.markdown(pattern)
        
        # 相關性分析（如果有其他數值欄位）
        if len(numeric_cols) > 1:
            st.write("### 🔗 相關性分析")
            other_cols = [col for col in numeric_cols if col != selected_col]
            
            if other_cols:
                correlation_data = []
                for col in other_cols:
                    corr = df[selected_col].corr(df[col])
                    if not np.isnan(corr):
                        correlation_data.append({'欄位': col, '相關係數': corr, '相關強度': 
                                               '強' if abs(corr) > 0.7 else '中' if abs(corr) > 0.3 else '弱'})
                
                if correlation_data:
                    corr_df = pd.DataFrame(correlation_data)
                    corr_df = corr_df.sort_values('相關係數', key=abs, ascending=False)
                    st.dataframe(corr_df, use_container_width=True)

    with tab4:
        st.subheader("📋 進階統計分析")
        
        advanced_stats = advanced_statistics(df, selected_col)
        
        for category, stats_dict in advanced_stats.items():
            st.write(f"### {category}")
            
            # 創建兩列顯示
            cols = st.columns(2)
            items = list(stats_dict.items())
            
            for i, (key, value) in enumerate(items):
                with cols[i % 2]:
                    st.metric(key, value)
        
        # 信賴區間計算
        st.write("### 📊 信賴區間")
        data = df[selected_col].dropna()
        confidence_levels = [0.90, 0.95, 0.99]
        
        ci_data = []
        for conf in confidence_levels:
            alpha = 1 - conf
            mean_val = data.mean()
            std_err = data.std() / np.sqrt(len(data))
            margin_error = stats.t.ppf(1 - alpha/2, len(data)-1) * std_err
            
            ci_data.append({
                '信賴水準': f"{conf*100:.0f}%",
                '下界': f"{mean_val - margin_error:.4f}",
                '上界': f"{mean_val + margin_error:.4f}",
                '誤差範圍': f"±{margin_error:.4f}"
            })
        
        ci_df = pd.DataFrame(ci_data)
        st.dataframe(ci_df, use_container_width=True)

    with tab5:
        st.subheader("📝 完整分析報告")
        
        # 生成完整報告
        report = f"""
# 📊 數據分析報告

## 基本資訊
- **分析欄位**: {selected_col}
- **資料筆數**: {len(df):,}
- **有效觀測**: {len(df[selected_col].dropna()):,}
- **缺失值**: {df[selected_col].isnull().sum():,}
- **分析時間**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 統計摘要
"""
        
        # 添加統計資訊
        stats_summary = df[selected_col].describe()
        for stat, value in stats_summary.items():
            report += f"- **{stat}**: {value:.4f}\n"
        
        report += "\n## 📈 主要發現\n"
        
        # 添加洞察
        insights = generate_data_insights(df, selected_col)
        for i, insight in enumerate(insights, 1):
            report += f"{i}. {insight}\n"
        
        # 添加模式分析
        patterns = detect_patterns(df, selected_col)
        if patterns:
            report += "\n## 🔄 發現的模式\n"
            for i, pattern in enumerate(patterns, 1):
                report += f"{i}. {pattern}\n"
        
        report += "\n## 💡 建議\n"
        report += "1. 定期監控數據品質，特別注意異常值\n"
        report += "2. 可考慮進行更深入的時間序列分析\n"
        report += "3. 建議與其他相關變數進行多元分析\n"
        report += "4. 根據業務需求設定適當的監控閾值\n"
        
        st.markdown(report)
        
        # 提供下載按鈕
        st.download_button(
            label="📥 下載分析報告",
            data=report,
            file_name=f"{selected_col}_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

else:
    # 示例數據展示
    st.info("👆 請上傳檔案開始分析，或查看以下功能說明")
    
    with st.expander("🎯 功能特色"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 統計分析**")
            st.write("• 完整描述性統計")
            st.write("• 分位數分析")
            st.write("• 分布特徵檢測")
            st.write("• 異常值識別")
            
            st.write("**🔍 智能洞察**")
            st.write("• 自動模式識別")
            st.write("• 趨勢分析")
            st.write("• 數據品質評估")
            st.write("• 實用建議生成")
        
        with col2:
            st.write("**📈 視覺化**")
            st.write("• 互動式圖表")
            st.write("• 多種圖表類型")
            st.write("• 分布分析圖")
            st.write("• 相關性矩陣")
            
            st.write("**📝 報告功能**")
            st.write("• 完整分析報告")
            st.write("• Markdown 格式匯出")
            st.write("• 專業統計術語")
            st.write("• 即時分析結果")

# 側邊欄
with st.sidebar:
    st.markdown("### 🚀 智能分析工具")
    st.markdown("**無需 API，本地分析**")
    
    st.markdown("### 📁 支援格式")
    st.markdown("• CSV 檔案 (多種編碼)")
    st.markdown("• Excel 檔案 (.xlsx)")
    st.markdown("• 自動偵測數值欄位")
    
    st.markdown("### ⭐ 核心功能")
    st.markdown("• 📊 完整統計分析")
    st.markdown("• 🤖 智能數據洞察") 
    st.markdown("• 📈 互動式視覺化")
    st.markdown("• 🔍 異常值檢測")
    st.markdown("• 📝 專業報告生成")
    
    st.markdown("### 💡 使用技巧")
    st.markdown("• 上傳前檢查數據格式")
    st.markdown("• 選擇有意義的數值欄位")
    st.markdown("• 查看各個分頁的不同分析")
    st.markdown("• 下載報告留存結果")
    
    st.markdown("---")
    st.markdown("**🔧 技術架構**")
    st.markdown("• Streamlit + Pandas")
    st.markdown("• Plotly + Seaborn")
    st.markdown("• SciPy 統計分析")
    st.markdown("• 100% 本地運算")
