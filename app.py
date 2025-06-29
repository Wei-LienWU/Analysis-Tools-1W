# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import locale
import warnings
warnings.filterwarnings('ignore')

# 嘗試載入進階套件，如果沒有就使用基礎功能
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    st.warning("⚠️ 未安裝 scipy，部分進階統計功能將不可用")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    st.info("💡 未安裝 plotly，將使用 matplotlib 圖表")

# 設定編碼
if sys.stdout.encoding != 'utf-8':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 頁面設定
st.set_page_config(page_title="輕量級數據分析工具", layout="wide")
st.title("🔍 輕量級數據分析工具")
st.markdown("上傳數據檔案，獲得專業分析報告 - 使用基礎 Python 套件")

# 顯示套件狀態
with st.expander("🔧 套件狀態檢查"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**基礎套件**")
        st.success("✅ Streamlit")
        st.success("✅ Pandas") 
        st.success("✅ Matplotlib")
        st.success("✅ NumPy")
    
    with col2:
        st.write("**進階統計**")
        if HAS_SCIPY:
            st.success("✅ SciPy")
        else:
            st.error("❌ SciPy (可選)")
    
    with col3:
        st.write("**互動圖表**")
        if HAS_PLOTLY:
            st.success("✅ Plotly")
        else:
            st.error("❌ Plotly (可選)")

# 分析函數
def calculate_basic_stats(data):
    """計算基本統計量"""
    stats_dict = {
        '觀測數': len(data),
        '平均值': np.mean(data),
        '中位數': np.median(data),
        '標準差': np.std(data, ddof=1),
        '變異數': np.var(data, ddof=1),
        '最小值': np.min(data),
        '最大值': np.max(data),
        '範圍': np.max(data) - np.min(data),
        '第一四分位數': np.percentile(data, 25),
        '第三四分位數': np.percentile(data, 75),
        '四分位距': np.percentile(data, 75) - np.percentile(data, 25)
    }
    return stats_dict

def detect_outliers(data):
    """檢測異常值"""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers, lower_bound, upper_bound

def calculate_skewness_kurtosis(data):
    """計算偏度和峰度（不使用 scipy）"""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    # 偏度
    skewness = np.sum(((data - mean) / std) ** 3) / n
    
    # 峰度
    kurtosis = np.sum(((data - mean) / std) ** 4) / n - 3
    
    return skewness, kurtosis

def generate_insights(df, column):
    """生成數據洞察"""
    insights = []
    data = df[column].dropna()
    
    if len(data) == 0:
        return ["所選欄位無有效數據"]
    
    # 基本統計
    stats = calculate_basic_stats(data)
    mean_val = stats['平均值']
    median_val = stats['中位數']
    std_val = stats['標準差']
    
    # 1. 集中趨勢分析
    if abs(mean_val - median_val) / std_val < 0.5:
        insights.append("📊 **集中趨勢**: 數據分布相對對稱，平均值與中位數接近")
    elif mean_val > median_val:
        insights.append("📊 **集中趨勢**: 數據呈現右偏分布，存在較大的極值拉高平均值")
    else:
        insights.append("📊 **集中趨勢**: 數據呈現左偏分布，存在較小的極值拉低平均值")
    
    # 2. 變異性分析
    cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')
    if cv < 0.1:
        insights.append("📈 **變異性**: 數據變化很小，相當穩定")
    elif cv < 0.3:
        insights.append("📈 **變異性**: 數據變化適中，具有一定穩定性")
    else:
        insights.append("📈 **變異性**: 數據變化較大，波動性高")
    
    # 3. 異常值檢測
    outliers, lower_bound, upper_bound = detect_outliers(data)
    if len(outliers) > 0:
        outlier_ratio = len(outliers) / len(data) * 100
        insights.append(f"⚠️ **異常值**: 發現 {len(outliers)} 個異常值 ({outlier_ratio:.1f}%)，建議進一步檢查")
    else:
        insights.append("✅ **異常值**: 未發現明顯異常值，數據品質良好")
    
    # 4. 分布特徵
    try:
        skewness, kurtosis = calculate_skewness_kurtosis(data)
        if abs(skewness) < 0.5:
            skew_desc = "近似對稱"
        elif skewness > 0.5:
            skew_desc = "右偏（正偏）"
        else:
            skew_desc = "左偏（負偏）"
        insights.append(f"📋 **分布形狀**: {skew_desc}，峰度 = {kurtosis:.2f}")
    except:
        insights.append("📋 **分布形狀**: 無法計算偏度和峰度")
    
    # 5. 數據範圍分析
    data_range = stats['範圍']
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

def create_matplotlib_charts(df, column):
    """使用 matplotlib 創建圖表"""
    data = df[column].dropna()
    
    # 創建 2x2 子圖
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 趨勢圖
    ax1.plot(data.values, marker='o', markersize=2, linewidth=1)
    ax1.set_title(f'{column} 數據趨勢')
    ax1.set_xlabel('觀測序號')
    ax1.set_ylabel('數值')
    ax1.grid(True, alpha=0.3)
    
    # 2. 直方圖
    ax2.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title(f'{column} 分布直方圖')
    ax2.set_xlabel('數值')
    ax2.set_ylabel('頻率')
    ax2.grid(True, alpha=0.3)
    
    # 3. 箱型圖
    ax3.boxplot(data, vert=True)
    ax3.set_title(f'{column} 箱型圖')
    ax3.set_ylabel('數值')
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-Q 圖 (簡化版)
    sorted_data = np.sort(data)
    n = len(sorted_data)
    theoretical_quantiles = np.linspace(0, 1, n)
    ax4.scatter(theoretical_quantiles, sorted_data, alpha=0.6)
    ax4.set_title(f'{column} 分位數圖')
    ax4.set_xlabel('理論分位數')
    ax4.set_ylabel('實際數值')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_plotly_charts(df, column):
    """使用 Plotly 創建互動圖表"""
    if not HAS_PLOTLY:
        return None
    
    data = df[column].dropna()
    
    # 創建子圖
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('數據趨勢', '分布直方圖', '箱型圖', '累積分布'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 趨勢圖
    fig.add_trace(
        go.Scatter(y=data.values, mode='lines+markers', name='數值', 
                  line=dict(width=2), marker=dict(size=4)),
        row=1, col=1
    )
    
    # 直方圖
    fig.add_trace(
        go.Histogram(x=data.values, nbinsx=30, name='分布', opacity=0.7),
        row=1, col=2
    )
    
    # 箱型圖
    fig.add_trace(
        go.Box(y=data.values, name='箱型圖', boxpoints='outliers'),
        row=2, col=1
    )
    
    # 累積分布
    sorted_data = np.sort(data)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    fig.add_trace(
        go.Scatter(x=sorted_data, y=cumulative, mode='lines', name='累積分布'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text=f"📊 {column} 完整分析")
    return fig

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
        memory_usage = df.memory_usage(deep=True).sum() / 1024
        st.write(f"**記憶體使用：** {memory_usage:.1f} KB")
    
    # 顯示前幾行數據
    st.dataframe(df.head(10), use_container_width=True)

    # 選擇分析欄位
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("⚠️ 找不到數值欄位，請確認檔案是否包含數值資料")
        st.stop()

    selected_col = st.selectbox("🎯 請選擇要分析的數值欄位", numeric_cols)

    # 建立分頁
    tab1, tab2, tab3, tab4 = st.tabs(["📊 基本統計", "📈 圖表分析", "🔍 深度洞察", "📝 完整報告"])
    
    with tab1:
        st.subheader("📊 基本統計資訊")
        
        data = df[selected_col].dropna()
        basic_stats = calculate_basic_stats(data)
        
        # 顯示統計量
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("觀測數", f"{basic_stats['觀測數']:,}")
            st.metric("平均值", f"{basic_stats['平均值']:.4f}")
            st.metric("中位數", f"{basic_stats['中位數']:.4f}")
            st.metric("標準差", f"{basic_stats['標準差']:.4f}")
        
        with col2:
            st.metric("最小值", f"{basic_stats['最小值']:.4f}")
            st.metric("最大值", f"{basic_stats['最大值']:.4f}")
            st.metric("範圍", f"{basic_stats['範圍']:.4f}")
            cv = basic_stats['標準差'] / abs(basic_stats['平均值']) if basic_stats['平均值'] != 0 else 0
            st.metric("變異係數", f"{cv:.4f}")
        
        with col3:
            st.metric("Q1", f"{basic_stats['第一四分位數']:.4f}")
            st.metric("Q3", f"{basic_stats['第三四分位數']:.4f}")
            st.metric("IQR", f"{basic_stats['四分位距']:.4f}")
            st.metric("變異數", f"{basic_stats['變異數']:.4f}")
        
        # 分位數表格
        st.subheader("📊 分位數分析")
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        percentile_values = [np.percentile(data, p) for p in percentiles]
        
        percentile_df = pd.DataFrame({
            '百分位數': [f'{p}%' for p in percentiles],
            '數值': [f'{v:.4f}' for v in percentile_values]
        })
        
        st.dataframe(percentile_df, use_container_width=True)

    with tab2:
        st.subheader("📈 圖表分析")
        
        # 選擇圖表類型
        chart_type = st.radio("選擇圖表類型", ["Matplotlib 圖表", "Plotly 互動圖表"] if HAS_PLOTLY else ["Matplotlib 圖表"])
        
        if chart_type == "Matplotlib 圖表":
            fig = create_matplotlib_charts(df, selected_col)
            st.pyplot(fig)
        
        elif chart_type == "Plotly 互動圖表" and HAS_PLOTLY:
            fig = create_plotly_charts(df, selected_col)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # 異常值分析
        st.subheader("⚠️ 異常值分析")
        data = df[selected_col].dropna()
        outliers, lower_bound, upper_bound = detect_outliers(data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("異常值數量", len(outliers))
            st.metric("異常值比例", f"{len(outliers)/len(data)*100:.2f}%")
        
        with col2:
            st.metric("下界", f"{lower_bound:.4f}")
            st.metric("上界", f"{upper_bound:.4f}")
        
        if len(outliers) > 0:
            st.write("**異常值列表:**")
            outlier_df = pd.DataFrame({
                '異常值': outliers.values,
                '與平均值差異': outliers.values - data.mean()
            })
            st.dataframe(outlier_df.head(20), use_container_width=True)

    with tab3:
        st.subheader("🔍 智能數據洞察")
        
        # 生成洞察
        insights = generate_insights(df, selected_col)
        
        st.write("### 🤖 自動分析結果")
        for insight in insights:
            st.markdown(insight)
        
        # 分布特徵分析
        if HAS_SCIPY:
            st.write("### 📊 統計檢定")
            data = df[selected_col].dropna()
            
            # 常態性檢定
            try:
                shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(5000, len(data))))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Shapiro-Wilk 統計量", f"{shapiro_stat:.4f}")
                with col2:
                    st.metric("p-value", f"{shapiro_p:.4f}")
                
                if shapiro_p > 0.05:
                    st.success("✅ 數據可能符合常態分布")
                else:
                    st.warning("⚠️ 數據不符合常態分布")
            except Exception as e:
                st.info("無法進行常態性檢定")
        
        # 相關性分析
        if len(numeric_cols) > 1:
            st.write("### 🔗 相關性分析")
            other_cols = [col for col in numeric_cols if col != selected_col]
            
            if other_cols:
                correlation_data = []
                for col in other_cols:
                    corr = df[selected_col].corr(df[col])
                    if not np.isnan(corr):
                        strength = '強' if abs(corr) > 0.7 else '中' if abs(corr) > 0.3 else '弱'
                        correlation_data.append({
                            '欄位': col, 
                            '相關係數': f"{corr:.4f}",
                            '相關強度': strength
                        })
                
                if correlation_data:
                    corr_df = pd.DataFrame(correlation_data)
                    st.dataframe(corr_df, use_container_width=True)

    with tab4:
        st.subheader("📝 完整分析報告")
        
        # 生成報告
        data = df[selected_col].dropna()
        basic_stats = calculate_basic_stats(data)
        insights = generate_insights(df, selected_col)
        
        report = f"""# 📊 數據分析報告

## 基本資訊
- **分析欄位**: {selected_col}
- **資料筆數**: {len(df):,}
- **有效觀測**: {len(data):,}
- **缺失值**: {df[selected_col].isnull().sum():,}
- **分析時間**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 統計摘要
"""
        
        for key, value in basic_stats.items():
            if isinstance(value, (int, float)):
                report += f"- **{key}**: {value:.4f}\n"
            else:
                report += f"- **{key}**: {value}\n"
        
        report += "\n## 📈 主要發現\n"
        for i, insight in enumerate(insights, 1):
            report += f"{i}. {insight}\n"
        
        report += "\n## 💡 建議\n"
        report += "1. 定期監控數據品質，特別注意異常值\n"
        report += "2. 可考慮進行更深入的時間序列分析\n"
        report += "3. 建議與其他相關變數進行多元分析\n"
        report += "4. 根據業務需求設定適當的監控閾值\n"
        
        st.markdown(report)
        
        # 下載按鈕
        st.download_button(
            label="📥 下載分析報告",
            data=report,
            file_name=f"{selected_col}_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

else:
    # 安裝指南
    st.info("👆 請上傳檔案開始分析")
    
    with st.expander("📦 套件安裝指南"):
        st.markdown("""
        ### 必要套件 (已包含在基本版本)
        ```bash
        pip install streamlit pandas matplotlib numpy
        ```
        
        ### 可選套件 (增強功能)
        ```bash
        # 統計分析
        pip install scipy
        
        # 互動圖表
        pip install plotly
        
        # 美化圖表
        pip install seaborn
        ```
        
        ### 完整安裝
        ```bash
        pip install streamlit pandas matplotlib numpy scipy plotly seaborn
        ```
        """)
    
    with st.expander("🎯 功能說明"):
        st.markdown("""
        ### 基礎功能 (無需額外套件)
        - ✅ 完整描述性統計
        - ✅ 異常值檢測
        - ✅ 基本圖表 (Matplotlib)
        - ✅ 智能數據洞察
        - ✅ 分析報告生成
        
        ### 進階功能 (需要額外套件)
        - 📊 統計檢定 (需要 SciPy)
        - 📈 互動圖表 (需要 Plotly)
        - 🎨 美化圖表 (需要 Seaborn)
        """)

# 側邊欄
with st.sidebar:
    st.markdown("### 🚀 輕量級分析工具")
    st.markdown("**最少依賴，最大功能**")
    
    st.markdown("### 📋 套件狀態")
    st.success("✅ 基礎功能可用")
    if HAS_SCIPY:
        st.success("✅ 進階統計")
    else:
        st.warning("⚠️ 進階統計需要 SciPy")
    
    if HAS_PLOTLY:
        st.success("✅ 互動圖表")
    else:
        st.warning("⚠️ 互動圖表需要 Plotly")
    
    st.markdown("### 🎯 核心功能")
    st.markdown("• 📊 完整統計分析")
    st.markdown("• 🤖 智能數據洞察")
    st.markdown("• 📈 多種圖表類型")
    st.markdown("• ⚠️ 異常值檢測")
    st.markdown("• 📝 專業報告生成")
    
    st.markdown("### 💡 使用建議")
    st.markdown("• 先用基礎功能測試")
    st.markdown("• 需要時安裝額外套件")
    st.markdown("• 查看套件狀態指示")
    st.markdown("• 下載報告保存結果")
