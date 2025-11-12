# -*- coding: utf-8 -*-
"""
Q/Hampel 稳健统计分析方法
作者：牛马姐妹
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import base64

# ==================== 核心Q/Hampel算法 ====================
def hampel_filter(data, k=3.0, window_size=5):
    """Hampel滤波器实现"""
    if window_size % 2 == 0:
        window_size += 1
    
    half_window = window_size // 2
    n = len(data)
    cleaned_data = data.copy()
    outliers = []
    median_series = np.zeros(n)
    mad_series = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window = data[start:end]
        
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        
        median_series[i] = median
        mad_series[i] = mad
        
        if mad > 0:
            z_score = 0.6745 * (data[i] - median) / mad
            if np.abs(z_score) > k:
                cleaned_data[i] = median
                outliers.append(i)
    
    return cleaned_data, outliers, median_series, mad_series

# ==================== Streamlit UI ====================
def main():
    st.set_page_config(
        page_title="Q/Hampel统计分析工具",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Q/Hampel 稳健统计分析工具")
    st.markdown("""
    **符合Q/Hampel国际标准 (ISO 16269-4, Hampel Filter)**  
    基于中位数和MAD的稳健异常值检测
    """)
    
    # 侧边栏参数
    st.sidebar.header("⚙️ 参数设置")
    
    uploaded_file = st.sidebar.file_uploader(
        "📁 上传数据文件 (CSV/Excel)",
        type=['csv', 'xlsx']
    )
    
    if uploaded_file is None:
        st.info("👈 请先在侧边栏上传数据文件")
        with st.expander("📖 查看示例数据格式"):
            sample_data = pd.DataFrame({
                '时间': pd.date_range('2024-01-01', periods=20, freq='D'),
                '测量值': np.random.randn(20) * 5 + np.random.randn(20) * 20
            })
            st.dataframe(sample_data.head(10))
            st.code("CSV格式: 时间,测量值\n2024-01-01,15.6", language="text")
        return
    
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ 数据加载成功！共 {len(df)} 行，{len(df.columns)} 列")
        
        with st.expander("👀 查看原始数据"):
            st.dataframe(df.head())
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("❌ 未找到数值型列")
            return
        
        col_to_analyze = st.sidebar.selectbox("📈 选择分析列", numeric_cols)
        
        k_value = st.sidebar.slider("🔍 敏感度系数 (k值)", 1.0, 5.0, 3.0, 0.1)
        window_size = st.sidebar.slider("🪟 滑动窗口大小", 3, 21, 5, 2)
        
        if st.sidebar.button("🚀 开始分析", type="primary"):
            with st.spinner("⏳ 执行Hampel滤波..."):
                data = df[col_to_analyze].values
                cleaned_data, outliers, median_series, mad_series = hampel_filter(
                    data, k=k_value, window_size=window_size
                )
                
                df['清洁值'] = cleaned_data
                df['中位数'] = median_series
                df['MAD'] = mad_series
                df['是否异常'] = ['是' if i in outliers else '否' for i in range(len(df))]
                
                # ==================== 结果展示 ====================
                st.subheader("📊 分析结果")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("异常值数量", len(outliers))
                with col2:
                    st.metric("异常比例", f"{len(outliers)/len(df)*100:.2f}%")
                with col3:
                    st.metric("原始均值", f"{data.mean():.3f}")
                with col4:
                    st.metric("清洁后均值", f"{cleaned_data.mean():.3f}")
                
                # ==================== 可视化 ====================
                st.subheader("📈 数据可视化")
                
                # 准备图表数据
                chart_data = pd.DataFrame({
                    '索引': df.index,
                    '原始数据': data,
                    '滑动中位数': median_series,
                    '清洁后数据': cleaned_data
                })
                st.line_chart(chart_data.set_index('索引'))
                
                # 异常值散点图
                if outliers:
                    outlier_df = pd.DataFrame({
                        '索引': outliers,
                        '异常值': data[outliers]
                    })
                    st.scatter_chart(outlier_df.set_index('索引'), color='#ff0000')
                
                # ==================== 数据表格 ====================
                st.subheader("📋 详细数据")
                with st.expander("展开查看完整结果"):
                    st.dataframe(
                        df.style.apply(
                            lambda x: ['background-color: #ffcccc' if x['是否异常'] == '是' else '' 
                                      for _ in x], axis=1
                        )
                    )
                
                # ==================== 下载 ====================
                st.subheader("⬇️ 下载结果")
                csv = df.to_csv(index=False)
                st.download_button(
                    label="下载CSV文件",
                    data=csv,
                    file_name="hampel_analysis_result.csv",
                    mime="text/csv"
                )
                
                # ==================== 技术说明 ====================
                with st.expander("ℹ️ 技术说明"):
                    st.markdown("""
                    ### 📖 Q/Hampel方法原理
                    
                    **Hampel滤波器**是一种稳健统计方法：
                    
                    1. **滑动窗口**：对每个点取邻域数据
                    2. **计算统计量**：窗口内中位数(median)和MAD
                    3. **标准化**：z = 0.6745 * (x - median) / MAD
                    4. **判断异常**：|z| > k 时判定为异常值
                    5. **替换处理**：异常值替换为窗口中位数
                    
                    **优点**：
                    - ✅ 对异常值不敏感
                    - ✅ 无需假设数据分布
                    - ✅ 保留真实数据趋势
                    """)
                
                # ==================== 异常值详情 ====================
                if outliers:
                    st.subheader("🚨 异常值详情")
                    outlier_df = df.iloc[outliers][[
                        col_to_analyze, '清洁值', '中位数', 'MAD', '是否异常'
                    ]].copy()
                    outlier_df['偏差'] = outlier_df[col_to_analyze] - outlier_df['中位数']
                    st.dataframe(outlier_df)
                else:
                    st.success("✅ 未检测到异常值！")
        
    except Exception as e:
        st.error(f"❌ 错误: {str(e)}")
        st.info("请检查文件格式")

if __name__ == "__main__":
    main()
