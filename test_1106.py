# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 09:59:01 2025

@author: ypan1
"""

# -*- coding: utf-8 -*-
"""
Q/Hampel 稳健统计分析方法 - Streamlit 完整实现
符合Q/Hampel国际标准 (ISO 16269-4, Hampel Filter)
作者：牛马姐妹
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO, BytesIO
import base64

# ==================== 核心Q/Hampel算法 ====================
def hampel_filter(data, k=3.0, window_size=5):
    """
    Hampel滤波器实现 - 基于中位数和MAD的稳健异常值检测
    
    参数:
        data: 输入数据 (numpy array)
        k: 阈值倍数 (通常2.5-3.5)
        window_size: 滑动窗口大小 (必须为奇数)
    
    返回:
        cleaned_data: 替换异常值后的数据
        outliers: 异常值索引列表
        median_series: 中位数序列
        mad_series: MAD序列
    """
    if window_size % 2 == 0:
        window_size += 1  # 确保为奇数
    
    half_window = window_size // 2
    n = len(data)
    cleaned_data = data.copy()
    outliers = []
    median_series = np.zeros(n)
    mad_series = np.zeros(n)
    
    for i in range(n):
        # 确定窗口范围
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window = data[start:end]
        
        # 计算中位数和MAD
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        
        median_series[i] = median
        mad_series[i] = mad
        
        # 标准化残差
        if mad > 0:
            z_score = 0.6745 * (data[i] - median) / mad
            if np.abs(z_score) > k:
                # 标记为异常值并替换为中位数
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
    
    # 标题和说明
    st.title("📊 Q/Hampel 稳健统计分析工具")
    st.markdown("""
    **符合Q/Hampel国际标准 (ISO 16269-4, Hampel Filter)**  
    基于中位数和MAD的稳健异常值检测与数据清洗
    """)
    
    # 侧边栏参数设置
    st.sidebar.header("⚙️ 参数设置")
    
    # 文件上传
    uploaded_file = st.sidebar.file_uploader(
        "📁 上传数据文件 (CSV/Excel)",
        type=['csv', 'xlsx'],
        help="支持CSV和Excel格式，第一行应为列名"
    )
    
    if uploaded_file is None:
        st.info("👈 请先在侧边栏上传数据文件")
        # 示例数据展示
        with st.expander("📖 查看示例数据格式"):
            sample_data = pd.DataFrame({
                '时间': pd.date_range('2024-01-01', periods=100, freq='D'),
                '测量值': np.random.randn(100) * 10 + np.random.randn(100) * 50
            })
            st.dataframe(sample_data.head(10))
            st.code("CSV格式示例: 时间,测量值\n2024-01-01,15.6\n2024-01-02,18.3", language="text")
        return
    
    # 数据加载
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ 数据加载成功！共 {len(df)} 行，{len(df.columns)} 列")
        
        # 显示原始数据预览
        with st.expander("👀 查看原始数据"):
            st.dataframe(df.head())
        
        # 选择分析列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("❌ 未找到数值型列，请选择包含数字的数据文件")
            return
        
        col_to_analyze = st.sidebar.selectbox(
            "📈 选择分析列",
            numeric_cols,
            help="请选择需要应用Q/Hampel方法的数值列"
        )
        
        # Hampel参数
        k_value = st.sidebar.slider(
            "🔍 敏感度系数 (k值)",
            1.0, 5.0, 3.0, 0.1,
            help="值越小越敏感，通常2.5-3.5"
        )
        
        window_size = st.sidebar.slider(
            "🪟 滑动窗口大小",
            3, 21, 5, 2,
            help="必须为奇数，越大越平滑"
        )
        
        # 执行分析
        if st.sidebar.button("🚀 开始Q/Hampel分析", type="primary"):
            with st.spinner("⏳ 正在执行Hampel滤波..."):
                # 获取数据
                data = df[col_to_analyze].values
                
                # 执行Hampel滤波
                cleaned_data, outliers, median_series, mad_series = hampel_filter(
                    data, k=k_value, window_size=window_size
                )
                
                # 添加到DataFrame
                df['清洁值'] = cleaned_data
                df['中位数'] = median_series
                df['MAD'] = mad_series
                
                # 标记异常值
                df['是否异常'] = ['是' if i in outliers else '否' for i in range(len(df))]
                
                # ==================== 结果展示 ====================
                st.subheader("📊 分析结果")
                
                # 统计信息
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("异常值数量", len(outliers))
                with col2:
                    st.metric("异常值比例", f"{len(outliers)/len(df)*100:.2f}%")
                with col3:
                    st.metric("原始均值", f"{data.mean():.3f}")
                with col4:
                    st.metric("清洁后均值", f"{cleaned_data.mean():.3f}")
                
                # 可视化
                st.subheader("📈 数据可视化")
                
                # 创建图表
                fig = go.Figure()
                
                # 原始数据
                fig.add_trace(go.Scatter(
                    x=df.index, y=data,
                    mode='lines+markers',
                    name='原始数据',
                    line=dict(color='gray', width=1),
                    marker=dict(size=4)
                ))
                
                # 中位数序列
                fig.add_trace(go.Scatter(
                    x=df.index, y=median_series,
                    mode='lines',
                    name='滑动中位数',
                    line=dict(color='blue', width=2, dash='dash')
                ))
                
                # 异常值
                if outliers:
                    fig.add_trace(go.Scatter(
                        x=outliers, y=data[outliers],
                        mode='markers',
                        name='异常值',
                        marker=dict(
                            color='red',
                            size=10,
                            symbol='x',
                            line=dict(width=2)
                        )
                    ))
                
                # 清洁后数据
                fig.add_trace(go.Scatter(
                    x=df.index, y=cleaned_data,
                    mode='lines+markers',
                    name='清洁后数据',
                    line=dict(color='green', width=2),
                    marker=dict(size=4)
                ))
                
                fig.update_layout(
                    title="Q/Hampel处理前后对比",
                    xaxis_title="索引",
                    yaxis_title="数值",
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 数据表格
                st.subheader("📋 详细数据")
                with st.expander("展开查看完整结果表格"):
                    st.dataframe(
                        df.style.apply(
                            lambda x: ['background-color: #ffcccc' if x['是否异常'] == '是' else '' 
                                      for _ in x], axis=1
                        )
                    )
                
                # 下载结果
                st.subheader("⬇️ 下载结果")
                csv = df.to_csv(index=False)
                st.download_button(
                    label="下载CSV文件",
                    data=csv,
                    file_name="hampel_analysis_result.csv",
                    mime="text/csv",
                    help="下载包含清洁后数据的完整结果"
                )
                
                # 技术说明
                with st.expander("ℹ️ Q/Hampel方法技术说明"):
                    st.markdown("""
                    ### 📖 Q/Hampel方法原理
                    
                    **Hampel滤波器**是一种基于中位数和MAD（中位数绝对偏差）的稳健统计方法：
                    
                    1. **滑动窗口**：对每个数据点，取其邻域窗口内的数据
                    2. **计算统计量**：窗口内中位数(median)和MAD
                    3. **标准化**：计算标准化残差 z = 0.6745 * (x - median) / MAD
                    4. **判断异常**：|z| > k 时判定为异常值（k通常取3.0）
                    5. **替换处理**：异常值替换为窗口中位数
                    
                    **优点**：
                    - ✅ 对异常值不敏感
                    - ✅ 无需假设数据分布
                    - ✅ 保留真实数据趋势
                    
                    **参数说明**：
                    - **k值**：敏感度阈值，越小越敏感
                    - **窗口大小**：局部统计范围，必须为奇数
                    """)
                    
                # 异常值详情
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
        st.error(f"❌ 数据处理错误: {str(e)}")
        st.info("请检查文件格式或联系技术支持")

if __name__ == "__main__":
    main()