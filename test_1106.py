# -*- coding: utf-8 -*-
"""
Q/Hampel 稳健统计分析方法 - 终极三合一版
新增：3种稳健标准差计算方法
作者：牛马姐妹
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

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
        robust_mean: 稳健平均值
        robust_std_clean: 清洁后数据标准差 (推荐)
        robust_std_mad: MAD稳健标准差 (可能为0)
        robust_std_iqr: IQR稳健标准差 (备选)
        global_mad: 全局MAD值
        global_median: 全局中位数
    """
    if window_size % 2 == 0:
        window_size += 1  # 确保为奇数
    
    half_window = window_size // 2
    n = len(data)
    cleaned_data = data.copy()
    outliers = []
    median_series = np.zeros(n)
    mad_series = np.zeros(n)
    
    # 全局统计量（稳健标准差计算基础）
    global_median = np.median(data)
    global_mad = np.median(np.abs(data - global_median))
    
    # IQR方法（四分位距）
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    # 滑动窗口处理
    for i in range(n):
        # 确定窗口范围
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window = data[start:end]
        
        # 计算中位数和局部MAD
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        
        median_series[i] = median
        mad_series[i] = mad
    
    # 第二步：使用全局MAD判断异常值（避免局部MAD=0的问题）
    residuals = data - median_series
    z_scores = 0.6745 * residuals / global_mad if global_mad > 0 else np.zeros_like(residuals)
    outliers = np.where(np.abs(z_scores) > k)[0]
    
    # 替换异常值
    if len(outliers) > 0:
        cleaned_data[outliers] = median_series[outliers]
    
    # 计算三种稳健标准差
    robust_mean = np.mean(cleaned_data)
    
    # 方法1：清洁后数据的标准差（最可靠）
    robust_std_clean = np.std(cleaned_data, ddof=1)
    
    # 方法2：MAD稳健标准差（可能为0）
    if global_mad > 0:
        robust_std_mad = 1.4826 * global_mad
        mad_warning = ""
    else:
        robust_std_mad = 0
        mad_warning = "⚠️ 全局MAD=0，数据重复性过高！"
    
    # 方法3：IQR稳健标准差（备选方案）
    robust_std_iqr = iqr / 1.349  # IQR与正态分布标准差的关系
    
    return cleaned_data, outliers, median_series, mad_series, robust_mean, robust_std_clean, robust_std_mad, robust_std_iqr, global_mad, global_median, mad_warning

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
    **🎯 新增：3种稳健标准差计算方法**
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
                '测量值': [-19, -19, -20, -20, -20, -20, -19, -19, -18, -21, -20, -19, -20, -20, -19, -20, -19, -19, -20, -20]
            })
            st.dataframe(sample_data.head(10))
            st.code("CSV格式示例: 测量值\n-19\n-20\n-18", language="text")
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
                (cleaned_data, outliers, median_series, mad_series, robust_mean, 
                 robust_std_clean, robust_std_mad, robust_std_iqr, 
                 global_mad, global_median, mad_warning) = hampel_filter(
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
                
                # 显示MAD警告（如果存在）
                if mad_warning:
                    st.warning(mad_warning)
                
                # 统计信息 - 第一行：基础统计
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("异常值数量", len(outliers))
                with col2:
                    st.metric("异常比例", f"{len(outliers)/len(df)*100:.2f}%")
                with col3:
                    st.metric("原始均值", f"{data.mean():.4f}")
                with col4:
                    st.metric("原始标准差", f"{data.std(ddof=1):.4f}")
                
                # 第二行：稳健统计量
                st.markdown("---")
                st.markdown("**🎯 稳健统计量**")
                
                col5, col6 = st.columns(2)
                with col5:
                    st.metric("稳健平均值", f"{robust_mean:.4f}", 
                             delta=f"{robust_mean-data.mean():.4f}", 
                             delta_color="inverse")
                    st.caption("清洁后数据的算术平均")
                
                with col6:
                    # 显示最可靠的稳健标准差
                    st.metric("稳健标准差(清洁数据)", f"{robust_std_clean:.4f}",
                             delta=f"{robust_std_clean-data.std(ddof=1):.4f}",
                             delta_color="inverse")
                    st.caption("推荐使用方法")
                
                # 第三行：两种备选稳健标准差
                st.markdown("---")
                st.markdown("**📏 备选稳健标准差**")
                
                col7, col8 = st.columns(2)
                with col7:
                    st.metric("MAD稳健标准差", f"{robust_std_mad:.4f}",
                             help=f"全局MAD方法: {global_mad:.4f} × 1.4826")
                    if global_mad == 0:
                        st.error("⚠️ 数据重复性过高，MAD=0！请参考其他方法")
                
                with col8:
                    st.metric("IQR稳健标准差", f"{robust_std_iqr:.4f}",
                             help="四分位距方法，适合任何数据")
                
                # 置信区间
                st.markdown("---")
                st.markdown("**📐 95%置信区间**")
                ci_95 = 1.96 * robust_std_clean / np.sqrt(len(cleaned_data))
                ci_lower = robust_mean - ci_95
                ci_upper = robust_mean + ci_95
                st.latex(f"CI_{{95\%}} = [{ci_lower:.4f}, {ci_upper:.4f}]")
                
                # 可视化
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
                
                # 添加统计摘要到CSV
                summary_stats = {
                    '统计量': ['原始平均值', '原始标准差', '稳健平均值', 
                             '稳健标准差(清洁数据)', 'MAD稳健标准差', 'IQR稳健标准差',
                             '全局MAD', '异常值数量', '异常比例(%)', 
                             '95%CI下限', '95%CI上限'],
                    '值': [data.mean(), data.std(ddof=1), robust_mean, 
                          robust_std_clean, robust_std_mad, robust_std_iqr,
                          global_mad, len(outliers), len(outliers)/len(df)*100,
                          ci_lower, ci_upper]
                }
                summary_df = pd.DataFrame(summary_stats)
                
                # 合并数据
                output_df = pd.concat([
                    summary_df,
                    pd.DataFrame([{}]),  # 空行分隔
                    pd.DataFrame(['详细数据']),
                    df
                ], ignore_index=True)
                
                csv = output_df.to_csv(index=False)
                st.download_button(
                    label="下载完整分析报告(CSV)",
                    data=csv,
                    file_name="hampel_robust_analysis_result.csv",
                    mime="text/csv",
                    help="包含三种稳健统计量和详细数据"
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
                    
                    **稳健统计量**：
                    - **稳健平均值**：清洁后数据的算术平均
                    - **稳健标准差(清洁数据)**：最可靠的方法
                    - **MAD稳健标准差**：基于全局MAD，数据重复时可能为0
                    - **IQR稳健标准差**：四分位距方法，适合任何数据
                    
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
                    outlier_df['标准化残差'] = 0.6745 * outlier_df['偏差'] / global_mad if global_mad > 0 else 0
                    st.dataframe(outlier_df)
                else:
                    st.success("✅ 未检测到异常值！")
        
    except Exception as e:
        st.error(f"❌ 数据处理错误: {str(e)}")
        st.info("请检查文件格式或联系技术支持")

# ==================== 部署配置 ====================
# requirements.txt:
# streamlit
# pandas
# numpy
# scipy
# openpyxl

if __name__ == "__main__":
    main()



