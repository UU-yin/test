# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 09:59:01 2025

@author: ypan1
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import base64

# 设置页面配置
st.set_page_config(
    page_title="智能稳健标准差分析器",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题和描述
st.title("📊 智能稳健标准差分析器")
st.markdown("""
此应用使用Huber's M-estimator自动分析数据的稳健标准差特性，能够智能识别数据类型并选择最优参数。
支持**高度集中数据**、**中等集中数据**和**正常分布数据**的自动识别与处理。
""")

# 侧边栏配置
st.sidebar.header("配置参数")
st.sidebar.markdown("调整Huber M-estimator的参数设置：")

# Huber参数设置
c_value = st.sidebar.slider(
    "Huber参数 c", 
    min_value=1.0, 
    max_value=2.0, 
    value=1.345, 
    step=0.01,
    help="较小的c值对异常值更敏感，较大的c值更接近传统标准差"
)

# 数据输入方式选择
input_method = st.radio("选择数据输入方式:", 
                       ["上传CSV文件", "直接输入数据", "使用示例数据"])

class RobustStdAnalyzer:
    """稳健标准差分析器"""
    
    def __init__(self):
        self.data_type = None
        self.analysis_results = {}
    
    def analyze_data_characteristics(self, data):
        """分析数据特征"""
        n = len(data)
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        mad_val = np.median(np.abs(data - median_val))
        
        # 数据分布特征
        unique_vals, counts = np.unique(data, return_counts=True)
        max_count = np.max(counts)
        concentration_ratio = max_count / n
        
        # 异常值检测
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_ratio = len(outliers) / n
        
        # 判断数据类型
        if concentration_ratio > 0.8 and mad_val == 0:
            data_type = "高度集中数据"
        elif concentration_ratio > 0.6 and outlier_ratio < 0.1:
            data_type = "中等集中数据"
        else:
            data_type = "正常分布数据"
            
        characteristics = {
            'n': n, 'mean': mean_val, 'median': median_val, 'std': std_val,
            'mad': mad_val, 'concentration_ratio': concentration_ratio,
            'outlier_ratio': outlier_ratio, 'data_type': data_type,
            'unique_values_count': len(unique_vals), 'IQR': IQR
        }
        
        return characteristics
    
    def huber_robust_std(self, data, c=1.345, tol=1e-6, max_iter=100):
        """Huber M-estimator计算稳健标准差"""
        n = len(data)
        location = np.median(data)
        
        # 初始尺度估计
        mad = np.median(np.abs(data - location))
        if mad == 0:
            q90, q10 = np.percentile(data, [90, 10])
            scale = (q90 - q10) / (2 * 1.645)
            if scale == 0:
                q75, q25 = np.percentile(data, [75, 25])
                iqr = q75 - q25
                scale = iqr / 1.349 if iqr > 0 else np.std(data) * 0.1
        else:
            scale = 1.4826 * mad
        
        # 迭代计算
        for i in range(max_iter):
            residuals = data - location
            standardized = residuals / scale
            
            psi_values = np.where(np.abs(standardized) <= c, 
                                 standardized, 
                                 c * np.sign(standardized))
            
            new_location = location + scale * np.mean(psi_values)
            
            chi_values = np.where(np.abs(standardized) <= c, 
                                 standardized**2, 
                                 c**2)
            
            new_scale = scale * np.sqrt(np.mean(chi_values) / 0.5)
            
            if (abs(new_location - location) < tol and 
                abs(new_scale - scale) < tol):
                break
                
            location, scale = new_location, new_scale
        
        return location, scale
    
    def comprehensive_analysis(self, data, c):
        """综合分析"""
        # 传统统计量
        traditional_std = np.std(data)
        mad_std = 1.4826 * np.median(np.abs(data - np.median(data)))
        
        # Huber稳健估计
        huber_location, huber_std = self.huber_robust_std(data, c)
        
        # 数据特征
        characteristics = self.analyze_data_characteristics(data)
        
        results = {
            'traditional_std': traditional_std,
            'mad_std': mad_std if mad_std > 0 else 0,
            'huber_std': huber_std,
            'huber_location': huber_location,
            'data_type': characteristics['data_type'],
            'characteristics': characteristics
        }
        
        self.analysis_results.update(results)
        return results

def plot_data_distribution(data, huber_location, data_type):
    """绘制数据分布图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 直方图
    ax1.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(huber_location, color='red', linestyle='--', 
                label=f'Huber位置: {huber_location:.3f}')
    ax1.set_xlabel('数值')
    ax1.set_ylabel('频数')
    ax1.set_title(f'数据分布 - {data_type}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 箱线图
    ax2.boxplot(data, vert=True)
    ax2.set_ylabel('数值')
    ax2.set_title('数据箱线图')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def get_sample_data(choice):
    """获取示例数据"""
    if choice == "数据一（高度集中）":
        return np.array([
            -21, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20,
            -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -19, -19,
            -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19,
            -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19,
            -19, -19, -19, -19, -19, -19, -18
        ])
    else:  # 数据二（中等集中）
        return np.array([
            827.6, 827.6, 827.6, 827.7, 827.7, 827.7, 827.6, 827.6, 827.6, 827.6,
            827.6, 827.7, 827.6, 827.6, 827.7, 827.6, 827.6, 827.7, 827.7, 827.7,
            827.7, 827.7, 827.6, 827.6, 827.6, 827.6, 827.7, 827.7, 827.7, 827.7,
            827.6, 827.5, 827.5, 827.5, 827.8, 827.6, 827.8, 827.9, 827.4, 827.4,
            827.8, 827.4, 827.7, 827.5, 827.5, 827.6, 827.4, 828.1, 827.4, 827.5,
            827.6, 827.7, 827.6, 827.4, 827.6, 827.4, 827.2, 827.4, 826.1, 826.8,
            827.5, 827.4, 827.6, 827.1, 827.4, 827.7
        ])

# 主应用逻辑
def main():
    analyzer = RobustStdAnalyzer()
    data = None
    
    # 数据输入部分
    if input_method == "上传CSV文件":
        uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("数据预览:", df.head())
                # 假设数据在第一列
                if len(df.columns) > 0:
                    data = df.iloc[:, 0].values
                    st.success(f"成功加载 {len(data)} 个数据点")
            except Exception as e:
                st.error(f"文件读取错误: {e}")
    
    elif input_method == "直接输入数据":
        data_input = st.text_area("输入数据（用逗号分隔）:", 
                                value="-21, -20, -20, -20, -19, -19, -18")
        if st.button("解析数据"):
            try:
                data_list = [float(x.strip()) for x in data_input.split(',')]
                data = np.array(data_list)
                st.success(f"成功解析 {len(data)} 个数据点")
            except Exception as e:
                st.error(f"数据解析错误: {e}")
    
    else:  # 使用示例数据
        sample_choice = st.selectbox("选择示例数据集:", 
                                   ["数据一（高度集中）", "数据二（中等集中）"])
        data = get_sample_data(sample_choice)
        st.success(f"已加载示例数据: {sample_choice} ({len(data)} 个数据点)")
    
    # 数据分析部分
    if data is not None:
        st.markdown("---")
        
        # 执行分析
        with st.spinner("正在分析数据..."):
            results = analyzer.comprehensive_analysis(data, c_value)
            characteristics = results['characteristics']
        
        # 显示分析结果
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("数据类型", characteristics['data_type'])
            st.metric("数据量", characteristics['n'])
            st.metric("唯一值数量", characteristics['unique_values_count'])
        
        with col2:
            st.metric("均值", f"{characteristics['mean']:.4f}")
            st.metric("中位数", f"{characteristics['median']:.4f}")
            st.metric("集中度比例", f"{characteristics['concentration_ratio']:.3f}")
        
        with col3:
            st.metric("传统标准差", f"{results['traditional_std']:.4f}")
            st.metric("MAD标准差", f"{results['mad_std']:.4f}")
            st.metric("Huber稳健标准差", f"{results['huber_std']:.4f}")
        
        # 可视化
        st.markdown("### 📈 数据可视化")
        fig = plot_data_distribution(data, results['huber_location'], 
                                   characteristics['data_type'])
        st.pyplot(fig)
        
        # 详细统计信息
        st.markdown("### 📋 详细统计信息")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**数据分布统计:**")
            unique_vals, counts = np.unique(data, return_counts=True)
            dist_df = pd.DataFrame({
                '数值': unique_vals,
                '频数': counts,
                '比例 (%)': (counts / len(data) * 100).round(2)
            })
            st.dataframe(dist_df, use_container_width=True)
        
        with col2:
            st.write("**分位数统计:**")
            quantiles = {
                '最小值': np.min(data),
                'Q1 (25%)': np.percentile(data, 25),
                '中位数 (50%)': np.percentile(data, 50),
                'Q3 (75%)': np.percentile(data, 75),
                '最大值': np.max(data),
                'IQR': characteristics['IQR']
            }
            quantile_df = pd.DataFrame(list(quantiles.items()), 
                                     columns=['统计量', '值'])
            st.dataframe(quantile_df, use_container_width=True)
        
        # 方法比较
        st.markdown("### ⚖️ 方法比较")
        methods_data = {
            '方法': ['传统标准差', 'MAD标准差', 'Huber稳健标准差'],
            '标准差估计': [
                results['traditional_std'],
                results['mad_std'],
                results['huber_std']
            ],
            '适用场景': [
                '无异常值的数据',
                '有异常值但高度集中',
                '各种数据类型（推荐）'
            ]
        }
        methods_df = pd.DataFrame(methods_data)
        st.dataframe(methods_df, use_container_width=True)
        
        # 解释说明
        st.markdown("### 💡 分析说明")
        st.info(f"""
        **数据类型识别**: {characteristics['data_type']}
        
        - **传统标准差**: {results['traditional_std']:.4f} - 对异常值敏感
        - **MAD标准差**: {results['mad_std']:.4f} - 对异常值稳健，但可能为0
        - **Huber稳健标准差**: {results['huber_std']:.4f} - 平衡稳健性和效率
        
        **推荐**: 对于{characteristics['data_type']}，建议使用Huber稳健标准差作为变异性度量。
        """)
        
        # 下载结果
        st.markdown("### 📥 下载分析结果")
        results_df = pd.DataFrame({
            '统计量': [
                '数据量', '均值', '中位数', '传统标准差', 'MAD标准差', 
                'Huber稳健标准差', '数据类型', '集中度比例'
            ],
            '值': [
                characteristics['n'], characteristics['mean'], 
                characteristics['median'], results['traditional_std'],
                results['mad_std'], results['huber_std'],
                characteristics['data_type'], characteristics['concentration_ratio']
            ]
        })
        
        csv = results_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="robust_std_analysis.csv">下载CSV分析报告</a>'
        st.markdown(href, unsafe_allow_html=True)

# 运行应用
if __name__ == "__main__":
    main()  