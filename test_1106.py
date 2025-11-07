# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 09:59:01 2025

@author: ypan1
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import io

class RobustQHampel:
    """
    Q/Hampel方法的Streamlit实现
    """
    
    def __init__(self):
        self.s_star = None
        self.robust_mean = None
        self.lab_means = None
        self.original_data = None
    
    def parse_input_data(self, input_text):
        """
        解析用户输入的文本数据
        """
        try:
            labs = input_text.split(';')
            lab_data = []
            
            for i, lab in enumerate(labs):
                measurements = [float(x.strip()) for x in lab.split(',') if x.strip()]
                if len(measurements) < 1:
                    st.warning(f"实验室 {i+1} 没有有效数据，已跳过")
                    continue
                lab_data.append(measurements)
            
            if len(lab_data) < 2:
                st.error("至少需要2个实验室的数据")
                return None
            
            return lab_data
        except Exception as e:
            st.error(f"数据解析错误: {e}")
            return None
    
    def calculate_q_method(self, lab_data):
        """
        Q方法计算稳健标准差 - 使用修正后的公式
        """
        st.info("正在计算Q方法稳健标准差...")
        
        # 计算所有成对绝对差
        all_data = []
        for lab in lab_data:
            all_data.extend(lab)
        
        absolute_diffs = []
        n = len(all_data)
        for i in range(n):
            for j in range(i + 1, n):
                diff = abs(all_data[i] - all_data[j])
                if diff > 1e-10:  # 避免浮点数误差
                    absolute_diffs.append(diff)
        
        if not absolute_diffs:
            st.error("错误：没有有效的成对差值")
            return 0.0
        
        # 进度条
        progress_bar = st.progress(0)
        
        # 排序并计算经验CDF
        sorted_diffs = sorted(absolute_diffs)
        unique_points = []
        if sorted_diffs:
            current_val = sorted_diffs[0]
            unique_points.append(current_val)
            
            for val in sorted_diffs[1:]:
                if abs(val - current_val) > 1e-10:
                    current_val = val
                    unique_points.append(current_val)
        
        progress_bar.progress(30)
        
        # 计算H1(x)
        H1_values = []
        n_total = len(sorted_diffs)
        for x in unique_points:
            count = sum(1 for d in sorted_diffs if d <= x + 1e-10)
            H1_values.append(count / n_total)
        
        progress_bar.progress(60)
        
        # 计算G1(x)
        G1_values = [0.0]  # G1(0) = 0
        x_with_zero = [0.0] + unique_points
        
        for i in range(len(unique_points)):
            if i == 0:
                if unique_points[i] > 1e-10:
                    G1_val = 0.5 * H1_values[i]
                else:
                    G1_val = 0.0
            else:
                G1_val = 0.5 * (H1_values[i] + H1_values[i-1])
            G1_values.append(G1_val)
        
        progress_bar.progress(80)
        
        # 计算稳健标准差 s* - 使用修正后的公式
        H1_0 = 0.0  # 因为只考虑正差值
        
        # 计算参数
        a = 0.25 + 0.75 * H1_0
        b = 0.625 + 0.375 * H1_0
        
        # 线性插值求G1^{-1}(a)
        G1_inv_a = self._inverse_interpolation(G1_values, x_with_zero, a)
        
        # 标准正态分布的分位数
        phi_inv_b = stats.norm.ppf(b)
        
        # 使用修正后的公式计算s*
        s_star = G1_inv_a / (np.sqrt(2) * phi_inv_b)
        
        progress_bar.progress(100)
        
        # 显示中间计算结果
        with st.expander("查看Q方法计算详情"):
            st.write(f"成对绝对差数量: {len(absolute_diffs)}")
            st.write(f"计算参数: a = {a:.4f}, b = {b:.4f}")
            st.write(f"G1_inv({a:.4f}) = {G1_inv_a:.6f}")
            st.write(f"φ_inv({b:.4f}) = {phi_inv_b:.6f}")
        
        return s_star
    
    def _inverse_interpolation(self, G_values, x_points, target):
        """线性插值求逆函数值"""
        for i in range(len(G_values) - 1):
            if (G_values[i] <= target <= G_values[i + 1]) or (G_values[i + 1] <= target <= G_values[i]):
                x1 = x_points[i]
                x2 = x_points[i + 1]
                y1 = G_values[i]
                y2 = G_values[i + 1]
                
                if abs(y2 - y1) < 1e-10:
                    return x1
                
                return x1 + (target - y1) * (x2 - x1) / (y2 - y1)
        
        # 边界情况
        if target <= G_values[0]:
            return x_points[0]
        else:
            return x_points[-1]
    
    def calculate_hampel_method(self, lab_data, s_star):
        """
        Hampel方法计算稳健平均值
        """
        st.info("正在计算Hampel方法稳健平均值...")
        
        # 计算实验室均值
        lab_means = [np.mean(lab) for lab in lab_data]
        self.lab_means = lab_means
        p = len(lab_means)
        
        # 生成插值节点
        nodes = []
        for y in lab_means:
            offsets = [-4.5, -3.0, -1.5, 1.5, 3.0, 4.5]
            for offset in offsets:
                nodes.append(y + offset * s_star)
        
        sorted_nodes = sorted(nodes)
        median_val = np.median(lab_means)
        
        # 寻找方程的解
        solutions = []
        for m in range(len(sorted_nodes) - 1):
            d_m = sorted_nodes[m]
            d_m1 = sorted_nodes[m + 1]
            
            P_m = sum(self._psi_function((y - d_m) / s_star) for y in lab_means)
            P_m1 = sum(self._psi_function((y - d_m1) / s_star) for y in lab_means)
            
            if abs(P_m) < 1e-10:
                solutions.append(d_m)
            elif abs(P_m1) < 1e-10:
                solutions.append(d_m1)
            elif P_m * P_m1 < 0:
                # 线性插值
                x_star = d_m - P_m * (d_m1 - d_m) / (P_m1 - P_m)
                solutions.append(x_star)
        
        # 选择最接近中位数的解
        if not solutions:
            robust_mean = median_val
            st.warning("未找到解，使用中位数作为稳健平均值")
        else:
            distances = [abs(sol - median_val) for sol in solutions]
            min_dist = min(distances)
            closest_solutions = [sol for sol, dist in zip(solutions, distances) 
                               if abs(dist - min_dist) < 1e-10]
            
            if len(closest_solutions) == 1:
                robust_mean = closest_solutions[0]
            else:
                robust_mean = median_val
                st.warning("多个解同样接近中位数，使用中位数作为稳健平均值")
        
        return robust_mean
    
    def _psi_function(self, q):
        """Hampel ψ函数"""
        if -1.5 <= q <= 1.5:
            return q
        elif 1.5 < q <= 3.0:
            return 1.5
        elif 3.0 < q <= 4.5:
            return 1.5 * (4.5 - q) / 1.5
        elif q > 4.5:
            return 0.0
        elif -3.0 <= q < -1.5:
            return -1.5
        elif -4.5 <= q < -3.0:
            return -1.5 * (-4.5 - q) / 1.5
        else:
            return 0.0
    
    def calculate_traditional_stats(self, lab_data):
        """计算传统统计量用于对比"""
        all_data = []
        for lab in lab_data:
            all_data.extend(lab)
        
        traditional_mean = np.mean(all_data)
        traditional_std = np.std(all_data, ddof=1)  # 样本标准差
        
        lab_means = [np.mean(lab) for lab in lab_data]
        between_lab_std = np.std(lab_means, ddof=1) if len(lab_means) > 1 else 0
        
        return traditional_mean, traditional_std, between_lab_std
    
    def plot_comparison(self, lab_data):
        """绘制结果对比图"""
        if lab_data is None or self.lab_means is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 图1: 各实验室数据分布
        all_data = []
        for i, lab in enumerate(lab_data):
            all_data.extend(lab)
            ax1.scatter([i+1] * len(lab), lab, alpha=0.6, label=f'Lab {i+1}')
        
        traditional_mean = np.mean(all_data)
        ax1.axhline(y=traditional_mean, color='r', linestyle='--', 
                   label=f'传统均值: {traditional_mean:.3f}')
        
        if self.robust_mean is not None:
            ax1.axhline(y=self.robust_mean, color='g', linestyle='-', 
                       label=f'稳健均值: {self.robust_mean:.3f}')
        
        ax1.set_xlabel('实验室编号')
        ax1.set_ylabel('测量值')
        ax1.set_title('各实验室数据分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2: 实验室均值比较
        lab_indices = range(1, len(self.lab_means) + 1)
        ax2.bar(lab_indices, self.lab_means, alpha=0.7, color='skyblue')
        ax2.axhline(y=traditional_mean, color='r', linestyle='--', 
                   label=f'传统均值: {traditional_mean:.3f}')
        
        if self.robust_mean is not None:
            ax2.axhline(y=self.robust_mean, color='g', linestyle='-', 
                       label=f'稳健均值: {self.robust_mean:.3f}')
        
        ax2.set_xlabel('实验室编号')
        ax2.set_ylabel('实验室均值')
        ax2.set_title('实验室均值比较')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

def main():
    st.set_page_config(
        page_title="Q/Hampel稳健统计方法",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Q/Hampel稳健统计方法计算器")
    st.markdown("""
    本应用实现Q方法和Hampel方法，用于计算稳健的标准差和平均值，对异常值具有鲁棒性。
    
    **Q方法**：基于实验室结果数据集的成对绝对差，直接估计重复性标准差和再现性标准差。
    
    **Hampel方法**：采用迭代加权法估计稳健平均值，通过回归残差大小确定各样本权重。
    """)
    
    # 初始化计算器
    calculator = RobustQHampel()
    
    # 侧边栏 - 数据输入选项
    st.sidebar.header("数据输入选项")
    
    input_method = st.sidebar.radio(
        "选择数据输入方式:",
        ["手动输入", "使用演示数据", "上传CSV文件"]
    )
    
    lab_data = None
    
    if input_method == "手动输入":
        st.header("手动输入数据")
        st.markdown("""
        输入格式要求：
        - 每个实验室的数据用**逗号**分隔
        - 不同实验室用**分号**分隔
        - 示例：`10.1,10.2,10.3;10.5,10.6,10.4;9.8,9.9,9.7`
        """)
        
        input_text = st.text_area(
            "输入实验室数据:",
            value="10.1,10.2,10.3;10.5,10.6,10.4;9.8,9.9,9.7;10.7,10.8,10.6;9.5,9.6,9.4",
            height=100
        )
        
        if st.button("解析数据"):
            lab_data = calculator.parse_input_data(input_text)
            if lab_data:
                st.success(f"成功解析 {len(lab_data)} 个实验室的数据")
                
                # 显示数据表格
                data_display = []
                for i, lab in enumerate(lab_data):
                    for j, value in enumerate(lab):
                        data_display.append({
                            "实验室": f"Lab {i+1}",
                            "测量序号": j+1,
                            "测量值": value
                        })
                
                df = pd.DataFrame(data_display)
                st.dataframe(df, use_container_width=True)
                
                # 显示汇总统计
                st.subheader("数据汇总")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_measurements = sum(len(lab) for lab in lab_data)
                    st.metric("实验室数量", len(lab_data))
                
                with col2:
                    st.metric("总测量次数", total_measurements)
                
                with col3:
                    avg_measurements = total_measurements / len(lab_data)
                    st.metric("平均每实验室测量次数", f"{avg_measurements:.1f}")
    
    elif input_method == "使用演示数据":
        st.header("演示数据")
        demo_data = [
            [10.1, 10.2, 10.3, 10.15],
            [10.5, 10.6, 10.4, 10.55],
            [9.8, 9.9, 9.7, 9.85],
            [10.7, 10.8, 10.6, 10.65],
            [9.5, 9.6, 9.4, 9.45],
            [10.3, 10.2, 10.4, 10.25]
        ]
        
        lab_data = demo_data
        
        # 显示演示数据
        st.info("使用预定义的演示数据")
        data_display = []
        for i, lab in enumerate(demo_data):
            for j, value in enumerate(lab):
                data_display.append({
                    "实验室": f"Lab {i+1}",
                    "测量序号": j+1,
                    "测量值": value
                })
        
        df = pd.DataFrame(data_display)
        st.dataframe(df, use_container_width=True)
    
    elif input_method == "上传CSV文件":
        st.header("上传CSV文件")
        st.markdown("""
        上传CSV文件格式要求：
        - 每行代表一个实验室的数据
        - 每列代表一次重复测量
        - 文件应包含数值数据，表头可选
        """)
        
        uploaded_file = st.file_uploader("选择CSV文件", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("文件上传成功！")
                
                # 显示数据预览
                st.subheader("数据预览")
                st.dataframe(df, use_container_width=True)
                
                # 转换为lab_data格式
                lab_data = []
                for i, row in df.iterrows():
                    lab_measurements = [val for val in row if not pd.isna(val)]
                    if len(lab_measurements) > 0:
                        lab_data.append(lab_measurements)
                
                if len(lab_data) < 2:
                    st.error("至少需要2个实验室的数据")
                    lab_data = None
                else:
                    st.success(f"成功解析 {len(lab_data)} 个实验室的数据")
            
            except Exception as e:
                st.error(f"文件读取错误: {e}")
    
    # 计算按钮和结果显示
    if lab_data is not None:
        st.header("计算结果")
        
        if st.button("开始计算", type="primary"):
            # 创建计算进度区域
            with st.spinner("正在进行稳健统计计算..."):
                # 计算传统统计量
                trad_mean, trad_std, between_std = calculator.calculate_traditional_stats(lab_data)
                
                # 计算Q方法
                s_star = calculator.calculate_q_method(lab_data)
                
                # 计算Hampel方法
                robust_mean = calculator.calculate_hampel_method(lab_data, s_star)
            
            # 显示结果对比
            st.subheader("结果对比")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "传统算术平均值", 
                    f"{trad_mean:.6f}",
                    delta=f"{(robust_mean - trad_mean):.6f}"
                )
            
            with col2:
                st.metric(
                    "传统标准差", 
                    f"{trad_std:.6f}",
                    delta=f"{(s_star - trad_std):.6f}"
                )
            
            with col3:
                st.metric("Q方法稳健标准差", f"{s_star:.6f}")
            
            with col4:
                st.metric("Hampel稳健平均值", f"{robust_mean:.6f}")
            
            # 相对差异
            st.subheader("相对差异分析")
            col1, col2 = st.columns(2)
            
            with col1:
                if abs(trad_mean) > 1e-10:
                    mean_diff_pct = abs(robust_mean - trad_mean) / abs(trad_mean) * 100
                    st.metric("均值相对差异", f"{mean_diff_pct:.2f}%")
                else:
                    st.metric("均值相对差异", "N/A")
            
            with col2:
                if abs(trad_std) > 1e-10:
                    std_diff_pct = abs(s_star - trad_std) / trad_std * 100
                    st.metric("标准差相对差异", f"{std_diff_pct:.2f}%")
                else:
                    st.metric("标准差相对差异", "N/A")
            
            # 绘制图形
            st.subheader("可视化结果")
            fig = calculator.plot_comparison(lab_data)
            st.pyplot(fig)
            
            # 下载结果
            st.subheader("下载结果")
            
            # 创建结果数据框
            results_df = pd.DataFrame({
                "统计量": ["传统算术平均值", "传统标准差", "实验室间标准差", "Q方法稳健标准差", "Hampel稳健平均值"],
                "数值": [trad_mean, trad_std, between_std, s_star, robust_mean]
            })
            
            # 转换为CSV
            csv = results_df.to_csv(index=False)
            
            st.download_button(
                label="下载结果为CSV",
                data=csv,
                file_name="q_hampel_results.csv",
                mime="text/csv"
            )
    
    # 侧边栏 - 方法说明
    st.sidebar.header("方法说明")
    st.sidebar.markdown("""
    **Q方法特点**：
    - 基于成对绝对差，不使用均值或中位数
    - 对异常值具有鲁棒性
    - 直接估计重复性和再现性标准差
    
    **Hampel方法特点**：
    - 采用迭代加权法
    - 残差大的点权重低，残差小的点权重高
    - 通过ψ函数实现稳健估计
    """)
    
    st.sidebar.header("参考文献")
    st.sidebar.markdown("""
    [1] ISO 5725-5: Accuracy of measurement methods and results
    
    [9] Rousseeuw, P.J., & Leroy, A.M. (1987). Robust Regression and Outlier Detection
    """)

if __name__ == "__main__":
    main()