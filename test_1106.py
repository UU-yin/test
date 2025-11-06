# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 09:59:01 2025

@author: ypan1
"""

import numpy as np
from scipy import stats
import warnings

class RobustStdAnalyzer:
    """
    智能稳健标准差分析器
    自动识别数据类型并选择合适的处理方法
    """
    
    def __init__(self):
        self.data_type = None
        self.analysis_results = {}
        
    def analyze_data_characteristics(self, data):
        """
        分析数据特征，自动识别数据类型
        """
        # 基本统计量
        n = len(data)
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        mad_val = np.median(np.abs(data - median_val))
        
        # 数据分布特征
        unique_vals, counts = np.unique(data, return_counts=True)
        max_count = np.max(counts)
        concentration_ratio = max_count / n  # 最大值的集中比例
        
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
            data_type = "highly_concentrated"  # 高度集中数据
        elif concentration_ratio > 0.6 and outlier_ratio < 0.1:
            data_type = "moderately_concentrated"  # 中等集中数据
        else:
            data_type = "normal_distributed"  # 正常分布数据
            
        characteristics = {
            'n': n,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'mad': mad_val,
            'concentration_ratio': concentration_ratio,
            'outlier_ratio': outlier_ratio,
            'data_type': data_type,
            'unique_values_count': len(unique_vals)
        }
        
        return characteristics
    
    def select_optimal_c_parameter(self, data_characteristics):
        """
        根据数据特征选择最优的c参数
        """
        data_type = data_characteristics['data_type']
        concentration_ratio = data_characteristics['concentration_ratio']
        outlier_ratio = data_characteristics['outlier_ratio']
        
        if data_type == "highly_concentrated":
            # 高度集中数据，使用较小的c值
            if concentration_ratio > 0.9:
                optimal_c = 1.35
            else:
                optimal_c = 1.40
                
        elif data_type == "moderately_concentrated":
            # 中等集中数据，平衡处理
            if outlier_ratio < 0.05:
                optimal_c = 1.45
            else:
                optimal_c = 1.42
                
        else:  # normal_distributed
            # 正常分布数据，使用标准c值
            optimal_c = 1.345
            
        return optimal_c
    
    def huber_robust_std(self, data, c=None, tol=1e-6, max_iter=100):
        """
        Huber M-estimator 计算稳健标准差
        """
        if c is None:
            # 自动选择c参数
            characteristics = self.analyze_data_characteristics(data)
            c = self.select_optimal_c_parameter(characteristics)
            self.data_type = characteristics['data_type']
            self.analysis_results['characteristics'] = characteristics
            self.analysis_results['auto_selected_c'] = c
        
        n = len(data)
        location = np.median(data)
        
        # 初始尺度估计
        mad = np.median(np.abs(data - location))
        if mad == 0:
            # 使用基于分位数的估计
            q90, q10 = np.percentile(data, [90, 10])
            scale = (q90 - q10) / (2 * 1.645)
            if scale == 0:
                # 如果仍然为0，使用基于IQR的估计
                q75, q25 = np.percentile(data, [75, 25])
                iqr = q75 - q25
                scale = iqr / 1.349 if iqr > 0 else np.std(data) * 0.1
        else:
            scale = 1.4826 * mad
        
        # 迭代计算
        for i in range(max_iter):
            residuals = data - location
            standardized = residuals / scale
            
            # Huber psi函数
            psi_values = np.where(np.abs(standardized) <= c, 
                                 standardized, 
                                 c * np.sign(standardized))
            
            # 更新位置参数
            new_location = location + scale * np.mean(psi_values)
            
            # Huber chi函数（用于尺度估计）
            chi_values = np.where(np.abs(standardized) <= c, 
                                 standardized**2, 
                                 c**2)
            
            # 更新尺度参数
            new_scale = scale * np.sqrt(np.mean(chi_values) / 0.5)
            
            # 检查收敛
            if (abs(new_location - location) < tol and 
                abs(new_scale - scale) < tol):
                break
                
            location, scale = new_location, new_scale
        
        return location, scale, c
    
    def comprehensive_analysis(self, data):
        """
        综合分析数据，提供多种稳健标准差估计
        """
        # 传统统计量
        traditional_std = np.std(data)
        mad_std = 1.4826 * np.median(np.abs(data - np.median(data)))
        
        # Huber稳健估计（自动选择c）
        huber_location, huber_std, auto_c = self.huber_robust_std(data)
        
        # 其他稳健方法
        qn_std = self.qn_estimator(data)  # Qn估计器
        sn_std = self.sn_estimator(data)  # Sn估计器
        
        results = {
            'traditional_std': traditional_std,
            'mad_std': mad_std,
            'huber_std': huber_std,
            'huber_location': huber_location,
            'auto_selected_c': auto_c,
            'qn_std': qn_std,
            'sn_std': sn_std,
            'data_type': self.data_type,
            'recommended_method': self.recommend_method(huber_std, mad_std, traditional_std)
        }
        
        self.analysis_results.update(results)
        return results
    
    def qn_estimator(self, data):
        """
        Qn稳健尺度估计器
        """
        n = len(data)
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append(abs(data[i] - data[j]))
        
        if len(pairs) > 0:
            qn = 2.2219 * np.percentile(pairs, 25)  # 第一四分位数
        else:
            qn = 0
            
        return qn
    
    def sn_estimator(self, data):
        """
        Sn稳健尺度估计器
        """
        n = len(data)
        medians = []
        for i in range(n):
            differences = [abs(data[i] - data[j]) for j in range(n) if j != i]
            medians.append(np.median(differences))
        
        sn = 1.1926 * np.median(medians)
        return sn
    
    def recommend_method(self, huber_std, mad_std, traditional_std):
        """
        推荐最适合的稳健标准差估计方法
        """
        if self.data_type == "highly_concentrated" and mad_std == 0:
            return "huber"
        elif abs(huber_std - mad_std) / mad_std < 0.1:
            return "mad"  # MAD足够好
        elif abs(huber_std - traditional_std) / traditional_std < 0.2:
            return "traditional"  # 传统方法可接受
        else:
            return "huber"  # Huber方法最优
    
    def print_analysis_report(self, data):
        """
        生成分析报告
        """
        results = self.comprehensive_analysis(data)
        characteristics = self.analysis_results['characteristics']
        
        print("=" * 60)
        print("智能稳健标准差分析报告")
        print("=" * 60)
        print(f"数据类型: {characteristics['data_type']}")
        print(f"数据量: {characteristics['n']}")
        print(f"唯一值数量: {characteristics['unique_values_count']}")
        print(f"集中度比例: {characteristics['concentration_ratio']:.3f}")
        print(f"异常值比例: {characteristics['outlier_ratio']:.3f}")
        print("-" * 60)
        print("各种方法的标准差估计:")
        print(f"  传统标准差: {results['traditional_std']:.4f}")
        print(f"  MAD标准差: {results['mad_std']:.4f}")
        print(f"  Huber稳健标准差: {results['huber_std']:.4f}")
        print(f"  Qn估计器: {results['qn_std']:.4f}")
        print(f"  Sn估计器: {results['sn_std']:.4f}")
        print("-" * 60)
        print(f"自动选择的Huber参数 c: {results['auto_selected_c']:.3f}")
        print(f"推荐使用方法: {results['recommended_method']}")
        print("=" * 60)

# 使用示例
def main():
    analyzer = RobustStdAnalyzer()
    
    # 数据一：高度集中数据
    data1 = np.array([
        -21, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20,
        -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -19, -19,
        -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19,
        -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19,
        -19, -19, -19, -19, -19, -19, -18
    ])
    
    print("数据一分析:")
    analyzer.print_analysis_report(data1)
    
    # 数据二：中等集中数据
    data2 = np.array([
        827.6, 827.6, 827.6, 827.7, 827.7, 827.7, 827.6, 827.6, 827.6, 827.6,
        827.6, 827.7, 827.6, 827.6, 827.7, 827.6, 827.6, 827.7, 827.7, 827.7,
        827.7, 827.7, 827.6, 827.6, 827.6, 827.6, 827.7, 827.7, 827.7, 827.7,
        827.6, 827.5, 827.5, 827.5, 827.8, 827.6, 827.8, 827.9, 827.4, 827.4,
        827.8, 827.4, 827.7, 827.5, 827.5, 827.6, 827.4, 828.1, 827.4, 827.5,
        827.6, 827.7, 827.6, 827.4, 827.6, 827.4, 827.2, 827.4, 826.1, 826.8,
        827.5, 827.4, 827.6, 827.1, 827.4, 827.7
    ])
    
    print("\n数据二分析:")
    analyzer2 = RobustStdAnalyzer()  # 新的分析器实例
    analyzer2.print_analysis_report(data2)

if __name__ == "__main__":
    main()
    
# 批量处理多个数据集
def batch_analyze(data_list, names=None):
    results = {}
    for i, data in enumerate(data_list):
        analyzer = RobustStdAnalyzer()
        name = names[i] if names else f"Dataset_{i+1}"
        results[name] = analyzer.comprehensive_analysis(data)
    return results

# 可视化分析结果
def plot_comparison(analyzer):
    import matplotlib.pyplot as plt
    
    results = analyzer.analysis_results
    methods = ['Traditional', 'MAD', 'Huber', 'Qn', 'Sn']
    values = [results['traditional_std'], results['mad_std'], 
              results['huber_std'], results['qn_std'], results['sn_std']]
    
    plt.figure(figsize=(10, 6))
    plt.bar(methods, values)
    plt.title('Robust Standard Deviation Comparison')
    plt.ylabel('Standard Deviation')
    plt.grid(axis='y', alpha=0.3)
    plt.show()    