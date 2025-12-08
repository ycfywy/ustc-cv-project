import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 配置
current_dir = os.path.dirname(os.path.abspath(__file__))
mse_results_dir = os.path.join(current_dir, 'mse_comparison_results')
output_dir = os.path.join(current_dir, 'mse_visualization_results')
os.makedirs(output_dir, exist_ok=True)

# 参数配置
MIN_SAMPLES_THRESHOLD = 10  # 只显示样本数大于等于这个值的storm

def load_mse_data():
    """加载MSE对比数据"""
    csv_path = os.path.join(mse_results_dir, 'mse_comparison_by_storm.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run test_mse_comparison.py first.")
        return None
    
    df = pd.read_csv(csv_path)
    # 过滤掉Transformer MSE为NaN的行
    df = df.dropna(subset=['Transformer_Total_MSE'])
    return df

def plot_bar_comparison(df, min_samples=None):
    """柱状图对比LSTM和Transformer的MSE"""
    if min_samples is not None:
        df = df[df['Num_Samples'] >= min_samples].copy()
    
    if len(df) == 0:
        print("No storms meet the minimum samples threshold.")
        return
    
    # 按Total MSE排序
    df = df.sort_values('LSTM_Total_MSE', ascending=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. Total MSE对比
    x = np.arange(len(df))
    width = 0.35
    
    axes[0].bar(x - width/2, df['LSTM_Total_MSE'], width, label='LSTM', alpha=0.8, color='#3498db')
    axes[0].bar(x + width/2, df['Transformer_Total_MSE'], width, label='Transformer', alpha=0.8, color='#e74c3c')
    axes[0].set_xlabel('Storm Name', fontsize=12)
    axes[0].set_ylabel('Total MSE', fontsize=12)
    axes[0].set_title(f'Total MSE Comparison (Storms with ≥{min_samples if min_samples else 0} samples)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df['Storm Name'], rotation=45, ha='right')
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    
    # 2. 分别对比Latitude和Longitude MSE
    axes[1].bar(x - width/2, df['LSTM_Latitude_MSE'], width, label='LSTM Latitude', alpha=0.8, color='#3498db')
    axes[1].bar(x + width/2, df['Transformer_Latitude_MSE'], width, label='Transformer Latitude', alpha=0.8, color='#e74c3c')
    axes[1].set_xlabel('Storm Name', fontsize=12)
    axes[1].set_ylabel('Latitude MSE', fontsize=12)
    axes[1].set_title('Latitude MSE Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df['Storm Name'], rotation=45, ha='right')
    axes[1].legend(fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'mse_bar_comparison_min{min_samples if min_samples else 0}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved bar comparison to {output_file}")
    plt.close()

def plot_scatter_comparison(df, min_samples=None):
    """散点图对比LSTM和Transformer的MSE"""
    if min_samples is not None:
        df = df[df['Num_Samples'] >= min_samples].copy()
    
    if len(df) == 0:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Total MSE散点图
    axes[0].scatter(df['LSTM_Total_MSE'], df['Transformer_Total_MSE'], 
                   s=df['Num_Samples']*5, alpha=0.6, c=df['Num_Samples'], 
                   cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # 添加对角线（y=x）
    max_val = max(df['LSTM_Total_MSE'].max(), df['Transformer_Total_MSE'].max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x (Equal Performance)')
    
    axes[0].set_xlabel('LSTM Total MSE', fontsize=12)
    axes[0].set_ylabel('Transformer Total MSE', fontsize=12)
    axes[0].set_title('Total MSE Scatter Comparison\n(Bubble size = Number of samples)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 添加标注（只标注几个重要的点）
    for idx, row in df.nlargest(3, 'Num_Samples').iterrows():
        axes[0].annotate(row['Storm Name'], 
                       (row['LSTM_Total_MSE'], row['Transformer_Total_MSE']),
                       fontsize=8, alpha=0.7)
    
    # 2. Latitude vs Longitude MSE对比
    axes[1].scatter(df['LSTM_Latitude_MSE'], df['LSTM_Longitude_MSE'], 
                   s=100, alpha=0.6, label='LSTM', color='#3498db', edgecolors='black', linewidth=0.5)
    axes[1].scatter(df['Transformer_Latitude_MSE'], df['Transformer_Longitude_MSE'], 
                   s=100, alpha=0.6, label='Transformer', color='#e74c3c', edgecolors='black', linewidth=0.5, marker='^')
    
    axes[1].set_xlabel('Latitude MSE', fontsize=12)
    axes[1].set_ylabel('Longitude MSE', fontsize=12)
    axes[1].set_title('Latitude vs Longitude MSE', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'mse_scatter_comparison_min{min_samples if min_samples else 0}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved scatter comparison to {output_file}")
    plt.close()

def plot_improvement_analysis(df, min_samples=None):
    """分析Transformer相对于LSTM的改进"""
    if min_samples is not None:
        df = df[df['Num_Samples'] >= min_samples].copy()
    
    if len(df) == 0:
        return
    
    # 计算改进百分比
    df['Improvement_Percent'] = ((df['LSTM_Total_MSE'] - df['Transformer_Total_MSE']) / df['LSTM_Total_MSE'] * 100)
    df = df.sort_values('Improvement_Percent', ascending=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. 改进百分比柱状图
    colors = ['green' if x > 0 else 'red' for x in df['Improvement_Percent']]
    axes[0].barh(range(len(df)), df['Improvement_Percent'], color=colors, alpha=0.7)
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[0].set_yticks(range(len(df)))
    axes[0].set_yticklabels(df['Storm Name'])
    axes[0].set_xlabel('Improvement Percentage (%)', fontsize=12)
    axes[0].set_title('Transformer vs LSTM: Improvement Percentage\n(Green = Better, Red = Worse)', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (idx, row) in enumerate(df.iterrows()):
        axes[0].text(row['Improvement_Percent'], i, 
                    f'{row["Improvement_Percent"]:.1f}%', 
                    va='center', ha='left' if row['Improvement_Percent'] > 0 else 'right',
                    fontsize=9)
    
    # 2. 绝对改进值
    df['Absolute_Improvement'] = df['LSTM_Total_MSE'] - df['Transformer_Total_MSE']
    df = df.sort_values('Absolute_Improvement', ascending=True)
    colors2 = ['green' if x > 0 else 'red' for x in df['Absolute_Improvement']]
    axes[1].barh(range(len(df)), df['Absolute_Improvement'], color=colors2, alpha=0.7)
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_yticks(range(len(df)))
    axes[1].set_yticklabels(df['Storm Name'])
    axes[1].set_xlabel('Absolute Improvement (MSE reduction)', fontsize=12)
    axes[1].set_title('Transformer vs LSTM: Absolute Improvement', 
                     fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'mse_improvement_analysis_min{min_samples if min_samples else 0}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved improvement analysis to {output_file}")
    plt.close()

def plot_heatmap_comparison(df, min_samples=None):
    """热力图对比"""
    if min_samples is not None:
        df = df[df['Num_Samples'] >= min_samples].copy()
    
    if len(df) == 0:
        return
    
    # 准备数据
    comparison_data = pd.DataFrame({
        'LSTM Latitude': df['LSTM_Latitude_MSE'].values,
        'LSTM Longitude': df['LSTM_Longitude_MSE'].values,
        'Transformer Latitude': df['Transformer_Latitude_MSE'].values,
        'Transformer Longitude': df['Transformer_Longitude_MSE'].values
    }, index=df['Storm Name'].values)
    
    # 归一化以便更好地可视化
    comparison_data_norm = (comparison_data - comparison_data.min()) / (comparison_data.max() - comparison_data.min())
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.4)))
    im = ax.imshow(comparison_data_norm.T.values, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
    
    # 添加数值标注
    for i in range(len(comparison_data_norm.columns)):
        for j in range(len(comparison_data_norm)):
            text = ax.text(j, i, f'{comparison_data_norm.iloc[j, i]:.2f}',
                         ha="center", va="center", color="black", fontsize=8)
    
    # 设置标签
    ax.set_xticks(range(len(comparison_data_norm)))
    ax.set_xticklabels(comparison_data_norm.index, rotation=45, ha='right')
    ax.set_yticks(range(len(comparison_data_norm.columns)))
    ax.set_yticklabels(comparison_data_norm.columns)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized MSE (0=Best, 1=Worst)', fontsize=11)
    ax.set_title('MSE Heatmap Comparison (Normalized)\n(Darker = Better Performance)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Storm Name', fontsize=12)
    ax.set_ylabel('Model & Component', fontsize=12)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'mse_heatmap_comparison_min{min_samples if min_samples else 0}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap comparison to {output_file}")
    plt.close()

def plot_statistical_summary(df):
    """统计摘要可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 总体统计箱线图
    data_for_box = pd.DataFrame({
        'LSTM': [df['LSTM_Total_MSE'].mean(), df['LSTM_Latitude_MSE'].mean(), df['LSTM_Longitude_MSE'].mean()],
        'Transformer': [df['Transformer_Total_MSE'].mean(), df['Transformer_Latitude_MSE'].mean(), df['Transformer_Longitude_MSE'].mean()]
    }, index=['Total MSE', 'Latitude MSE', 'Longitude MSE'])
    
    data_for_box.T.plot(kind='bar', ax=axes[0, 0], width=0.8, alpha=0.8)
    axes[0, 0].set_title('Average MSE by Model and Component', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('MSE', fontsize=11)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. 标准差对比
    std_data = pd.DataFrame({
        'LSTM': [df['LSTM_Total_MSE'].std(), df['LSTM_Latitude_MSE'].std(), df['LSTM_Longitude_MSE'].std()],
        'Transformer': [df['Transformer_Total_MSE'].std(), df['Transformer_Latitude_MSE'].std(), df['Transformer_Longitude_MSE'].std()]
    }, index=['Total MSE', 'Latitude MSE', 'Longitude MSE'])
    
    std_data.T.plot(kind='bar', ax=axes[0, 1], width=0.8, alpha=0.8)
    axes[0, 1].set_title('MSE Standard Deviation by Model', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Standard Deviation', fontsize=11)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. 样本数分布
    axes[1, 0].hist(df['Num_Samples'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_title('Distribution of Number of Samples per Storm', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Number of Samples', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. MSE分布对比（箱线图）
    mse_data = pd.DataFrame({
        'LSTM Total': df['LSTM_Total_MSE'].values,
        'Transformer Total': df['Transformer_Total_MSE'].values,
        'LSTM Latitude': df['LSTM_Latitude_MSE'].values,
        'Transformer Latitude': df['Transformer_Latitude_MSE'].values
    })
    mse_data.boxplot(ax=axes[1, 1], rot=45)
    axes[1, 1].set_title('MSE Distribution Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('MSE', fontsize=11)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'mse_statistical_summary.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved statistical summary to {output_file}")
    plt.close()

def main():
    print("Loading MSE comparison data...")
    df = load_mse_data()
    
    if df is None or len(df) == 0:
        print("No data to visualize.")
        return
    
    print(f"Loaded {len(df)} storms with valid data")
    print(f"Storms with ≥{MIN_SAMPLES_THRESHOLD} samples: {len(df[df['Num_Samples'] >= MIN_SAMPLES_THRESHOLD])}")
    
    # 生成所有可视化
    print("\nGenerating visualizations...")
    
    # 1. 柱状图对比（所有数据）
    print("1. Creating bar comparison (all data)...")
    plot_bar_comparison(df.copy(), min_samples=None)
    
    # 2. 柱状图对比（过滤后）
    print(f"2. Creating bar comparison (min {MIN_SAMPLES_THRESHOLD} samples)...")
    plot_bar_comparison(df.copy(), min_samples=MIN_SAMPLES_THRESHOLD)
    
    # 3. 散点图对比
    print(f"3. Creating scatter comparison (min {MIN_SAMPLES_THRESHOLD} samples)...")
    plot_scatter_comparison(df.copy(), min_samples=MIN_SAMPLES_THRESHOLD)
    
    # 4. 改进分析
    print(f"4. Creating improvement analysis (min {MIN_SAMPLES_THRESHOLD} samples)...")
    plot_improvement_analysis(df.copy(), min_samples=MIN_SAMPLES_THRESHOLD)
    
    # 5. 热力图
    print(f"5. Creating heatmap comparison (min {MIN_SAMPLES_THRESHOLD} samples)...")
    plot_heatmap_comparison(df.copy(), min_samples=MIN_SAMPLES_THRESHOLD)
    
    # 6. 统计摘要
    print("6. Creating statistical summary...")
    plot_statistical_summary(df.copy())
    
    print(f"\nAll visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()

