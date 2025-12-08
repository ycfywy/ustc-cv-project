# MSE可视化图表说明文档

本文档详细说明了 `visualize_mse_comparison.py` 生成的所有可视化图表的含义和解读方法。

## 目录

1. [柱状图对比 (Bar Comparison)](#1-柱状图对比-bar-comparison)
2. [散点图对比 (Scatter Comparison)](#2-散点图对比-scatter-comparison)
3. [改进分析图 (Improvement Analysis)](#3-改进分析图-improvement-analysis)
4. [热力图对比 (Heatmap Comparison)](#4-热力图对比-heatmap-comparison)
5. [统计摘要图 (Statistical Summary)](#5-统计摘要图-statistical-summary)

---

## 1. 柱状图对比 (Bar Comparison)

### 文件名称
- `mse_bar_comparison_min0.png` - 包含所有台风的数据
- `mse_bar_comparison_min10.png` - 只包含样本数≥10的台风

### 图表结构
包含两个子图，上下排列：

#### 上图：Total MSE 对比
- **X轴**：台风名称（按LSTM的Total MSE从小到大排序）
- **Y轴**：Total MSE值
- **蓝色柱子**：LSTM模型的Total MSE
- **红色柱子**：Transformer模型的Total MSE
- **含义**：直接对比两个模型在每个台风上的总体预测误差

#### 下图：Latitude MSE 对比
- **X轴**：台风名称（与上图相同顺序）
- **Y轴**：Latitude MSE值
- **蓝色柱子**：LSTM模型的纬度预测误差
- **红色柱子**：Transformer模型的纬度预测误差
- **含义**：对比两个模型在纬度预测上的表现

### 如何解读
1. **柱子高度**：越低越好，表示预测误差越小
2. **颜色对比**：
   - 如果红色柱子低于蓝色柱子：Transformer在该台风上表现更好
   - 如果蓝色柱子低于红色柱子：LSTM在该台风上表现更好
3. **排序**：台风按LSTM的Total MSE排序，可以快速看出哪些台风预测难度较大

### 使用场景
- 快速浏览所有台风的模型表现
- 识别哪些台风对模型来说更具挑战性
- 对比两个模型在不同台风上的相对表现

---

## 2. 散点图对比 (Scatter Comparison)

### 文件名称
- `mse_scatter_comparison_min10.png` - 只包含样本数≥10的台风

### 图表结构
包含两个子图，左右排列：

#### 左图：Total MSE 散点对比
- **X轴**：LSTM Total MSE
- **Y轴**：Transformer Total MSE
- **每个点**：代表一个台风
- **气泡大小**：该台风的样本数量（样本越多，气泡越大）
- **颜色**：样本数量（使用viridis颜色映射，越亮表示样本越多）
- **红色虚线**：y=x对角线，表示两个模型性能相等的参考线

**如何解读：**
- **点在红色虚线下方**：Transformer的MSE更小，Transformer表现更好
- **点在红色虚线上方**：LSTM的MSE更小，LSTM表现更好
- **点在红色虚线上**：两个模型性能相同
- **距离对角线的距离**：表示性能差异的大小
- **气泡大小**：样本数越多，结果越可靠

#### 右图：Latitude vs Longitude MSE 对比
- **X轴**：Latitude MSE（纬度预测误差）
- **Y轴**：Longitude MSE（经度预测误差）
- **蓝色圆点**：LSTM模型的结果
- **红色三角**：Transformer模型的结果

**如何解读：**
- **位置越靠近原点(0,0)**：表示预测误差越小，模型越好
- **模型对比**：
  - 如果蓝色点在红色三角的左下方：LSTM在纬度和经度上都更好
  - 如果红色三角在蓝色点的左下方：Transformer在纬度和经度上都更好
  - 如果交叉分布：两个模型在不同维度各有优势
- **误差分布特征**：
  - 点主要分布在X轴附近：经度预测误差小，纬度误差大
  - 点主要分布在Y轴附近：纬度预测误差小，经度误差大

### 使用场景
- 快速识别哪个模型整体性能更好
- 了解模型在不同维度（纬度/经度）上的表现差异
- 识别样本数量对结果可靠性的影响

---

## 3. 改进分析图 (Improvement Analysis)

### 文件名称
- `mse_improvement_analysis_min10.png` - 只包含样本数≥10的台风

### 图表结构
包含两个子图，上下排列：

#### 上图：改进百分比 (Improvement Percentage)
- **Y轴**：台风名称（按改进百分比从小到大排序）
- **X轴**：改进百分比 (%)
- **绿色柱子**：Transformer比LSTM改进（正值）
- **红色柱子**：Transformer比LSTM差（负值）
- **黑色虚线**：x=0参考线
- **数值标签**：显示具体的改进百分比

**计算公式：**
```
改进百分比 = (LSTM_Total_MSE - Transformer_Total_MSE) / LSTM_Total_MSE × 100%
```

**如何解读：**
- **正值（绿色）**：Transformer的MSE更小，Transformer表现更好，数值越大改进越多
- **负值（红色）**：LSTM的MSE更小，LSTM表现更好，数值越小差距越大
- **接近0**：两个模型性能相近

#### 下图：绝对改进值 (Absolute Improvement)
- **Y轴**：台风名称（按绝对改进值从小到大排序）
- **X轴**：绝对改进值（MSE的绝对减少量）
- **绿色柱子**：Transformer的MSE减少量（正值）
- **红色柱子**：LSTM的MSE减少量（负值，表示Transformer的MSE增加）
- **黑色虚线**：x=0参考线

**计算公式：**
```
绝对改进 = LSTM_Total_MSE - Transformer_Total_MSE
```

**如何解读：**
- **正值（绿色）**：Transformer的MSE比LSTM小，改进值为MSE的绝对减少量
- **负值（红色）**：Transformer的MSE比LSTM大，差值为MSE的绝对增加量
- **数值大小**：表示改进或退步的绝对幅度

### 使用场景
- 量化Transformer相对于LSTM的改进程度
- 识别哪些台风上Transformer有明显优势或劣势
- 评估模型改进的统计显著性

---

## 4. 热力图对比 (Heatmap Comparison)

### 文件名称
- `mse_heatmap_comparison_min10.png` - 只包含样本数≥10的台风

### 图表结构
- **行（Y轴）**：模型和组件
  - LSTM Latitude
  - LSTM Longitude
  - Transformer Latitude
  - Transformer Longitude
- **列（X轴）**：台风名称
- **颜色**：归一化后的MSE值（使用RdYlGn_r颜色映射）
  - **深绿色**：MSE最小（性能最好）
  - **黄色**：MSE中等
  - **深红色**：MSE最大（性能最差）
- **数值标注**：每个格子中显示归一化后的MSE值（0-1之间）

**归一化公式：**
```
归一化MSE = (原始MSE - 最小值) / (最大值 - 最小值)
```

### 如何解读
1. **颜色深浅**：
   - 颜色越深（越绿）：该模型在该台风上的MSE越小，性能越好
   - 颜色越浅（越红）：该模型在该台风上的MSE越大，性能越差
2. **横向对比**（同一台风）：
   - 比较四个模型-组件组合在同一台风上的表现
   - 可以快速看出哪个模型在哪个维度上表现更好
3. **纵向对比**（同一模型-组件）：
   - 比较同一模型-组件在不同台风上的表现
   - 可以识别哪些台风对该模型来说更具挑战性
4. **模式识别**：
   - 如果某一行整体颜色较深：该模型-组件在所有台风上都表现较好
   - 如果某一列整体颜色较浅：该台风对所有模型来说都较难预测

### 使用场景
- 快速浏览所有模型-台风组合的表现
- 识别模型在不同台风上的表现模式
- 发现异常值或特殊情况

---

## 5. 统计摘要图 (Statistical Summary)

### 文件名称
- `mse_statistical_summary.png` - 包含所有有效数据

### 图表结构
包含四个子图，2×2排列：

#### 左上：平均MSE对比 (Average MSE by Model and Component)
- **X轴**：MSE类型（Total MSE, Latitude MSE, Longitude MSE）
- **Y轴**：平均MSE值
- **蓝色柱子**：LSTM模型的平均MSE
- **红色柱子**：Transformer模型的平均MSE
- **含义**：展示两个模型在三个指标上的平均表现

**如何解读：**
- 柱子越低越好
- 对比两个模型在总体、纬度、经度上的平均误差
- 可以看出哪个模型在哪个维度上平均表现更好

#### 右上：MSE标准差对比 (MSE Standard Deviation by Model)
- **X轴**：MSE类型（Total MSE, Latitude MSE, Longitude MSE）
- **Y轴**：标准差
- **蓝色柱子**：LSTM模型MSE的标准差
- **红色柱子**：Transformer模型MSE的标准差
- **含义**：展示两个模型预测误差的稳定性

**如何解读：**
- 标准差越小，表示模型表现越稳定
- 如果某个模型的标准差较大，说明它在不同台风上的表现差异较大
- 标准差小但平均值也小：模型既准确又稳定

#### 左下：样本数分布 (Distribution of Number of Samples per Storm)
- **X轴**：每个台风的样本数量
- **Y轴**：频数（有多少个台风具有该样本数）
- **直方图**：展示样本数的分布情况

**如何解读：**
- 了解数据集中台风的样本数分布
- 识别样本数特别多或特别少的台风
- 评估数据集的平衡性

#### 右下：MSE分布箱线图 (MSE Distribution Comparison)
- **X轴**：模型和MSE类型
  - LSTM Total
  - Transformer Total
  - LSTM Latitude
  - Transformer Latitude
- **Y轴**：MSE值
- **箱线图**：展示MSE值的分布情况

**箱线图元素说明：**
- **箱体**：包含50%的数据（Q1到Q3）
- **中线**：中位数
- **上须**：最大值（或Q3+1.5×IQR）
- **下须**：最小值（或Q1-1.5×IQR）
- **异常值**：超出须线的点

**如何解读：**
- **箱体位置**：越低表示MSE越小，性能越好
- **箱体大小**：表示MSE的分散程度，箱体越小表示越稳定
- **中位数位置**：表示大多数台风的表现
- **异常值**：识别表现特别差或特别好的台风

### 使用场景
- 获得模型性能的整体统计概览
- 评估模型的稳定性和一致性
- 了解数据集的分布特征
- 识别需要特别关注的异常情况

---

## 参数说明

### MIN_SAMPLES_THRESHOLD
- **默认值**：10
- **含义**：只显示样本数大于等于该值的台风
- **作用**：过滤掉样本数太少的台风，提高可视化结果的可靠性
- **修改方法**：在代码中修改 `MIN_SAMPLES_THRESHOLD` 变量的值

---

## 使用建议

1. **初步了解**：先看统计摘要图，获得整体印象
2. **详细对比**：查看柱状图，了解每个台风的具体表现
3. **模式识别**：使用热力图发现模型在不同台风上的表现模式
4. **改进量化**：使用改进分析图量化模型间的差异
5. **深入分析**：使用散点图分析模型在不同维度上的表现

---

## 注意事项

1. **样本数影响**：样本数少的台风，MSE可能不够稳定，需要谨慎解读
2. **归一化影响**：热力图中的数值是归一化后的，不能直接与原始MSE值比较
3. **异常值**：注意识别和解释异常值，它们可能代表特殊情况
4. **综合判断**：不要仅凭单一图表做结论，应该综合多个图表的信息

---

## 文件位置

所有可视化结果保存在：
```
train/mse_visualization_results/
```

对应的数据文件在：
```
train/mse_comparison_results/mse_comparison_by_storm.csv
train/mse_comparison_results/mse_comparison_statistics.csv
```

