# 轨迹预测与可视化说明文档

本文档详细说明了 `test_trajectory_plot.py` 如何预测台风轨迹并绘制可视化图表。

## 目录

1. [概述](#1-概述)
2. [预测流程](#2-预测流程)
3. [核心函数详解](#3-核心函数详解)
4. [可视化绘制](#4-可视化绘制)
5. [数据准备](#5-数据准备)
6. [完整工作流程](#6-完整工作流程)

---

## 1. 概述

`test_trajectory_plot.py` 的主要功能：
1. 加载训练好的LSTM和Transformer模型
2. 对每个台风进行轨迹预测
3. 将预测结果与实际轨迹对比
4. 生成带有地图的可视化图表

### 预测策略

使用**滚动窗口预测（Rolling Window Prediction）**：
- 使用连续的8个时间步作为输入窗口
- 预测未来2个时间步
- 窗口向前滑动，对每个位置进行预测
- 只使用每个预测对的第一个值，形成连续轨迹

---

## 2. 预测流程

### 2.1 整体流程图

```
加载数据
    ↓
加载模型（LSTM + Transformer）
    ↓
对每个台风：
    ↓
┌─────────────────────────┐
│ 1. 提取台风数据           │
│ 2. 滚动窗口预测           │
│    - 窗口[i:i+8]         │
│    - 预测[i+8, i+9]      │
│    - 取第一个预测值       │
│ 3. 收集所有预测点         │
└─────────────────────────┘
    ↓
绘制轨迹图
    ↓
保存图片
```

### 2.2 滚动窗口预测示例

假设一个台风有15个时间步的数据：

```
时间步:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
实际值:  A  B  C  D  E  F  G  H  I  J  K  L  M  N  O
         └─────窗口1─────┘
预测值1:              P1
              └─────窗口2─────┘
预测值2:                   P2
                └─────窗口3─────┘
预测值3:                        P3
                  └─────窗口4─────┘
预测值4:                             P4
                    └─────窗口5─────┘
预测值5:                                  P5
                      └─────窗口6─────┘
预测值6:                                       P6
                        └─────窗口7─────┘
预测值7:                                            P7
```

**说明**：
- 窗口1使用时间步[0-7]预测时间步[8-9]，但只取P1（对应时间步8的预测）
- 窗口2使用时间步[1-8]预测时间步[9-10]，但只取P2（对应时间步9的预测）
- 以此类推...

**最终预测轨迹**：P1, P2, P3, P4, P5, P6, P7（7个点）

---

## 3. 核心函数详解

### 3.1 `prepare_input()` - 输入准备函数

#### 功能
将8个时间步的台风数据转换为模型输入张量。

#### 输入参数
- `sequence_df`: DataFrame，包含8行数据（8个时间步）
- `sp_data`: 字典，包含海平面气压网格数据
- `sst_data`: 字典，包含海表温度网格数据
- `lat_scaler`: MinMaxScaler，用于归一化纬度差值
- `lon_scaler`: MinMaxScaler，用于归一化经度差值

#### 处理步骤

**步骤1：计算经纬度差值**
```python
# 计算相邻时间步的经纬度差值
Latitude_Diff = Latitude[i] - Latitude[i-1]
Longitude_Diff = Longitude[i] - Longitude[i-1]
```

**步骤2：归一化差值**
```python
# 使用训练时拟合的scaler进行归一化
Latitude_Diff_scaled = lat_scaler.transform(Latitude_Diff)
Longitude_Diff_scaled = lon_scaler.transform(Longitude_Diff)
```

**步骤3：获取网格数据**
```python
# 对每个时间步，从JSON数据中获取对应的SP和SST网格
for each time step:
    key = f"{storm_name}{datetime}"
    sp_grid = sp_data[key]['sp_grid']  # 2D数组
    sst_grid = sst_data[key]['sst_grid']  # 2D数组
```

**步骤4：归一化网格数据**
```python
# 对每个网格进行min-max归一化
normalized_grid = (grid - min(grid)) / (max(grid) - min(grid))
```

**步骤5：展平网格数据**
```python
# 将2D网格展平为1D向量
sp_grid_flat = sp_grid.reshape(-1)  # 例如：21×21 → 441
sst_grid_flat = sst_grid.reshape(-1)  # 例如：21×21 → 441
```

**步骤6：拼接特征**
```python
# 对每个时间步，拼接所有特征
features_per_timestep = [
    Latitude_Diff_scaled,      # 1维
    Longitude_Diff_scaled,      # 1维
    sp_grid_flat,              # 441维
    sst_grid_flat              # 441维
]
# 总计：1 + 1 + 441 + 441 = 884维
```

**步骤7：构建输入张量**
```python
# 8个时间步，每个884维
input_tensor = torch.tensor(inputs, dtype=torch.float32)  # (8, 884)
input_tensor = input_tensor.unsqueeze(0)  # (1, 8, 884) - 添加batch维度
```

#### 输出
- 形状：`(1, 8, 884)` 的张量
- 内容：8个时间步的特征序列，每个时间步884维

---

### 3.2 `predict_next_steps()` - 预测函数

#### 功能
使用训练好的模型预测未来2个时间步的经纬度差值。

#### 输入参数
- `model_lat`: 纬度预测模型
- `model_lon`: 经度预测模型
- `input_tensor`: 输入张量 `(1, 8, 884)`
- `lat_scaler`: 纬度scaler（用于逆归一化）
- `lon_scaler`: 经度scaler（用于逆归一化）
- `model_name`: 模型名称（用于错误提示）

#### 处理步骤

**步骤1：模型前向传播**
```python
with torch.no_grad():  # 不计算梯度
    lat_out = model_lat(input_tensor)  # (1, 2)
    lon_out = model_lon(input_tensor)  # (1, 2)
```

**步骤2：NaN检测**
```python
if torch.isnan(lat_out).any() or torch.isnan(lon_out).any():
    return np.array([np.nan] * 2), np.array([np.nan] * 2)
```

**步骤3：逆归一化**
```python
# 将归一化的差值转换回实际的度数差值
lat_diffs = lat_scaler.inverse_transform(lat_out.cpu().numpy())
lon_diffs = lon_scaler.inverse_transform(lon_out.cpu().numpy())
```

#### 输出
- `lat_diffs`: 未来2个时间步的纬度差值数组（长度2）
- `lon_diffs`: 未来2个时间步的经度差值数组（长度2）

---

### 3.3 `simulate_track()` - 轨迹模拟函数

#### 功能
对指定台风进行完整的轨迹预测，使用滚动窗口方法。

#### 输入参数
- `storm_name`: 台风名称
- `data`: 完整的台风数据DataFrame
- `sp_data`: SP网格数据字典
- `sst_data`: SST网格数据字典
- `lstm_lat`, `lstm_lon`: LSTM模型
- `trans_lat`, `trans_lon`: Transformer模型
- `lat_scaler`, `lon_scaler`: 归一化器

#### 处理流程

**步骤1：提取台风数据**
```python
storm_data = data[data['Storm Name'] == storm_name]
actual_lats = storm_data['Latitude (°N)'].values
actual_lons = storm_data['Longitude (°E)'].values
```

**步骤2：滚动窗口预测**
```python
for i in range(len(storm_data) - SEQ_LEN):  # SEQ_LEN = 8
    # 获取窗口数据
    window = storm_data.iloc[i:i+8]  # 8个时间步
    
    # 准备输入
    input_tensor = prepare_input(window, sp_data, sst_data, ...)
    
    # 获取最后一个已知位置
    last_lat = window.iloc[-1]['Latitude (°N)']
    last_lon = window.iloc[-1]['Longitude (°E)']
    
    # LSTM预测
    l_lat_diff, l_lon_diff = predict_next_steps(lstm_lat, lstm_lon, ...)
    # 只取第一个预测值（对应时间步i+8）
    predicted_lat = last_lat + l_lat_diff[0]
    predicted_lon = last_lon + l_lon_diff[0]
    lstm_predictions.append((predicted_lat, predicted_lon))
    
    # Transformer预测（相同过程）
    t_lat_diff, t_lon_diff = predict_next_steps(trans_lat, trans_lon, ...)
    predicted_lat = last_lat + t_lat_diff[0]
    predicted_lon = last_lon + t_lon_diff[0]
    trans_predictions.append((predicted_lat, predicted_lon))
```

**步骤3：返回结果**
```python
return (
    actual_lats,        # 实际轨迹的纬度数组
    actual_lons,        # 实际轨迹的经度数组
    lstm_predictions,   # LSTM预测的(lat, lon)列表
    trans_predictions,  # Transformer预测的(lat, lon)列表
    plot_start_index    # 绘图起始索引（=8）
)
```

#### 预测点数量

如果台风有N个时间步：
- **预测点数** = N - 8
- **原因**：需要8个时间步作为输入，所以从第9个时间步开始预测

**示例**：
- 台风有15个时间步 → 预测7个点
- 台风有20个时间步 → 预测12个点

---

## 4. 可视化绘制

### 4.1 `plot_track_map()` - 轨迹绘制函数

#### 功能
使用Cartopy绘制带有地图的台风轨迹对比图。

#### 输入参数
- `storm_name`: 台风名称
- `actual_lats`, `actual_lons`: 实际轨迹坐标
- `lstm_preds`: LSTM预测的(lat, lon)列表
- `trans_preds`: Transformer预测的(lat, lon)列表
- `start_idx`: 绘图起始索引
- `output_dir`: 输出目录

#### 绘制步骤

**步骤1：过滤有效值**
```python
# 过滤掉NaN和Inf值
valid_lstm_lats = [p[0] for p in lstm_preds if not (np.isnan(p[0]) or np.isinf(p[0]))]
valid_lstm_lons = [p[1] for p in lstm_preds if not (np.isnan(p[1]) or np.isinf(p[1]))]
# 同样处理Transformer预测
```

**步骤2：确定地图范围**
```python
# 计算所有坐标的最小/最大值
all_valid_lats = actual_lats + valid_lstm_lats + valid_trans_lats
all_valid_lons = actual_lons + valid_lstm_lons + valid_trans_lons

min_lat = min(all_valid_lats) - 5  # 留5度边距
max_lat = max(all_valid_lats) + 5
min_lon = min(all_valid_lons) - 5
max_lon = max(all_valid_lons) + 5
```

**步骤3：创建地图投影**
```python
fig, ax = plt.subplots(
    subplot_kw={'projection': ccrs.PlateCarree()}, 
    figsize=(10, 8)
)
ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
```

**步骤4：添加地图要素**
```python
ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.OCEAN)
ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
```

**步骤5：绘制轨迹**

**实际轨迹**：
```python
ax.plot(actual_lons, actual_lats, 
       'k-o',  # 黑色实线，圆点标记
       label='Actual Track', 
       transform=ccrs.PlateCarree(), 
       markersize=4, 
       linewidth=2)
```

**LSTM预测轨迹**：
```python
ax.plot(lstm_lons, lstm_lats, 
       'b--x',  # 蓝色虚线，X标记
       label=f'LSTM Prediction ({len(lstm_lats)} points)', 
       transform=ccrs.PlateCarree(), 
       markersize=4, 
       linewidth=1.5)
```

**Transformer预测轨迹**：
```python
ax.plot(trans_lons, trans_lats, 
       'r--^',  # 红色虚线，三角标记
       label=f'Transformer Prediction ({len(trans_lats)} points)', 
       transform=ccrs.PlateCarree(), 
       markersize=4, 
       linewidth=1.5)
```

**步骤6：添加图例和标题**
```python
plt.legend(loc='upper right')
plt.title(f"Typhoon {storm_name} Trajectory Prediction")
```

**步骤7：保存图片**
```python
output_file = os.path.join(output_dir, f"{storm_name}_trajectory.png")
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()  # 关闭图形，释放内存
```

#### 图例说明

- **黑色实线（k-o）**：实际轨迹
- **蓝色虚线（b--x）**：LSTM预测轨迹
- **红色虚线（r--^）**：Transformer预测轨迹

---

## 5. 数据准备

### 5.1 数据加载

```python
# 加载CSV数据
data = pd.read_csv('best_track_records_p6.csv')

# 加载JSON数据
with open('sp_data_matrix.json', 'r') as f:
    sp_data = json.load(f)
with open('sst_data_matrix.json', 'r') as f:
    sst_data = json.load(f)
```

### 5.2 Scaler拟合

```python
# 计算所有台风的经纬度差值
temp_data = data.copy()
temp_grouped = temp_data.groupby('Storm Name')
def compute_diff(group):
    group['Latitude_Diff'] = group['Latitude (°N)'].diff().fillna(0)
    group['Longitude_Diff'] = group['Longitude (°E)'].diff().fillna(0)
    return group
temp_data = temp_grouped.apply(compute_diff).reset_index(drop=True)

# 拟合scaler（必须与训练时使用相同的scaler）
lat_scaler.fit(temp_data[['Latitude_Diff']])
lon_scaler.fit(temp_data[['Longitude_Diff']])
```

**重要**：必须使用与训练时相同的方法拟合scaler，否则预测结果会不准确。

---

## 6. 完整工作流程

### 6.1 主函数流程

```python
def main():
    # 1. 加载数据和Scalers
    data = pd.read_csv(csv_path)
    sp_data = json.load(sp_json_path)
    sst_data = json.load(sst_json_path)
    lat_scaler, lon_scaler = fit_scalers(data)
    
    # 2. 加载模型
    lstm_lat = torch.load("lat_cnn_lstm_model.pth")
    lstm_lon = torch.load("lon_cnn_lstm_model.pth")
    trans_lat = torch.load("lat_cnn_transformer_model.pth")
    trans_lon = torch.load("lon_cnn_transformer_model.pth")
    
    # 3. 获取所有台风
    unique_storms = data['Storm Name'].unique()
    
    # 4. 创建输出目录
    output_dir = 'trajectory_comparison_results'
    
    # 5. 对每个台风进行预测和绘制
    for storm in unique_storms:
        result = simulate_track(storm, data, sp_data, sst_data, 
                              lstm_lat, lstm_lon, trans_lat, trans_lon,
                              lat_scaler, lon_scaler)
        if result:
            plot_track_map(storm, *result, output_dir)
```

### 6.2 执行顺序

1. **数据准备阶段**
   - 加载CSV和JSON数据
   - 拟合归一化器

2. **模型加载阶段**
   - 加载4个模型（LSTM纬度、LSTM经度、Transformer纬度、Transformer经度）
   - 设置为评估模式（`model.eval()`）

3. **预测阶段**
   - 遍历每个台风
   - 对每个台风进行滚动窗口预测
   - 收集LSTM和Transformer的预测结果

4. **可视化阶段**
   - 绘制地图和轨迹
   - 保存图片到指定目录

---

## 7. 关键参数

### 7.1 模型参数

```python
SEQ_LEN = 8      # 输入序列长度（必须与训练时一致）
PRED_LEN = 2     # 预测长度（必须与训练时一致）
input_dim = 884  # 输入特征维度
```

### 7.2 预测参数

- **窗口大小**：8个时间步
- **预测步数**：2步（但只使用第一步）
- **滑动步长**：1（每次向前移动1个时间步）

---

## 8. 输出结果

### 8.1 文件位置

所有轨迹图保存在：
```
train/trajectory_comparison_results/
  ├── Storm1_trajectory.png
  ├── Storm2_trajectory.png
  └── ...
```

### 8.2 图片内容

每张图片包含：
- **地图背景**：使用Cartopy绘制的地理地图
- **实际轨迹**：黑色实线，显示台风的真实路径
- **LSTM预测**：蓝色虚线，显示LSTM模型的预测路径
- **Transformer预测**：红色虚线，显示Transformer模型的预测路径
- **图例**：说明各条线的含义
- **标题**：台风名称

---

## 9. 常见问题

### Q1: 为什么预测点数比实际数据少？

**原因**：
- 需要8个时间步作为输入窗口
- 如果台风有N个时间步，只能预测N-8个点
- 这是滚动窗口预测的正常现象

### Q2: 为什么只使用预测对的第一个值？

**原因**：
- 为了形成连续的预测轨迹
- 如果使用两个值，轨迹会有跳跃
- 只使用第一个值可以形成平滑的预测线

### Q3: 如何处理预测失败（NaN）？

**处理方式**：
- 在`predict_next_steps()`中检测NaN
- 如果检测到NaN，返回NaN数组
- 在`simulate_track()`中，如果预测为NaN，添加NaN到预测列表
- 在`plot_track_map()`中，过滤掉NaN值，只绘制有效点

### Q4: 为什么需要逆归一化？

**原因**：
- 模型输出的是归一化后的差值
- 需要转换为实际的经纬度差值（度数）
- 然后加上最后一个已知位置，得到预测的绝对坐标

---

## 10. 代码位置

主要函数位置：
- `prepare_input()`: 第134-209行
- `predict_next_steps()`: 第211-245行
- `simulate_track()`: 第247-314行
- `plot_track_map()`: 第316-389行
- `main()`: 第391-487行

---

## 11. 使用示例

```bash
# 运行轨迹预测和可视化
cd train
python test_trajectory_plot.py
```

**输出**：
```
Using device: cuda
Loading data...
Loading models...
Found 28 unique typhoons
Output directory: train/trajectory_comparison_results
Saved trajectory plot to .../Storm1_trajectory.png
Saved trajectory plot to .../Storm2_trajectory.png
...
Processing complete!
Successfully processed: 27 typhoons
Failed/Skipped: 1 typhoons
```

---

## 12. 扩展阅读

- Cartopy文档：https://scitools.org.uk/cartopy/docs/latest/
- Matplotlib文档：https://matplotlib.org/stable/contents.html
- PyTorch文档：https://pytorch.org/docs/stable/index.html

