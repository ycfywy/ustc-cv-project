# CNNTransformer 架构说明文档

本文档详细说明了CNNTransformer模型的架构设计、各组件功能以及数据流转过程。

## 目录

1. [模型概述](#1-模型概述)
2. [整体架构](#2-整体架构)
3. [各组件详解](#3-各组件详解)
4. [数据流转过程](#4-数据流转过程)
5. [关键设计决策](#5-关键设计决策)
6. [与CNNLSTM的对比](#6-与cnnlstm的对比)

---

## 1. 模型概述

CNNTransformer是一个混合架构模型，结合了：
- **CNN（卷积神经网络）**：用于提取局部特征和降维
- **Transformer Encoder**：用于捕获序列中的长距离依赖关系
- **全连接层**：用于最终预测

该模型专门设计用于台风轨迹预测任务，输入是8个时间步的特征序列，输出是未来2个时间步的预测值。

---

## 2. 整体架构

```
输入 (batch, seq_len=8, input_dim=884)
    ↓
[转置] (batch, input_dim, seq_len)
    ↓
┌─────────────────────────────────────┐
│  CNN特征提取层                        │
│  - Conv1d: 884 → 256                 │
│  - Conv1d: 256 → 512                 │
│  - Conv1d: 512 → 1024                │
│  (保持序列长度=8)                     │
└─────────────────────────────────────┘
    ↓
[转置] (batch, seq_len=8, d_model=1024)
    ↓
┌─────────────────────────────────────┐
│  位置编码 (Positional Encoding)      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Transformer Encoder                │
│  - 2层 TransformerEncoderLayer      │
│  - Pre-norm架构                      │
│  - 8个注意力头                        │
└─────────────────────────────────────┘
    ↓
[LayerNorm] (batch, seq_len=8, d_model=1024)
    ↓
[取最后一个时间步] (batch, d_model=1024)
    ↓
┌─────────────────────────────────────┐
│  全连接层 (FC)                        │
│  - Linear: 1024 → 512                │
│  - LayerNorm + LeakyReLU + Dropout  │
│  - Linear: 512 → 2 (PRED_LEN)       │
└─────────────────────────────────────┘
    ↓
输出 (batch, 2)
```

---

## 3. 各组件详解

### 3.1 CNN特征提取层

#### 设计目标
- 从高维输入（884维）中提取有效特征
- 将特征维度扩展到适合Transformer的维度（1024维）
- **保持序列长度不变**（关键设计决策）

#### 具体结构
```python
CNN层结构：
1. Conv1d(input_dim=884, out_channels=256, kernel_size=3, padding=1)
   → BatchNorm1d(256) → LeakyReLU(0.01)
   
2. Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
   → BatchNorm1d(512) → LeakyReLU(0.01)
   
3. Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
   → BatchNorm1d(1024) → LeakyReLU(0.01)
```

#### 关键参数
- **kernel_size=3**：使用3×1的卷积核，捕获局部时间模式
- **padding=1**：保持序列长度不变（8 → 8）
- **不使用MaxPool**：这是与CNNLSTM的关键区别，确保序列长度保持为8

#### 为什么保持序列长度？
- Transformer需要序列信息来学习时间依赖关系
- 如果序列长度变为1，Transformer无法工作
- 保持序列长度=8，Transformer可以学习8个时间步之间的关系

---

### 3.2 位置编码 (Positional Encoding)

#### 作用
为序列中的每个时间步添加位置信息，因为Transformer本身不包含位置信息。

#### 实现方式
使用正弦和余弦函数生成位置编码：

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

其中：
- `pos`：位置索引（0-7，对应8个时间步）
- `i`：维度索引
- `d_model`：模型维度（1024）

#### 特点
- 位置编码是固定的，不参与训练
- 可以处理任意长度的序列（理论上）
- 通过正弦/余弦函数的周期性，可以学习相对位置关系

---

### 3.3 Transformer Encoder

#### 架构配置
```python
TransformerEncoderLayer配置：
- d_model: 1024 (特征维度)
- nhead: 8 (注意力头数，1024/8=128，每个头128维)
- num_layers: 2 (编码器层数)
- dim_feedforward: 512 (前馈网络维度)
- dropout: 0.1 (dropout率)
- activation: 'gelu' (激活函数)
- norm_first: True (Pre-norm架构)
- batch_first: True (批次维度在前)
```

#### Pre-norm vs Post-norm
本模型使用**Pre-norm架构**（`norm_first=True`），这是关键的设计决策：

**Pre-norm架构**（本模型使用）：
```
输入 → LayerNorm → Multi-Head Attention → 残差连接
     → LayerNorm → Feed Forward → 残差连接 → 输出
```

**Post-norm架构**（传统）：
```
输入 → Multi-Head Attention → LayerNorm → 残差连接
     → Feed Forward → LayerNorm → 残差连接 → 输出
```

**为什么使用Pre-norm？**
- 训练更稳定，梯度流动更好
- 减少梯度消失问题
- 对于深层网络更有效

#### Multi-Head Attention机制
- **8个注意力头**：每个头关注不同的特征子空间
- **自注意力**：序列中的每个位置可以关注到所有其他位置
- **计算过程**：
  1. 将输入分为8个头，每个头128维
  2. 计算Query、Key、Value矩阵
  3. 计算注意力分数：`Attention(Q, K, V) = softmax(QK^T / √d_k) V`
  4. 合并8个头的输出

#### Feed Forward Network
- 两层全连接网络：1024 → 512 → 1024
- 使用GELU激活函数
- 提供非线性变换能力

---

### 3.4 输出层 (Fully Connected Layers)

#### 结构
```python
FC层结构：
1. Linear(1024 → 512)
   → LayerNorm(512)
   → LeakyReLU(0.01)
   → Dropout(0.1)
   
2. Linear(512 → 2)
```

#### 设计考虑
- **两层设计**：提供足够的非线性变换能力
- **LayerNorm**：稳定训练，防止梯度爆炸
- **Dropout**：防止过拟合
- **输出维度=2**：对应PRED_LEN=2，预测未来2个时间步

---

## 4. 数据流转过程

### 4.1 输入数据格式

**输入形状**：`(batch_size, seq_len=8, input_dim=884)`

**输入内容**：
- 每个时间步包含884维特征：
  - 2维：经纬度差值（Latitude_Diff, Longitude_Diff）
  - 约441维：SP（海平面气压）网格数据（展平后）
  - 约441维：SST（海表温度）网格数据（展平后）

### 4.2 前向传播过程

#### 步骤1：CNN特征提取
```python
输入: (batch, 8, 884)
  ↓ transpose(1, 2)
(batch, 884, 8)
  ↓ CNN层
(batch, 1024, 8)  # 特征维度从884扩展到1024，序列长度保持8
  ↓ transpose(1, 2)
(batch, 8, 1024)  # 转回序列格式
```

#### 步骤2：位置编码
```python
(batch, 8, 1024)
  ↓ 位置编码（逐元素相加）
(batch, 8, 1024)  # 每个时间步都添加了位置信息
```

#### 步骤3：Transformer编码
```python
(batch, 8, 1024)
  ↓ Transformer Encoder (2层)
(batch, 8, 1024)  # 每个时间步都包含了全局上下文信息
```

#### 步骤4：提取最后时间步
```python
(batch, 8, 1024)
  ↓ 取最后一个时间步 [:, -1, :]
(batch, 1024)  # 只保留最后一个时间步的特征
```

#### 步骤5：全连接层预测
```python
(batch, 1024)
  ↓ FC层
(batch, 2)  # 输出未来2个时间步的预测值
```

### 4.3 输出格式

**输出形状**：`(batch_size, PRED_LEN=2)`

**输出含义**：
- 对于纬度模型：输出2个未来时间步的纬度差值
- 对于经度模型：输出2个未来时间步的经度差值
- 需要经过逆归一化（inverse transform）才能得到实际的经纬度差值

---

## 5. 关键设计决策

### 5.1 为什么移除MaxPool？

**问题**：原始设计中使用了3次MaxPool，导致序列长度：8 → 4 → 2 → 1

**解决方案**：完全移除MaxPool，保持序列长度为8

**原因**：
1. Transformer需要序列信息来学习时间依赖
2. 序列长度为1时，Transformer无法工作
3. 保持序列长度=8，Transformer可以学习8个时间步之间的复杂关系

### 5.2 为什么使用Pre-norm架构？

**优势**：
- 训练更稳定，减少梯度消失问题
- 对于深层网络更有效
- 在Transformer中表现更好

### 5.3 为什么使用2层而不是3层？

**原因**：
- 减少模型复杂度，降低过拟合风险
- 2层已经足够捕获序列依赖关系
- 减少训练时间，提高训练稳定性

### 5.4 为什么d_model=1024？

**考虑**：
- 需要足够大的维度来编码丰富的特征信息
- 1024是8的倍数，可以均匀分配给8个注意力头
- 与CNN的输出维度（256×4=1024）匹配

### 5.5 NaN检测和处理

模型在多个关键点进行NaN检测：
1. CNN输出后
2. Transformer输出后
3. 最终输出前

如果检测到NaN，使用`torch.nan_to_num(x, nan=0.0)`替换为0，防止训练崩溃。

---

## 6. 与CNNLSTM的对比

### 6.1 架构对比

| 特性 | CNNLSTM | CNNTransformer |
|------|---------|----------------|
| CNN层 | 3层，使用MaxPool | 3层，不使用MaxPool |
| 序列长度变化 | 8 → 4 → 2 → 1 | 8 → 8（保持不变） |
| 序列建模 | LSTM（3层） | Transformer Encoder（2层） |
| 位置信息 | LSTM隐式处理 | 显式位置编码 |
| 注意力机制 | 无 | Multi-Head Attention（8头） |
| 架构类型 | Post-norm | Pre-norm |

### 6.2 优势对比

**CNNLSTM的优势**：
- 训练更稳定，不容易出现NaN
- 计算效率更高
- 对于短序列任务表现良好

**CNNTransformer的优势**：
- 可以并行处理所有时间步（LSTM是顺序的）
- 注意力机制可以学习长距离依赖
- 理论上可以处理更长的序列

### 6.3 适用场景

**CNNLSTM适合**：
- 序列长度较短（≤10）
- 需要快速训练
- 对稳定性要求高

**CNNTransformer适合**：
- 需要捕获长距离依赖
- 序列长度较长
- 需要并行计算

---

## 7. 模型参数统计

### 参数量估算

**CNN层**：
- Conv1d(884→256): 约 884 × 256 × 3 ≈ 680K
- Conv1d(256→512): 约 256 × 512 × 3 ≈ 393K
- Conv1d(512→1024): 约 512 × 1024 × 3 ≈ 1.5M
- BatchNorm参数：约 256 + 512 + 1024 = 1.8K
- **CNN总计**：约 2.6M参数

**Transformer Encoder**：
- 每层TransformerEncoderLayer：
  - Multi-Head Attention: 约 4 × 1024² ≈ 4.2M
  - Feed Forward: 约 2 × 1024 × 512 ≈ 1.0M
  - LayerNorm: 约 2 × 1024 = 2K
- 2层总计：约 10.4M参数

**输出层**：
- Linear(1024→512): 约 1024 × 512 ≈ 524K
- Linear(512→2): 约 512 × 2 ≈ 1K
- **FC总计**：约 525K参数

**总参数量**：约 **13.5M参数**

---

## 8. 训练配置

### 超参数设置

```python
学习率: LEARNING_RATE * 0.1 = 0.0001  # LSTM的10%
梯度裁剪: 0.5  # 更严格的梯度裁剪
Dropout: 0.1
优化器: Adam
权重衰减: 1e-4
```

### 为什么使用更小的学习率？

- Transformer对学习率更敏感
- 更小的学习率可以防止训练不稳定
- 配合梯度裁剪，确保训练过程稳定

---

## 9. 常见问题

### Q1: 为什么Transformer输出NaN？

**可能原因**：
1. 序列长度变为1（已通过移除MaxPool解决）
2. 梯度爆炸（已通过梯度裁剪和更小学习率解决）
3. 输入数据异常

**解决方案**：
- 使用Pre-norm架构
- 降低学习率
- 添加梯度裁剪
- 在关键点进行NaN检测和修复

### Q2: 为什么d_model必须是nhead的倍数？

**原因**：
- Multi-Head Attention需要将d_model均匀分配给nhead个头
- 每个头的维度 = d_model / nhead
- 如果无法整除，会导致维度不匹配

### Q3: 为什么只取最后一个时间步？

**原因**：
- 最后一个时间步包含了前面所有时间步的信息（通过注意力机制）
- 对于预测任务，我们关心的是基于整个序列的最终状态
- 简化输出，直接得到预测值

---

## 10. 代码位置

模型定义位置：
- 训练代码：`train/train.py` (第177-229行)
- 测试代码：`train/test_trajectory_plot.py` (第83-166行)
- MSE对比代码：`train/test_mse_comparison.py` (第85-167行)

模型保存位置：
- `train/model/lat_cnn_transformer_model.pth`
- `train/model/lon_cnn_transformer_model.pth`

---

## 参考文献

- Transformer原始论文：Attention Is All You Need (Vaswani et al., 2017)
- Pre-norm架构：On Layer Normalization in the Transformer Architecture (Xiong et al., 2020)

