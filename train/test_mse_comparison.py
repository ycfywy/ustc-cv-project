import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import os
import math
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_dir = os.path.join(project_root, 'data_preprocess')
model_dir = os.path.join(current_dir, 'model')
output_dir = os.path.join(current_dir, 'mse_comparison_results')

# Create output directory
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(data_dir, 'best_track_records_p6.csv')
sp_json_path = os.path.join(data_dir, 'sp_data_matrix.json')
sst_json_path = os.path.join(data_dir, 'sst_data_matrix.json')

# Model Parameters (Must match training)
SEQ_LEN = 8
PRED_LEN = 2

# --- Model Definitions ---

class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, cnn_channels=256, kernel_size=3, lstm_layers=3):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=cnn_channels * 2, out_channels=cnn_channels * 4, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(cnn_channels * 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=2)
        )

        self.lstm = nn.LSTM(input_size=cnn_channels * 4, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, PRED_LEN),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = x.transpose(1, 2)  
        x = self.cnn(x)
        x = x.transpose(1, 2)  
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :] 
        output = self.fc(last_out)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class CNNTransformer(nn.Module):
    def __init__(self, input_dim, cnn_channels=256, kernel_size=3, nhead=8, num_encoder_layers=2, dim_feedforward=512, dropout=0.1):
        super(CNNTransformer, self).__init__()
        # 使用更保守的CNN设计，保持序列长度
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.LeakyReLU(negative_slope=0.01),
            # 不使用MaxPool，保持序列长度为8

            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Conv1d(in_channels=cnn_channels * 2, out_channels=cnn_channels * 4, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(cnn_channels * 4),
            nn.LeakyReLU(negative_slope=0.01)
            # 完全移除MaxPool，保持序列长度为8
        )
        
        self.d_model = cnn_channels * 4  # 1024
        # 确保nhead能整除d_model
        assert self.d_model % nhead == 0, f"d_model ({self.d_model}) must be divisible by nhead ({nhead})"
        
        # 使用更稳定的位置编码（缩放版本）
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # 使用更少的层数和更稳定的配置
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True,
            activation='gelu',
            norm_first=True  # Pre-norm架构，更稳定
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_encoder_layers)
        
        # 添加LayerNorm稳定输出
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, PRED_LEN)
        )

    def forward(self, x):
        # 输入形状: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # (batch, seq_len, features)
        
        # 检查NaN
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        transformer_out = self.transformer_encoder(x)
        
        # 检查NaN
        if torch.isnan(transformer_out).any():
            transformer_out = torch.nan_to_num(transformer_out, nan=0.0)
        
        # LayerNorm
        transformer_out = self.layer_norm(transformer_out)
        
        # 取最后一个时间步
        last_out = transformer_out[:, -1, :] 
        
        # 全连接层
        output = self.fc(last_out)
        
        # 最终检查
        if torch.isnan(output).any():
            output = torch.nan_to_num(output, nan=0.0)
        
        return output

# --- Helper Functions ---

def normalize_grid(grid):
    grid_min = np.min(grid)
    grid_max = np.max(grid)
    if grid_max == grid_min:
        return grid - grid_min
    return (grid - grid_min) / (grid_max - grid_min)

def prepare_input(sequence_df, sp_data, sst_data, lat_scaler, lon_scaler):
    """
    Prepares a single input tensor from a sequence of 8 time steps.
    """
    # Calculate differences
    seq_copy = sequence_df.copy()
    seq_copy['Latitude_Diff'] = seq_copy['Latitude (°N)'].diff().fillna(0)
    seq_copy['Longitude_Diff'] = seq_copy['Longitude (°E)'].diff().fillna(0)
    
    # Scale differences
    seq_copy['Latitude_Diff'] = lat_scaler.transform(seq_copy[['Latitude_Diff']])
    seq_copy['Longitude_Diff'] = lon_scaler.transform(seq_copy[['Longitude_Diff']])
    
    inputs_diff = seq_copy[['Latitude_Diff', 'Longitude_Diff']].values.astype(np.float32)
    
    datetime_keys = seq_copy['DateTime(UTC)'].tolist()
    storm_name = seq_copy['Storm Name'].iloc[0]
    
    # Retrieve and normalize grid data
    sp_grids = [sp_data[f"{storm_name}{dt}"]['sp_grid'] for dt in datetime_keys]
    sst_grids = [sst_data[f"{storm_name}{dt}"]['sst_grid'] for dt in datetime_keys]
    
    sp_grids = np.array([normalize_grid(sp) for sp in sp_grids])
    sst_grids = np.array([normalize_grid(sst) for sst in sst_grids])
    
    sp_grids_flat = sp_grids.reshape(sp_grids.shape[0], -1)
    sst_grids_flat = sst_grids.reshape(sst_grids.shape[0], -1)
    
    # Concatenate features
    inputs = np.concatenate([inputs_diff, sp_grids_flat, sst_grids_flat], axis=1)
    
    # Convert to tensor and add batch dimension
    return torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(device)

def compute_mse_for_storm(storm_name, data, sp_data, sst_data, 
                          lstm_lat, lstm_lon, trans_lat, trans_lon,
                          lat_scaler, lon_scaler):
    """
    Compute MSE for a single storm using both models.
    Returns: (lstm_lat_mse, lstm_lon_mse, trans_lat_mse, trans_lon_mse, num_samples)
    """
    storm_data = data[data['Storm Name'] == storm_name].reset_index(drop=True)
    
    if len(storm_data) < SEQ_LEN + PRED_LEN:
        return None
    
    lstm_lat_errors = []
    lstm_lon_errors = []
    trans_lat_errors = []
    trans_lon_errors = []
    
    # Rolling window prediction
    for i in range(len(storm_data) - SEQ_LEN):
        if i + SEQ_LEN + PRED_LEN > len(storm_data):
            break
            
        # Get input window
        window = storm_data.iloc[i:i+SEQ_LEN]
        
        # Get actual targets
        actual_window = storm_data.iloc[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN]
        
        if len(actual_window) == 0:
            continue
            
        # Get last known position
        last_lat = window.iloc[-1]['Latitude (°N)']
        last_lon = window.iloc[-1]['Longitude (°E)']
        
        # Calculate actual differences from last known position
        actual_lat_diffs = actual_window['Latitude (°N)'].values - last_lat
        actual_lon_diffs = actual_window['Longitude (°E)'].values - last_lon
        
        # Prepare input
        input_tensor = prepare_input(window, sp_data, sst_data, lat_scaler, lon_scaler)
        
        # Predict with both models
        with torch.no_grad():
            lstm_lat_out = lstm_lat(input_tensor)
            lstm_lon_out = lstm_lon(input_tensor)
            trans_lat_out = trans_lat(input_tensor)
            trans_lon_out = trans_lon(input_tensor)
        
        # 检查NaN
        if torch.isnan(lstm_lat_out).any() or torch.isnan(lstm_lon_out).any():
            continue
        if torch.isnan(trans_lat_out).any() or torch.isnan(trans_lon_out).any():
            # Transformer输出NaN，跳过这个样本
            continue
        
        # Inverse transform predictions
        lstm_lat_diffs = lat_scaler.inverse_transform(lstm_lat_out.cpu().numpy().reshape(-1, 1)).flatten()
        lstm_lon_diffs = lon_scaler.inverse_transform(lstm_lon_out.cpu().numpy().reshape(-1, 1)).flatten()
        trans_lat_diffs = lat_scaler.inverse_transform(trans_lat_out.cpu().numpy().reshape(-1, 1)).flatten()
        trans_lon_diffs = lon_scaler.inverse_transform(trans_lon_out.cpu().numpy().reshape(-1, 1)).flatten()
        
        # 再次检查NaN
        if np.isnan(lstm_lat_diffs).any() or np.isnan(lstm_lon_diffs).any() or \
           np.isnan(trans_lat_diffs).any() or np.isnan(trans_lon_diffs).any():
            continue
        
        # Compute MSE for each prediction step
        for step in range(min(PRED_LEN, len(actual_lat_diffs))):
            lstm_lat_errors.append((actual_lat_diffs[step] - lstm_lat_diffs[step]) ** 2)
            lstm_lon_errors.append((actual_lon_diffs[step] - lstm_lon_diffs[step]) ** 2)
            trans_lat_errors.append((actual_lat_diffs[step] - trans_lat_diffs[step]) ** 2)
            trans_lon_errors.append((actual_lon_diffs[step] - trans_lon_diffs[step]) ** 2)
    
    if len(lstm_lat_errors) == 0:
        return None
    
    lstm_lat_mse = np.mean(lstm_lat_errors)
    lstm_lon_mse = np.mean(lstm_lon_errors)
    trans_lat_mse = np.mean(trans_lat_errors) if len(trans_lat_errors) > 0 else np.nan
    trans_lon_mse = np.mean(trans_lon_errors) if len(trans_lon_errors) > 0 else np.nan
    
    return (lstm_lat_mse, lstm_lon_mse, trans_lat_mse, trans_lon_mse, len(lstm_lat_errors))

def main():
    # 1. Load Data and Scalers
    print("Loading data...")
    data = pd.read_csv(csv_path)
    data['Latitude (°N)'] = pd.to_numeric(data['Latitude (°N)'], errors='coerce')
    data['Longitude (°E)'] = pd.to_numeric(data['Longitude (°E)'], errors='coerce')
    
    with open(sp_json_path, 'r') as f:
        sp_data = json.load(f)
    with open(sst_json_path, 'r') as f:
        sst_data = json.load(f)
        
    # Fit scalers globally (same as training logic)
    lat_scaler = MinMaxScaler()
    lon_scaler = MinMaxScaler()
    
    temp_data = data.copy()
    temp_grouped = temp_data.groupby('Storm Name')
    def compute_diff(group):
        group = group.copy()  # 避免SettingWithCopyWarning
        group['Latitude_Diff'] = group['Latitude (°N)'].diff().fillna(0)
        group['Longitude_Diff'] = group['Longitude (°E)'].diff().fillna(0)
        return group
    temp_data = temp_grouped.apply(compute_diff).reset_index(drop=True)
    
    lat_scaler.fit(temp_data[['Latitude_Diff']])
    lon_scaler.fit(temp_data[['Longitude_Diff']])
    
    # 2. Determine Input Dimension
    input_dim = 884  # From training
    
    # 3. Load Models
    print("Loading models...")
    lstm_lat = CNNLSTM(input_dim).to(device)
    lstm_lon = CNNLSTM(input_dim).to(device)
    trans_lat = CNNTransformer(input_dim).to(device)
    trans_lon = CNNTransformer(input_dim).to(device)
    
    try:
        lstm_lat = torch.load(os.path.join(model_dir, "lat_cnn_lstm_model.pth"), map_location=device)
        lstm_lon = torch.load(os.path.join(model_dir, "lon_cnn_lstm_model.pth"), map_location=device)
        lstm_lat.eval()
        lstm_lon.eval()
        print("LSTM models loaded successfully")
    except Exception as e:
        print(f"Error loading LSTM models: {e}")
        return
        
    try:
        trans_lat = torch.load(os.path.join(model_dir, "lat_cnn_transformer_model.pth"), map_location=device)
        trans_lon = torch.load(os.path.join(model_dir, "lon_cnn_transformer_model.pth"), map_location=device)
        trans_lat.eval()
        trans_lon.eval()
        print("Transformer models loaded successfully")
    except Exception as e:
        print(f"Error loading Transformer models: {e}")
        return

    # 4. Get all unique typhoons
    unique_storms = data['Storm Name'].unique()
    print(f"\nFound {len(unique_storms)} unique typhoons")
    print("Computing MSE for each typhoon...\n")
    
    # 5. Compute MSE for each storm
    results = []
    
    for storm in unique_storms:
        result = compute_mse_for_storm(storm, data, sp_data, sst_data,
                                      lstm_lat, lstm_lon, trans_lat, trans_lon,
                                      lat_scaler, lon_scaler)
        if result:
            lstm_lat_mse, lstm_lon_mse, trans_lat_mse, trans_lon_mse, num_samples = result
            results.append({
                'Storm Name': storm,
                'LSTM_Latitude_MSE': lstm_lat_mse,
                'LSTM_Longitude_MSE': lstm_lon_mse,
                'Transformer_Latitude_MSE': trans_lat_mse,
                'Transformer_Longitude_MSE': trans_lon_mse,
                'LSTM_Total_MSE': lstm_lat_mse + lstm_lon_mse,
                'Transformer_Total_MSE': trans_lat_mse + trans_lon_mse,
                'Num_Samples': num_samples
            })
            print(f"{storm}: LSTM Total MSE = {lstm_lat_mse + lstm_lon_mse:.6f}, "
                  f"Transformer Total MSE = {trans_lat_mse + trans_lon_mse:.6f}")
    
    # 6. Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # 7. Compute overall statistics
    overall_stats = {
        'Metric': [
            'LSTM Latitude MSE (Mean)',
            'LSTM Longitude MSE (Mean)',
            'LSTM Total MSE (Mean)',
            'Transformer Latitude MSE (Mean)',
            'Transformer Longitude MSE (Mean)',
            'Transformer Total MSE (Mean)',
            'LSTM Latitude MSE (Std)',
            'LSTM Longitude MSE (Std)',
            'LSTM Total MSE (Std)',
            'Transformer Latitude MSE (Std)',
            'Transformer Longitude MSE (Std)',
            'Transformer Total MSE (Std)',
        ],
        'Value': [
            results_df['LSTM_Latitude_MSE'].mean(),
            results_df['LSTM_Longitude_MSE'].mean(),
            results_df['LSTM_Total_MSE'].mean(),
            results_df['Transformer_Latitude_MSE'].mean(),
            results_df['Transformer_Longitude_MSE'].mean(),
            results_df['Transformer_Total_MSE'].mean(),
            results_df['LSTM_Latitude_MSE'].std(),
            results_df['LSTM_Longitude_MSE'].std(),
            results_df['LSTM_Total_MSE'].std(),
            results_df['Transformer_Latitude_MSE'].std(),
            results_df['Transformer_Longitude_MSE'].std(),
            results_df['Transformer_Total_MSE'].std(),
        ]
    }
    stats_df = pd.DataFrame(overall_stats)
    
    # 8. Save results
    results_csv_path = os.path.join(output_dir, 'mse_comparison_by_storm.csv')
    stats_csv_path = os.path.join(output_dir, 'mse_comparison_statistics.csv')
    
    results_df.to_csv(results_csv_path, index=False)
    stats_df.to_csv(stats_csv_path, index=False)
    
    print(f"\n{'='*60}")
    print("MSE Comparison Results")
    print(f"{'='*60}")
    print(f"\nOverall Statistics:")
    print(stats_df.to_string(index=False))
    print(f"\n\nResults saved to:")
    print(f"  - Detailed results: {results_csv_path}")
    print(f"  - Statistics: {stats_csv_path}")
    print(f"\nTotal storms analyzed: {len(results_df)}")

if __name__ == "__main__":
    main()

