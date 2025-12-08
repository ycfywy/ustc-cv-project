import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch
import torch.nn as nn
import json
import os
import math
from sklearn.preprocessing import MinMaxScaler

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_dir = os.path.join(project_root, 'data_preprocess')
model_dir = os.path.join(current_dir, 'model')

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

def predict_next_steps(model_lat, model_lon, input_tensor, lat_scaler, lon_scaler, model_name=""):
    """
    Predicts the next PRED_LEN steps (differences) given an input tensor.
    """
    with torch.no_grad():
        try:
            lat_out = model_lat(input_tensor)
            lon_out = model_lon(input_tensor)
            
            # 检查NaN
            if torch.isnan(lat_out).any() or torch.isnan(lon_out).any():
                if model_name:
                    print(f"Warning: {model_name} model output contains NaN")
                return np.array([np.nan] * PRED_LEN), np.array([np.nan] * PRED_LEN)
            
        except Exception as e:
            if model_name:
                print(f"Error in {model_name} prediction: {e}")
            return np.array([np.nan] * PRED_LEN), np.array([np.nan] * PRED_LEN)
    
    # Inverse transform to get actual degree differences
    try:
        lat_diffs = lat_scaler.inverse_transform(lat_out.cpu().numpy().reshape(-1, 1)).flatten()
        lon_diffs = lon_scaler.inverse_transform(lon_out.cpu().numpy().reshape(-1, 1)).flatten()
        
        # 再次检查NaN
        if np.isnan(lat_diffs).any() or np.isnan(lon_diffs).any():
            if model_name:
                print(f"Warning: {model_name} inverse transform contains NaN")
    except Exception as e:
        if model_name:
            print(f"Error in {model_name} inverse transform: {e}")
        return np.array([np.nan] * PRED_LEN), np.array([np.nan] * PRED_LEN)
    
    return lat_diffs, lon_diffs

def simulate_track(storm_name, data, sp_data, sst_data, 
                  lstm_lat, lstm_lon, trans_lat, trans_lon, 
                  lat_scaler, lon_scaler):
    """
    Simulates the track prediction for a given storm.
    For simplification in this demo, we will predict the *next* 2 steps based on the *current* 8 steps,
    and then slide the window if we were doing fully autoregressive. 
    However, standard practice often evaluates on rolling windows.
    Here we will take the full track, and for every window of 8, predict the next 2.
    Then we stitch them together or just plot the series of predictions vs actual.
    
    To make a clean "trajectory plot" like the inference notebook:
    We will take the first 8 steps, predict 2.
    Then take steps 1-9 (using actual data for 9? or predicted?), predict 2.
    Standard "rolling forecast" uses actual history.
    """
    storm_data = data[data['Storm Name'] == storm_name].reset_index(drop=True)
    
    if len(storm_data) < SEQ_LEN + PRED_LEN:
        print(f"Not enough data for {storm_name}")
        return None

    actual_lats = storm_data['Latitude (°N)'].values
    actual_lons = storm_data['Longitude (°E)'].values
    
    lstm_predictions = [] # List of (lat, lon)
    trans_predictions = [] # List of (lat, lon)
    
    # Rolling window prediction
    # We predict for indices [i+8, i+9] based on [i, i+7]
    # We will store the first prediction of the pair to form a continuous track approximation
    
    # To plot a continuous line for "Predicted", we can just chain the 1st step of each prediction.
    
    plot_start_index = SEQ_LEN 
    
    for i in range(len(storm_data) - SEQ_LEN):
        # Window
        window = storm_data.iloc[i:i+SEQ_LEN]
        
        # Prepare input
        try:
            input_tensor = prepare_input(window, sp_data, sst_data, lat_scaler, lon_scaler)
        except Exception as e:
            print(f"Error preparing input for {storm_name} at step {i}: {e}")
            continue
        
        # Last known position
        last_lat = window.iloc[-1]['Latitude (°N)']
        last_lon = window.iloc[-1]['Longitude (°E)']
        
        # LSTM Predict
        l_lat_diff, l_lon_diff = predict_next_steps(lstm_lat, lstm_lon, input_tensor, lat_scaler, lon_scaler, "LSTM")
        # Take 1st step prediction (检查NaN)
        if not (np.isnan(l_lat_diff[0]) or np.isnan(l_lon_diff[0])):
            lstm_predictions.append((last_lat + l_lat_diff[0], last_lon + l_lon_diff[0]))
        else:
            lstm_predictions.append((np.nan, np.nan))
        
        # Transformer Predict
        t_lat_diff, t_lon_diff = predict_next_steps(trans_lat, trans_lon, input_tensor, lat_scaler, lon_scaler, "Transformer")
        # Take 1st step prediction (检查NaN)
        if not (np.isnan(t_lat_diff[0]) or np.isnan(t_lon_diff[0])):
            trans_predictions.append((last_lat + t_lat_diff[0], last_lon + t_lon_diff[0]))
        else:
            trans_predictions.append((np.nan, np.nan))
        
    return actual_lats, actual_lons, lstm_predictions, trans_predictions, plot_start_index

def plot_track_map(storm_name, actual_lats, actual_lons, lstm_preds, trans_preds, start_idx, output_dir):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 8))
    
    # 过滤有效值用于确定范围
    valid_lstm_lats = [p[0] for p in lstm_preds if not (np.isnan(p[0]) or np.isinf(p[0]))]
    valid_lstm_lons = [p[1] for p in lstm_preds if not (np.isnan(p[1]) or np.isinf(p[1]))]
    valid_trans_lats = [p[0] for p in trans_preds if not (np.isnan(p[0]) or np.isinf(p[0]))]
    valid_trans_lons = [p[1] for p in trans_preds if not (np.isnan(p[1]) or np.isinf(p[1]))]
    
    # Determine extent - 使用所有有效值
    all_valid_lats = list(actual_lats) + valid_lstm_lats + valid_trans_lats
    all_valid_lons = list(actual_lons) + valid_lstm_lons + valid_trans_lons
    
    # 检查是否有有效值
    if len(all_valid_lats) == 0 or len(all_valid_lons) == 0:
        print(f"Skipping {storm_name}: No valid coordinate values")
        plt.close()
        return
    
    # 过滤NaN和Inf
    all_valid_lats = [v for v in all_valid_lats if not (np.isnan(v) or np.isinf(v))]
    all_valid_lons = [v for v in all_valid_lons if not (np.isnan(v) or np.isinf(v))]
    
    if len(all_valid_lats) == 0 or len(all_valid_lons) == 0:
        print(f"Skipping {storm_name}: No valid coordinate values after filtering")
        plt.close()
        return
    
    min_lat, max_lat = min(all_valid_lats) - 5, max(all_valid_lats) + 5
    min_lon, max_lon = min(all_valid_lons) - 5, max(all_valid_lons) + 5
    
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    # Add features
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.OCEAN)
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    # Plot Actual
    ax.plot(actual_lons, actual_lats, 'k-o', label='Actual Track', transform=ccrs.PlateCarree(), markersize=4, linewidth=2)
    
    # Plot Predictions
    # LSTM - 过滤NaN和Inf，只绘制有效点
    lstm_lats = [p[0] for p in lstm_preds if not (np.isnan(p[0]) or np.isinf(p[0]) or np.isnan(p[1]) or np.isinf(p[1]))]
    lstm_lons = [p[1] for p in lstm_preds if not (np.isnan(p[0]) or np.isinf(p[0]) or np.isnan(p[1]) or np.isinf(p[1]))]
    if len(lstm_lats) > 0 and len(lstm_lons) > 0 and len(lstm_lats) == len(lstm_lons):
        ax.plot(lstm_lons, lstm_lats, 'b--x', label=f'LSTM Prediction ({len(lstm_lats)} points)', 
                transform=ccrs.PlateCarree(), markersize=4, linewidth=1.5)
    else:
        print(f"Warning: {storm_name} - LSTM predictions have invalid data, skipping plot")
    
    # Transformer - 过滤NaN和Inf，只绘制有效点
    trans_lats = [p[0] for p in trans_preds if not (np.isnan(p[0]) or np.isinf(p[0]) or np.isnan(p[1]) or np.isinf(p[1]))]
    trans_lons = [p[1] for p in trans_preds if not (np.isnan(p[0]) or np.isinf(p[0]) or np.isnan(p[1]) or np.isinf(p[1]))]
    if len(trans_lats) > 0 and len(trans_lons) > 0 and len(trans_lats) == len(trans_lons):
        ax.plot(trans_lons, trans_lats, 'r--^', label=f'Transformer Prediction ({len(trans_lats)} points)', 
                transform=ccrs.PlateCarree(), markersize=4, linewidth=1.5)
    else:
        print(f"Warning: {storm_name} - Transformer predictions have invalid data ({len(trans_lats)} valid points), skipping plot")
        # 即使Transformer预测失败，也显示统计信息
        total_trans = len(trans_preds)
        valid_trans = len(trans_lats)
        print(f"  Transformer: {valid_trans}/{total_trans} valid predictions")
    
    plt.legend(loc='upper right')
    plt.title(f"Typhoon {storm_name} Trajectory Prediction")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{storm_name}_trajectory.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved trajectory plot to {output_file}")
    plt.close()

def main():
    # 1. Load Data and Scalers
    print("Loading data...")
    data = pd.read_csv(csv_path)
    # Ensure numeric
    data['Latitude (°N)'] = pd.to_numeric(data['Latitude (°N)'], errors='coerce')
    data['Longitude (°E)'] = pd.to_numeric(data['Longitude (°E)'], errors='coerce')
    
    with open(sp_json_path, 'r') as f:
        sp_data = json.load(f)
    with open(sst_json_path, 'r') as f:
        sst_data = json.load(f)
        
    # Fit scalers globally (same as training logic)
    lat_scaler = MinMaxScaler()
    lon_scaler = MinMaxScaler()
    
    # We need to compute diffs on the whole dataset to fit scaler correctly
    # (Or load saved scalers if they existed, but here we refit to reproduce)
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
    # We can just create a dummy input to check size or hardcode if known
    # From previous run: 884
    input_dim = 884 
    
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
    except Exception as e:
        print(f"Error loading LSTM: {e}")
        
    try:
        trans_lat = torch.load(os.path.join(model_dir, "lat_cnn_transformer_model.pth"), map_location=device)
        trans_lon = torch.load(os.path.join(model_dir, "lon_cnn_transformer_model.pth"), map_location=device)
        trans_lat.eval()
        trans_lon.eval()
    except Exception as e:
        print(f"Error loading Transformer: {e}")
        return

    # 4. Get all unique typhoons
    unique_storms = data['Storm Name'].unique()
    print(f"Found {len(unique_storms)} unique typhoons")
    
    # 5. Create output directory for trajectory plots
    output_dir = os.path.join(current_dir, 'trajectory_comparison_results')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 6. Process all typhoons
    successful_count = 0
    failed_count = 0
    
    for storm in unique_storms:
        try:
            result = simulate_track(storm, data, sp_data, sst_data, 
                                  lstm_lat, lstm_lon, trans_lat, trans_lon, 
                                  lat_scaler, lon_scaler)
            if result:
                actual_lats, actual_lons, lstm_preds, trans_preds, start_idx = result
                plot_track_map(storm, actual_lats, actual_lons, lstm_preds, trans_preds, start_idx, output_dir)
                successful_count += 1
            else:
                print(f"Skipping {storm}: Not enough data")
                failed_count += 1
        except Exception as e:
            print(f"Error processing {storm}: {e}")
            failed_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful_count} typhoons")
    print(f"Failed/Skipped: {failed_count} typhoons")
    print(f"All trajectory plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
