# Typhoon-Trajectory-Prediction
## Environment Setting
```
conda create -n typoon python=3.10
conda activate typoon
pip install -r requirements.txt
```
## Dataset preparation
- CMA Tropical Cyclone Best Track Dataset available at: https://tcdata.typhoon.org.cn/en/zjljsjj.html
- ERA5 dataset available at: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview
- Typhoon-Trajectory-Prediction-Based-On-CNN-LSTM-model\data_preprocess\data_preprocess.ipynb has already provided a solution to access the ERA5 dataset with the CMA dataset and the data preprocessing. You only need to replace the key to your CDSAPI key
```
KEY = 'your key'
```
- For detailed prepocess code, please see data_preprocess/data_preprocess.ipynb.
## Training
- For the CNN-LSTM model plsease refer to some functions defined in train/train.py.
- For CNN-Transformer model training details, please refer to train/CNNTransformer_Architecture.md

## Inference and Visualization
- Please refer to  inference/inference.ipynb
- Please refer to train/test_trajectory_plot.py

## Results
- Please refer to train/MSE_Visualization_Guide.md and train/Trajectory_Prediction_Guide.md.

