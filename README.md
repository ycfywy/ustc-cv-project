# Typhoon-Trajectory-Prediction
This repository is the final project for the course STATS402
## Environment Setting
```
conda create -n typoon python=3.10
conda activate typoon
pip install -r requirements.txt
```
## Dataset preparation
- CMA Tropical Cyclone Best Track Dataset avaliable at: https://tcdata.typhoon.org.cn/en/zjljsjj.html
- ERA5 dataset available at: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview
- Typhoon-Trajectory-Prediction-Based-On-CNN-LSTM-model\data_preprocess\data_preprocess.ipynb has already provided a solution to access ERA5 dataset with the CMA dataset and the data preprocessing. You only need to replace the key to your CDSAPI key
```
KEY = 'your key'
```
## Training
```
python train\train.py
```

## Inference and Visualization
- Change the '/train/best_track_records_test.csv', paste the coordinates and the datetime of your typhoon to this csv file.
- Change the typhoon name to the typhoon you want to predict.
```
typhoon_name='your typhoon'
```
- Run the code in "Typhoon-Trajectory-Prediction-Based-On-CNN-LSTM-model\inference\inference.ipynb"
