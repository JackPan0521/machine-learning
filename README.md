# 風力發電預測專案

## 專案簡介

使用機器學習模型預測風力發電機的發電量，並進行異常檢測。

## 資料集

- `wtbdata_245days.csv`: 245 天的風力發電機運行資料

## 功能

- 資料清洗與特徵工程
- 使用 RandomForest 進行發電量預測
- 使用 IsolationForest 進行異常檢測
- 視覺化功率曲線

## 安裝與執行

### 建立虛擬環境

```bash
python -m venv venv
venv\Scripts\activate
```

### 安裝套件

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 執行程式

```bash
python prediction.py
```

## 模型效能

- 使用 RandomForest Regressor
- 特徵包含: 風速、風向、溫度、時間特徵、滯後特徵等
- 評估指標: MAE, RMSE

## 技術堆疊

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
