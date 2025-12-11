import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------
# 1. 讀取資料
# ---------------------------
df = pd.read_csv("wtbdata_245days.csv")

# ---------------------------
# 2. 資料清洗
# ---------------------------
# 先推斷數據類型,然後只對數值列進行插值
df = df.infer_objects(copy=False)
df_numeric = df.select_dtypes(include=[np.number])
df_non_numeric = df.select_dtypes(exclude=[np.number])
df = pd.concat([df_numeric.interpolate(), df_non_numeric], axis=1)

df["Patv"] = df["Patv"].clip(lower=0)

df = df[(df["Wdir"] >= -180) & (df["Wdir"] <= 180)]
df = df[(df["Ndir"] >= -720) & (df["Ndir"] <= 720)]
df = df[df[["Pab1","Pab2","Pab3"]].max(axis=1) <= 89]

# ---------------------------
# 3. 修正時間格式：Day + Tmstamp 合併成 datetime
# ---------------------------

# 建立一個起始日期
start_date = pd.to_datetime("2020-01-01")

# Day=1 → 2020-01-01
df["Date"] = start_date + pd.to_timedelta(df["Day"] - 1, unit="D")

# Tmstamp 是 HH:MM，所以我們用 '%H:%M' 解析
df["Time"] = pd.to_datetime(df["Tmstamp"], format="%H:%M").dt.time

# 合併 Date + Time
df["Datetime"] = df.apply(lambda row: pd.Timestamp.combine(row["Date"], row["Time"]), axis=1)

# ---------------------------
# 4. 特徵工程
# ---------------------------
df["hour"] = df["Datetime"].dt.hour
df["weekday"] = df["Datetime"].dt.weekday

# 時間序列滯後特徵
for lag in [1,2,3]:
    df[f"Wspd_lag{lag}"] = df["Wspd"].shift(lag)
    df[f"Patv_lag{lag}"] = df["Patv"].shift(lag)

df = df.dropna()

# 特徵選取
features = [
    "Wspd","Wdir","Etmp","Itmp","Ndir",
    "Pab1","Pab2","Pab3",
    "hour","weekday",
    "Wspd_lag1","Wspd_lag2","Patv_lag1","Patv_lag2"
]

X = df[features]
y = df["Patv"]

# ---------------------------
# 5. 訓練 RandomForest 預測模型
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 減少樹的數量和深度以加快訓練速度
model = RandomForestRegressor(
    n_estimators=100,  # 從 200 減少到 100
    max_depth=15,      # 限制樹的深度
    random_state=42,
    n_jobs=-1          # 使用所有 CPU 核心
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# ---------------------------
# 6. Power Curve 圖
# ---------------------------
plt.scatter(df["Wspd"], df["Patv"], s=1, alpha=0.3)
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Active Power (kW)")
plt.title("Wind Speed vs Power Curve")
plt.show()

# ---------------------------
# 7. 異常偵測
# ---------------------------
iso = IsolationForest(contamination=0.02)
df["anomaly"] = iso.fit_predict(df[["Wspd","Wdir","Patv","Pab1","Itmp"]])

anomaly_df = df[df["anomaly"] == -1]

plt.figure(figsize=(8,5))
plt.scatter(df["Wspd"], df["Patv"], s=1, alpha=0.2, label='Normal')
plt.scatter(anomaly_df["Wspd"], anomaly_df["Patv"], s=3, color="red", label='Anomaly')
plt.xlabel("Wind Speed")
plt.ylabel("Power")
plt.title("Power Curve with Anomaly Detection")
plt.legend()
plt.show()
