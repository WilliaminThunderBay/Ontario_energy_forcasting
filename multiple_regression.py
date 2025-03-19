import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# ---------- 数据加载与特征提取 ----------
# 假设 merged_data.csv 中已经合并了电力需求和气象数据，并包含 datetime 列
df = pd.read_csv("merged_data.csv", parse_dates=["datetime"])
df.set_index("datetime", inplace=True)

# 现有气象特征
meteo_features = ["Temp (°C)", "Dew Point Temp (°C)", "Rel Hum (%)", "Wind Spd (km/h)", "Stn Press (kPa)"]

# 增加时间特征
df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek  # Monday=0, Sunday=6
df["month"] = df.index.month

time_features = ["hour", "day_of_week", "month"]

# 合并所有特征
features = meteo_features + time_features
target = "hourly_demand"

# 删除缺失值
df = df.dropna(subset=features + [target])

# 对目标变量进行对数变换（平滑大数值波动）
df["log_demand"] = np.log1p(df[target])

# 准备特征矩阵 X 与目标向量 y (用对数目标)
X = df[features]
y = df["log_demand"]

# ---------- 使用时间序列交叉验证 ----------
tscv = TimeSeriesSplit(n_splits=5)
results = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # 构建流水线：标准化 -> 多项式特征扩展（degree=2） -> Ridge 回归
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", Ridge(alpha=1.0, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # 反变换：预测结果和实际值转换回原始需求
    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred)
    
    mae = mean_absolute_error(y_test_real, y_pred_real)
    mse = mean_squared_error(y_test_real, y_pred_real)
    r2  = r2_score(y_test_real, y_pred_real)
    
    # ---------- 分类指标 ----------
    # 这里将回归问题转换为二分类问题：以训练集原始需求中位数为阈值判断高/低需求
    threshold = np.median(np.expm1(y_train))
    y_test_class = (y_test_real >= threshold).astype(int)
    y_pred_class = (y_pred_real >= threshold).astype(int)
    
    acc  = accuracy_score(y_test_class, y_pred_class)
    rec  = recall_score(y_test_class, y_pred_class)
    prec = precision_score(y_test_class, y_pred_class)
    f1   = f1_score(y_test_class, y_pred_class)
    cm   = confusion_matrix(y_test_class, y_pred_class)
    
    results.append({
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "accuracy": acc,
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "confusion_matrix": cm
    })

# 计算所有折交叉验证的平均指标
avg_mae = np.mean([res["mae"] for res in results])
avg_mse = np.mean([res["mse"] for res in results])
avg_r2  = np.mean([res["r2"] for res in results])
avg_acc = np.mean([res["accuracy"] for res in results])
avg_rec = np.mean([res["recall"] for res in results])
avg_prec = np.mean([res["precision"] for res in results])
avg_f1 = np.mean([res["f1"] for res in results])

print("TimeSeries CV - Polynomial+Ridge Regression:")
print(f"  Average MAE: {avg_mae:.2f}")
print(f"  Average MSE: {avg_mse:.2f}")
print(f"  Average R²:  {avg_r2:.2f}")
print(f"  Average Accuracy:  {avg_acc:.2f}")
print(f"  Average Recall:    {avg_rec:.2f}")
print(f"  Average Precision: {avg_prec:.2f}")
print(f"  Average F1 Score:  {avg_f1:.2f}")
