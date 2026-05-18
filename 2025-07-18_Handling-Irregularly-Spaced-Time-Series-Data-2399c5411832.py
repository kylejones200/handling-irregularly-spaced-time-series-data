# Description: Short example for Handling Irregularly Spaced Time Series Data.

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import signalplot
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def chrono_split(df, time_col, test_size=0.2):
    df = df.sort_values(time_col).copy()
    split_idx = int(np.floor((1 - test_size) * len(df)))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


# Build example dataframe
df = pd.DataFrame(
    {
        "timestamp": [
            "2025-01-01 10:00:00",
            "2025-01-01 10:15:00",
            "2025-01-01 10:45:00",
        ],
        "value": [10, 15, 20],
    }
)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp").sort_index()

# Chronological split on index
df_train, df_test = chrono_split(df.reset_index(), "timestamp", test_size=0.33)
df_train = df_train.set_index("timestamp").sort_index()
df_test = df_test.set_index("timestamp").sort_index()

# Resample separately
train_reg = df_train.resample("10min").ffill()  # causal within train
# Carry last known train value into test as initial state (causal)
init = train_reg.iloc[[-1]]
test_reg = df_test.resample("10min").asfreq()
test_reg.loc[test_reg.index[0], "value"] = init["value"].iloc[0]
test_reg = test_reg.ffill()

regular_df = train_reg.append(test_reg)
logger.info(regular_df.head(10))

# Interpolate only inside TRAIN
train_interp = df_train.resample("10min").interpolate(method="time")

# For TEST, avoid symmetric interpolation; use forward-fill (causal)
test_interp = df_test.resample("10min").asfreq()
test_interp.loc[test_interp.index[0], "value"] = train_interp.iloc[-1]["value"]
test_interp = test_interp.ffill()

df_model = train_interp.append(test_interp)

# Irregular times (in minutes)
ts = np.array([0, 5, 15, 60, 75, 120], dtype=float)
vals = np.array([10, 15, 20, 18, 22, 19], dtype=float)
df_gp = pd.DataFrame({"t": ts, "y": vals})

# Chronological split
df_gp_train, df_gp_test = chrono_split(df_gp, "t", test_size=0.33)

X_train = df_gp_train[["t"]].values
y_train = df_gp_train["y"].values
X_test = df_gp_test[["t"]].values  # future times

gp = GaussianProcessRegressor(
    kernel=RBF(length_scale=10.0), alpha=1e-6, normalize_y=True, random_state=42
)
gp.fit(X_train, y_train)

y_pred, y_std = gp.predict(X_test, return_std=True)
logger.info("GP predictions:", y_pred)


# Build causal windows aware of time gaps
def make_windows(df, time_col, value_col, window=30):
    df = df.sort_values(time_col).copy()
    t = df[time_col].astype("int64").values  # ns
    t = (t - t.min()) / 1e9  # seconds from start
    v = df[value_col].values
    dt = np.diff(t, prepend=t[0])  # delta_t, 0 for first
    X, y = [], []
    for i in range(len(v) - window):
        X.append(np.c_[v[i : i + window], dt[i : i + window]])  # value and delta_t
        y.append(v[i + window])
    return np.array(X), np.array(y)


# Fit scaler on TRAIN ONLY if you scale features
# from sklearn.preprocessing import StandardScaler
# scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
# X_train = scaler.transform(...), X_test = scaler.transform(...)

# Example usage
plt.scatter(df.index, df["value"], label="Irregular Data")
plt.plot(regular_df.index, regular_df["value"], label="Resampled Data")
plt.legend()
plt.title("Visualization of Irregular Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
signalplot.save("irregular_time_series_plot.png")
