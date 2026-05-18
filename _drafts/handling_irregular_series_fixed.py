import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import signalplot
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as _f:
        import yaml as _yaml

        return _yaml.safe_load(_f) or {}


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
np.random.seed(config.get("data", {}).get("seed", 42))

signalplot.apply(font_family="serif")


def chrono_split(df: pd.DataFrame, time_col: str, test_size: float = 0.33):
    df = df.sort_values(time_col).copy()
    split_idx = int(np.floor((1 - test_size) * len(df)))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def build_irregular_df():
    df = pd.DataFrame(
        {
            "timestamp": [
                "2025-01-01 10:00:00",
                "2025-01-01 10:15:00",
                "2025-01-01 10:45:00",
                "2025-01-01 11:50:00",
                "2025-01-01 12:00:00",
                "2025-01-01 13:15:00",
            ],
            "value": [10, 15, 20, 18, 22, 19],
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp").sort_index()


def plot_irregular_and_resampled(df: pd.DataFrame, freq: str = "10min", plot: bool = False):
    regular_df = df.resample(freq).ffill()
    if plot:
        plt.figure(figsize=tuple(config.get("output", {}).get("figsize", [8, 4])))
        plt.scatter(df.index, df["value"], label="Irregular")
        plt.plot(regular_df.index, regular_df["value"], label="Resampled ffill")
        plt.title("Irregular vs Resampled (ffill)")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        signalplot.save("irregular_vs_resampled.png")


def plot_causal_resample_split(df: pd.DataFrame, freq: str = "10min", plot: bool = False):
    df_train, df_test = chrono_split(df.reset_index(), "timestamp", test_size=0.33)
    df_train = df_train.set_index("timestamp").sort_index()
    df_test = df_test.set_index("timestamp").sort_index()
    train_reg = df_train.resample(freq).ffill()
    init = train_reg.iloc[[-1]]
    test_reg = df_test.resample(freq).asfreq()
    if len(test_reg) > 0:
        test_reg.loc[test_reg.index[0], "value"] = init["value"].iloc[0]
        test_reg = test_reg.ffill()

    regular_df = pd.concat([train_reg, test_reg])
    if plot:
        plt.figure(figsize=tuple(config.get("output", {}).get("figsize", [8, 4])))
        plt.scatter(df.index, df["value"], label="Irregular")
        plt.plot(regular_df.index, regular_df["value"], label="Causal ffill across split")
        plt.title("Causal Resampling Across Chrono Split")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        signalplot.save("causal_resample_split.png")


def plot_causal_interpolation_split(df: pd.DataFrame, freq: str = "10min", plot: bool = False):
    df_train, df_test = chrono_split(df.reset_index(), "timestamp", test_size=0.33)
    df_train = df_train.set_index("timestamp").sort_index()
    df_test = df_test.set_index("timestamp").sort_index()
    train_interp = df_train.resample(freq).interpolate(method="time")
    test_interp = df_test.resample(freq).asfreq()
    if len(test_interp) > 0:
        test_interp.loc[test_interp.index[0], "value"] = train_interp.iloc[-1]["value"]
        test_interp = test_interp.ffill()

    df_model = pd.concat([train_interp, test_interp])
    if plot:
        plt.figure(figsize=tuple(config.get("output", {}).get("figsize", [8, 4])))
        plt.scatter(df.index, df["value"], label="Irregular")
        plt.plot(
            df_model.index,
            df_model["value"],
            label="Causal interpolation (train), ffill (test)",
        )
        plt.title("Causal Interpolation Across Chrono Split")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        signalplot.save("causal_interpolation_split.png")


def plot_gp_predictions(df: pd.DataFrame, plot: bool = False):
    t_ns = df.index.asi8
    t = (t_ns - t_ns.min()) / 6e10
    vals = df["value"].values
    df_gp = pd.DataFrame({"t": t, "y": vals})
    df_train, df_test = chrono_split(df_gp, "t", test_size=0.33)
    X_train = df_train[["t"]].values
    y_train = df_train["y"].values
    X_test = df_test[["t"]].values
    gp = GaussianProcessRegressor(
        kernel=RBF(length_scale=30.0), alpha=1e-2, normalize_y=True, random_state=42
    )
    gp.fit(X_train, y_train)
    y_pred, y_std = gp.predict(X_test, return_std=True)
    if plot:
        plt.figure(figsize=tuple(config.get("output", {}).get("figsize", [8, 4])))
        plt.scatter(df_gp["t"], df_gp["y"], label="Observations")
        plt.plot(X_test.ravel(), y_pred, label="GP mean (test)")
        plt.fill_between(
            X_test.ravel(),
            y_pred - 2 * y_std,
            y_pred + 2 * y_std,
            alpha=0.2,
            label="95% CI",
        )
        plt.title("Gaussian Process on Irregular Times (Train-only fit)")
        plt.xlabel("Minutes")
        plt.ylabel("Value")
        plt.legend()
        signalplot.save("gp_predictions.png")


if __name__ == "__main__":
    df = build_irregular_df()
    plot_irregular_and_resampled(df)
    plot_causal_resample_split(df)
    plot_causal_interpolation_split(df)
    plot_gp_predictions(df)
    logger.info(
        "Images written: irregular_vs_resampled.png, causal_resample_split.png, causal_interpolation_split.png, gp_predictions.png"
    )
