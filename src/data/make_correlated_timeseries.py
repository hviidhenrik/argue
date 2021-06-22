import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.data.data_utils import *

plt.style.use('seaborn')

if __name__ == "__main__":
    np.random.seed(1234)

    df = make_univariate_timeseries(100000, 1)

    # apply different transforms + noise to obtain correlated time series
    df["x2"] = np.sqrt(abs(df["x1"])) + np.random.normal(0, 0.3, size=df.shape[0])
    df["x3"] = np.exp(df["x1"]) + np.random.normal(0, 0.8, size=df.shape[0])
    df["x4"] = 1 + 0.5 * df["x2"] + np.random.normal(0, 0.3, size=df.shape[0])
    df["x5"] = -2 - df["x1"] * df["x2"] + np.random.normal(0, 0.7, size=df.shape[0])
    df["x6"] = df["x3"] * df["x4"] + np.random.normal(0, 0.7, size=df.shape[0])

    df.plot()
    plt.show()

    print(df.corr())

    sns.heatmap(df.corr().round(2), annot=True, center=0, cmap="bwr")
    plt.show()

    path = get_data_path() / "simulated_data" / "data_simulated_timeseries.csv"
    save_local_data(df, filename=path)