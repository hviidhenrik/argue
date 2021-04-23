"""
This script takes a large dataset and makes and saves a smaller one for faster testing and debugging.
The subset selection is based on a random shuffling, so it should be representative of the full dataset.
"""
from src.data.data_utils import *
from src.config.definitions import *

if __name__ == "__main__":
    path = get_data_path() / "covtype"
    df_original = pd.read_csv(path / "covtype.csv")
    df_normal_A = df_original[(df_original["Cover_Type"] < 5)]
    df_anomaly_A = df_original[(df_original["Cover_Type"] > 4)]

    df_normal_B = df_original[(df_original["Cover_Type"] > 3)]
    df_anomaly_B = df_original[(df_original["Cover_Type"] < 4)]

    make_and_save_debugging_dataset(df_normal_A, size=30000,
                                    filename=path / "data_covtype_normal_A_small.csv",
                                    index=False)
    make_and_save_debugging_dataset(df_anomaly_A, size=3000,
                                    filename=path / "data_covtype_anomaly_A_small.csv",
                                    index=False)
    make_and_save_debugging_dataset(df_normal_B, size=30000,
                                    filename=path / "data_covtype_normal_B_small.csv",
                                    index=False)
    make_and_save_debugging_dataset(df_anomaly_B, size=3000,
                                    filename=path / "data_covtype_anomaly_B_small.csv",
                                    index=False)
