"""
This script takes a large dataset and makes and saves a smaller one for faster testing and debugging.
The subset selection is based on a random shuffling, so it should be representative of the full dataset.
"""
from src.data.utils import *
from src.config.definitions import *

if __name__ == "__main__":
    path = get_data_path() / "ssv_feedwater_pump"

    df_large = pd.read_csv(path / "data_pump_20_ARGUE-test_large.csv")
    make_and_save_debugging_dataset(df_large, size=30000,
                                    filename=path / "data_pump_20_ARGUE-test_small.csv")

    df_large = pd.read_csv(path / "data_pump_30_ARGUE-test_large.csv")
    make_and_save_debugging_dataset(df_large, size=30000,
                                    filename=path / "data_pump_30_ARGUE-test_small.csv")
