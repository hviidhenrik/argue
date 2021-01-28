"""
This script takes a large dataset and makes and saves a smaller one for faster testing and debugging.
The subset selection is based on a random shuffling, so it should be representative of the full dataset.
"""
from src.data.data_utils import *
from src.config.definitions import *

if __name__ == "__main__":
    df_large = get_local_data(get_pump_data_path() / "data_cwp_pump_10_real.csv")
    make_and_save_debugging_dataset(df_large, size=30000,
                                    filename=get_pump_data_path() / "data_cwp_pump_10_debugging.csv")
