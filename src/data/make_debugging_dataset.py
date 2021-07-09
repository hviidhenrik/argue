"""
This script takes a large dataset and makes and saves a smaller one for faster testing and debugging.
The subset selection is based on a random shuffling, so it should be representative of the full dataset.
"""
from src.config.definitions import *
from src.data.utils import *

if __name__ == "__main__":
    path = get_data_path() / "covtype"
    df_large = pd.read_csv(path / "covtype.csv")
    make_and_save_debugging_dataset(df_large, size=30000,
                                    filename=path / "covtype_small.csv")
