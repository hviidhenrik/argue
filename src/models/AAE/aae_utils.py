import pandas as pd
from pandas import DataFrame
from src.models.AAE.definitions import *


def get_local_pump_data(small_dataset: bool = True,
                        station: str = None,
                        component: str = None,
                        pump_number: str = None):
    assert None not in [small_dataset, station, component, pump_number], "One or more arguments unspecified"
    dataset_size = "small" if small_dataset else "large"
    filename = f"{DATA_PATH}\\{station.upper()}_{component.upper()}\\data_pump_{pump_number}_{dataset_size}.csv"
    df = pd.read_csv(filename, index_col="timelocal")
    print(f'Local data loaded from file: "{filename}"')
    return df


def save_local_pump_data(df_to_save: DataFrame = None,
                         small_dataset: bool = None,
                         station: str = None,
                         component: str = None,
                         pump_number: str = None):
    assert None not in [df_to_save, small_dataset, station, component, pump_number], "One or more arguments unspecified"
    dataset_size = "small" if small_dataset else "large"
    filename = f"{DATA_PATH}\\{station.upper()}_{component.upper()}\\data_pump_{pump_number}_{dataset_size}.csv"
    df_to_save.to_csv(filename)
    print(f"Data saved locally as {filename}")