from src.data.utils import *

test_data_file_debugging = get_test_data_path() / "data_cwp_pump_10_debugging.csv"
test_data_file_real = get_test_data_path() / "data_cwp_pump_10_real.csv"

pump_data_file_debugging = get_cwp_data_path() / "data_cwp_pump_10_debugging.csv"
pump_data_file_real = get_cwp_data_path() / "data_cwp_pump_10_real.csv"


def test_get_local_data():
    df = get_local_data(filename=test_data_file_debugging)
    assert isinstance(df, DataFrame)
    assert not df.empty


def test_save_local_data():
    df_original = get_local_data(filename=test_data_file_debugging)
    save_local_data(df_original, filename=test_data_file_debugging)
    df_loaded = get_local_data(filename=test_data_file_debugging)
    assert isinstance(df_loaded, DataFrame)
    assert df_loaded.shape == df_original.shape


def test_make_and_save_debugging_dataset():
    df_large = get_local_data(filename=test_data_file_real)
    make_and_save_debugging_dataset(df_large, size=df_large.shape[0]-10,
                                    filename=test_data_file_debugging)
    df_debugging = get_local_data(filename=test_data_file_debugging)
    assert isinstance(df_debugging, DataFrame)
    assert not df_debugging.empty
    assert df_debugging.shape[0] == df_large.shape[0]-10
