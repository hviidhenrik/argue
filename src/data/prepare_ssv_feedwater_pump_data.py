"""
This file prepares the test data from SSV Feedwater pump 30 for testing ARGUE and related models.
Two leaks take place: one in 2017 and one in end 2020. The leaks represent the phase 2 data.
Healthy phase 1 data is selected as the fault free data up to each leak.
"""
import os
from src.utils.misc import *
from src.data.utils import *


def put_sample_and_faulty_cols_first(df):
    cols = df.columns.values.tolist()
    assert "sample" in cols, "\"sample\" column not in the dataframe"
    assert "faulty" in cols, "\"faulty\" column not in the dataframe"
    cols = cols[-2:] + cols[:-2]
    return df[cols]


def make_sample_and_faulty_cols(df):
    df_copy = df.copy()
    df_copy["sample"] = [i for i in range(1, df_copy.shape[0] + 1)]
    df_copy["faulty"] = [0 for _ in range(1, df_copy.shape[0] + 1)]
    df_copy = put_sample_and_faulty_cols_first(df_copy)
    return df_copy


# Prepare dataset for the leak in end of 2020 ----------------------------------

path = get_data_path() / "ssv_feedwater_pump"
df_pump30_large = get_local_data(path / f"data_pump_30_large_cleaned.csv")
df_pump30_large = df_pump30_large.drop(columns=["effect_pump_10_MW"])

# select the healthy phase 1 data
df_phase1 = pd.concat([df_pump30_large.loc[:"2019-12-01 23:59:59"],
                       df_pump30_large.loc["2020-02-30 23:59:59": "2020-09-14 23:59:59"]])

# make columns with sample number and binary fault status as the first columns in the df
df_phase1 = make_sample_and_faulty_cols(df_phase1)

# the pump has a leak starting very secretly around the 14/11-20 at 8 pm. Clearly visible after 15/12-20.
df_phase2 = df_pump30_large.loc["2020-09-15":"2021-01-08"]
df_phase2 = make_sample_and_faulty_cols(df_phase2)
df_phase2.loc["2020-11-14 20:00:00":"2021-01-08", "faulty"] = 1

# save it to csv
save_local_data(df_phase1, path / "data_pump30_leak20_phase1.csv", index="timelocal")
save_local_data(df_phase2, path / "data_pump30_leak20_phase2.csv", index="timelocal")


# Prepare dataset for the leak in Fall of 2017 ----------------------------------

# need to find the dataset for 2017 first... somewhere on my harddrive. Yikes...