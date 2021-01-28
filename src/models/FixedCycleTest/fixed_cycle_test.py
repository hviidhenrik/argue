from typing import Tuple

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from src.config.definitions import *
from src.utilities.utility_functions import *
from src.data.data_utils import *
from src.models.FixedCycleTest.imputer_class import *

plt.style.use("seaborn")

is_debugging = True
# is_debugging = False
use_saved_imputer = True
# use_saved_imputer = False

filename = f"data_cwp_pump_10_{get_dataset_purpose_as_str(is_debugging)}_imputed.csv"
df = get_local_data(get_data_path() / "ssv_cooling_water_pump" / filename)

# investigate stationarity given a "driving" feature in a fixed interval
feature_to_filter_on = "rotation"

plot_column_as_timeseries(df[feature_to_filter_on])
plt.show()

sns.histplot(df[feature_to_filter_on])
plt.show()

feature_min, feature_max = find_bin_with_most_values(df[feature_to_filter_on], interval_size=0.5)

# df_feature = df[(df[feature_to_filter_on] >= feature_min) & (df[feature_to_filter_on] < feature_max)]
# for column in df_feature.columns:
#     if "vibr" in column:
#         plot_column_as_timeseries(df_feature[column],
#                                   title=f"Feature {column} for {feature_to_filter_on} in ({feature_min}, {feature_max})",
#                                   save_path=get_fixed_cycle_figures_path() / f"{column}_interval_fixed_{feature_to_filter_on}")
#         plt.show()

edge_interval_list = find_all_nonempty_bins(df[feature_to_filter_on], interval_size=1, required_bin_size=50)
feature_of_interest = "vibr_motor_y"

df[feature_of_interest].plot()
plt.show()


for interval in edge_interval_list:
    feature_min, feature_max = interval
    df_feature = df[(df[feature_to_filter_on] >= feature_min) & (df[feature_to_filter_on] < feature_max)]
    filename = f"{feature_to_filter_on}_in_{feature_min}_to_{feature_max}_for_{feature_of_interest}.png"
    plot_column_as_timeseries(df_feature[feature_of_interest],
                              title=f"Feature {feature_of_interest} for {feature_to_filter_on} in "
                                    f"({feature_min}, {feature_max})",
                              save_path=get_fixed_cycle_figures_path() / "fixed_rotation" / filename)
    plt.show()

