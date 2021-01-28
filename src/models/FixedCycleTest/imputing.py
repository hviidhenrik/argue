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

# is_debugging = True
is_debugging = False
use_saved_imputer = True
# use_saved_imputer = False

# load data
filename = get_pump_data_path() / f"data_cwp_pump_10_{get_dataset_purpose_as_str(is_debugging)}.csv"
df = get_local_data(filename)

# inspect missing values
plot_missing_values_bar_chart(df, save_path=get_fixed_cycle_figures_path() / "missingvalues.png")
plt.show()

plot_missing_values_heatmap(df, save_path=get_fixed_cycle_figures_path() / "missingvalues_heatmap.png")
plt.show()

imputer = FFNImputer(["flush_indicator"],
                     hidden_layers=[14, 12, 10, 8, 6, 4, 2],
                     activation_functions="elu",
                     epochs=20,
                     loss="binary_crossentropy",
                     metrics="accuracy")
if use_saved_imputer:
    imputer.load_model(get_model_archive_path() / "flush_indicator_impute_model")
    df = imputer.impute(df)
else:
    imputer.fit(df, class_weight={0: 4, 1: 65})
    df = imputer.impute(df)
    imputer.save_model(get_model_archive_path() / "flush_indicator_impute_model")

df["flush_indicator"] = np.round(df["flush_indicator"])

plot_missing_values_heatmap(df, save_path=get_fixed_cycle_figures_path() / "missingvalues_imputed_heatmap.png")
plt.show()

# look at fixed cycle down here
df = df.dropna()
print(df.shape)

filename = f"data_cwp_pump_10_{get_dataset_purpose_as_str(is_debugging)}_imputed.csv"
save_local_data(df, get_data_path() / "ssv_cooling_water_pump" / filename)