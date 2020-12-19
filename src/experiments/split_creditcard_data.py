import pandas as pd
from sklearn.model_selection import train_test_split

data_path = "..\\..\\data"

if __name__ == "__main__":
    df = pd.read_csv(data_path + "\\creditcard.csv")
    df = df.drop("Time", axis=1)
    df_anomalies = df[df["Class"] == 1].reset_index(drop=True)
    df_nominal = df[df["Class"] == 0].reset_index(drop=True)

    df_small, _ = train_test_split(df_nominal, train_size=0.01)

    df_small.to_csv(data_path + "\\dataset_nominal_small.csv", index=False)
    df_nominal.to_csv(data_path + "\\dataset_nominal_full.csv", index=False)
    df_anomalies.to_csv(data_path + "\\dataset_anomalies.csv", index=False)

    print(f"Data splitted and saved locally in: {data_path}")

