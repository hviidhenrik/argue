import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.AnoGen.anomaly_generator import *
from src.models.AnoGen.utility_functions import *
from src.models.AnoGen.utils_test_creditcard import *
from sklearn.metrics import precision_recall_curve, average_precision_score


data_path = "..\\..\\data\\creditcard_fraud"

if __name__ == "__main__":
    df_nominal = pd.read_csv(data_path + "\\dataset_nominal_small.csv")
    df_anomalies = pd.read_csv(data_path + "\\dataset_anomalies.csv")
    # df_nominal = pd.read_csv(data_path + "\\dataset_nominal_full.csv")
    df_nominal = df_nominal.drop(["Class"], axis=1)
    df_anomalies = df_anomalies.drop(["Class"], axis=1)

    credit_card_data = DataProvider()
    x_train, x_val, x_test = credit_card_data.train_val_test_split(df_nominal, 0.8, 0.1, 0.1)
    x_train_scaled, x_val_scaled, x_test_scaled = credit_card_data.scale()

    # train anomaly detection algorithm
    detector = AnomalyDetectorAutoencoder(first_hidden_layer_dimension=12, latent_space_dimension=2)
    # load_model = True
    load_model = False
    if load_model:
        detector.load()
    else:
        detector.fit(x_train,
                     x_val,
                     plot_learning_curve=True)
        detector.save()

    df_test = concat_testset_and_anomalies_rowwise(x_test, df_anomalies)
    y_true = make_binary_labels(x_test.shape[0], 0) + make_binary_labels(df_anomalies.shape[0], 1)

    mse = detector.predict_new_and_return_mse(df_test)
    y_scores = MinMaxScaler().fit_transform(mse)
    precision_scores, recall_scores, thresholds = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    plot_precision_recall_curve(recall_scores, precision_scores)
    print(f"AVerage precision score over thresholds: {avg_precision:.3f}")


    """
    Now synthesize anomalies from AnoGen and use for training the autoencoder or a supervised algorithm 
    """