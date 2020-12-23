import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from src.models.AAE.aae_utils import *
from src.models.AAE.definitions import *
from src.models.AAE.aae_class import *
import time


if __name__ == "__main__":
    import keras
    pd.set_option('display.max_columns', None)
    # Loading data
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Flatten the dataset
    x_train = x_train.reshape((-1, 28 * 28))
    x_test = x_test.reshape((-1, 28 * 28))

    aae = AdversarialAutoencoder(latent_layer_dimension=2,
                                 encoder_hidden_layers=[600, 400, 200, 50, 20],
                                 decoder_hidden_layers=[20, 50, 200, 400, 600],
                                 discriminator_hidden_layers=[600, 400, 200, 50, 20])
    aae.fit(x_train, dropout_fraction=0.01, epochs=20, batch_size=1024, verbose=2)
    aae.save(model_folder="mnist_aae")
    aae.plot_latent_space(x_test, coloring_column=pd.DataFrame({"class": y_test}))