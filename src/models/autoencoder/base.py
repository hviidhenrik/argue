import matplotlib.pyplot as plt
import numpy as np

from pandas import DataFrame
import seaborn as sns


class AutoencoderMixin:
    """
    This class stores autoencoder specific functions that are useful when using this model type
    """

    @staticmethod
    def plot_learning(history):
        """
        Plots learning curve given a history object from a trained Keras model

        :param history: history object from a Keras model
        """
        fig = plt.plot(history["loss"])
        if len(history.keys()):
            plt.plot(history["val_loss"])
            plt.legend(["train", "validation"], loc="upper right")
        else:
            plt.legend(["train"], loc="upper right")
        plt.title("Learning curve")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        return fig

    @staticmethod
    def plot_latent_space(
        latent_activations: np.array = None, latent_reference_col: DataFrame = None
    ):
        """
        Plots latent activations from an autoencoder network. Basically just a scatter plot
        with color gradient for reference with some other feature from the dataset, and histograms
        on the axes to show the distribution of points in the latent space.

        :param latent_activations: a numpy array of dimension N_rows x 2 with the latent activations
        :param latent_reference_col: optional pandas Dataframe with a reference feature for color coding
        :return: matplotlib.pyplot figure object
        """
        assert latent_activations is not None, "No latent activations provided!"

        cmap = sns.cubehelix_palette(gamma=0.7, reverse=True, as_cmap=True)
        fig, ax = plt.subplots(
            2,
            2,
            figsize=(8, 6),
            gridspec_kw={
                "hspace": 0,
                "wspace": 0,
                "width_ratios": [5, 1],
                "height_ratios": [1, 5],
            },
        )
        ax[0, 0].axis("off")
        ax[0, 1].axis("off")
        ax[1, 1].axis("off")

        if latent_reference_col is None:
            fig = ax[1, 0].scatter(x=latent_activations[:, 0], y=latent_activations[:, 1], s=10)
        else:
            points = ax[1, 0].scatter(
                x=latent_activations[:, 0],
                y=latent_activations[:, 1],
                c=np.array(latent_reference_col),
                cmap=cmap,
            )
            cbar = fig.colorbar(points)
            cbar.ax.get_yaxis().labelpad = 20
            cbar.ax.set_ylabel(latent_reference_col.name, rotation=270)

        sns.histplot(x=latent_activations[:, 0], ax=ax[0, 0], color="LightBlue")
        sns.histplot(y=latent_activations[:, 1], ax=ax[1, 1], color="LightBlue")
        plt.suptitle("Latent layer", fontsize=18)
        ax[1, 0].set_xlabel("Latent unit 1")
        ax[1, 0].set_ylabel("Latent unit 2")
        return fig
