import pickle
from pathlib import Path
from typing import Dict, Optional, Union

import tensorflow as tf
from tensorflow.python.keras.engine.functional import Functional
from tqdm import tqdm

from argue.utils.misc import vprint
from argue.utils.model import Network


class BaseModel:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self.hyperparameters = None

    def save(self, save_path: Union[Path, str] = None, model_name: str = None):
        assert save_path is not None, "No path specified to save model to"
        vprint(self.verbose, f"\nSaving model to: {save_path}\n")
        save_path = Path(save_path)

        def _save_models_in_dict(model_dict: Dict):
            for name, model in model_dict.items():
                model.save(save_path / name)

        # iterate over all the different item types in the self dictionary to be saved
        non_model_attributes_dict = {}
        with tqdm(total=len(vars(self))) as pbar:
            for name, attribute in vars(self).items():
                if isinstance(attribute, Network):
                    attribute.save(save_path / attribute.name)
                elif isinstance(attribute, dict) and attribute != self.hyperparameters:
                    _save_models_in_dict(attribute)
                elif isinstance(attribute, Functional):
                    attribute.save(save_path / name)
                else:
                    non_model_attributes_dict[name] = attribute
                pbar.update(1)

        with open(save_path / "non_model_attributes.pkl", "wb") as file:
            pickle.dump(non_model_attributes_dict, file)

        vprint(self.verbose, f"... Model saved succesfully in {save_path}")

    def load(self, load_path: Union[Path, str] = None):
        assert load_path is not None, "No path specified to load model from"
        vprint(self.verbose, f"\nLoading model from: {load_path}\n")
        load_path = Path(load_path)

        # finally, load the dictionary storing the builtin/simple types, e.g. ints
        with open(load_path / "non_model_attributes.pkl", "rb") as file:
            non_model_attributes_dict = pickle.load(file)
        for name, attribute in non_model_attributes_dict.items():
            vars(self)[name] = attribute

        # an untrained model needs to be built before we can start loading it
        self.verbose = False
        self.build_model()

        # iterate over all the different item types to be loaded into the untrained model
        with tqdm(total=len(vars(self))) as pbar:
            for name, attribute in vars(self).items():
                if isinstance(attribute, Network):
                    attribute.load(load_path / name)
                elif isinstance(attribute, Functional):
                    vars(self)[name] = tf.keras.models.load_model(load_path / name, compile=False)
                elif isinstance(attribute, dict):
                    for item_name, item_in_dict in attribute.items():
                        if isinstance(item_in_dict, Network):
                            item_in_dict.load(load_path / item_in_dict.name)
                        elif isinstance(item_in_dict, Functional):
                            vars(self)[name][item_name] = tf.keras.models.load_model(
                                load_path / item_name, compile=False
                            )
                pbar.update(1)

        print("... Model loaded and ready!")

        return self

    def build_model(self):
        pass
