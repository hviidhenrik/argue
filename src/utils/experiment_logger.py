import csv
from datetime import datetime
from typing import Dict

from src.models.base_model import BaseModel
from src.utils.misc import *
from src.utils.model import *


class ExperimentLogger:
    def __init__(self):
        self.save_path = get_experiment_logs_path()
        self.experiment_id = None

    def save_model_parameter_log(self,
                                 model: BaseModel,
                                 experiment_name: Optional[str],
                                 path: Optional[Path] = None):
        path = path if path else self.save_path
        date_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        experiment_id = self._determine_experiment_id()
        params_to_save = {"model_name": model.model_name, "run_date_time": date_time, "experiment_id": experiment_id}
        params_to_save.update(model.hyperparameters)
        filename = path / f"{experiment_name}_ID{experiment_id}.csv"
        self._save_dict_as_csv(params_to_save, filename=filename)

    def _determine_experiment_id(self):
        experiment_id = "001"
        existing_files = os.listdir(self.save_path)
        # if previous experiments exist, just follow up on their ID's
        if len(existing_files) > 0:
            existing_ids = []
            for file in existing_files:
                # find the latest existing experiment
                id = file[file.find("ID") + 2: file.find(".")]
                existing_ids.append(int(id))
            experiment_id = str(np.max(existing_ids) + 1)
        # prefix the id number with 0's to ensure the format is always IDxxx
        while len(experiment_id) < 3:
            experiment_id = "0" + experiment_id
        self.experiment_id = experiment_id
        return experiment_id

    @staticmethod
    def _save_dict_as_csv(dict: Dict, filename: Union[str, Path]):
        filename = str(filename)
        assert ".csv" in filename, "File type must be .csv"
        with open(filename, 'w') as file:
            w = csv.writer(file)
            w.writerows(dict.items())

    def get_experiment_id(self):
        assert self.experiment_id is not None, "Experiment ID not yet generated, call save_model_parameter_log first"
        return self.experiment_id

    def _find_duplicate_experiment_logs(self):
        pass

    def _remove_duplicate_experiment_logs(self):
        # go over log files with the same names (apart from the ID) and import them to see if they are identical
        # keep only the latest log file
        pass

    def _remove_old_experiment_figures(self):
        # check which experiment ID's exist in reports folder and delete figures that don't have
        # any matching ID in there
        pass
