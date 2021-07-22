from pathlib import Path

_ROOT_PATH = Path(__file__).parents[2]
_SRC_PATH = _ROOT_PATH / "src"
_DATA_PATH = _ROOT_PATH / "data"
_TEST_DATA_PATH = _SRC_PATH / "tests" / "test_data"
_CWP_DATA_PATH = _DATA_PATH / "ssv_cooling_water_pump"
_FWP_DATA_PATH = _DATA_PATH / "ssv_feedwater_pump"
_SERIALIZED_MODELS_PATH = _ROOT_PATH / "serialized_models"
_MODELS_PATH = _SRC_PATH / "models"
_REPORTS_PATH = _ROOT_PATH / "reports"
_EXPERIMENT_LOG_PATH = _REPORTS_PATH / "experiment_logs"
_FIGURES_PATH = _REPORTS_PATH / "figures"
_CLEANING_FIGURES_PATH = _FIGURES_PATH / "data_cleaning"


def get_root_path() -> Path:
    return _ROOT_PATH


def get_src_path() -> Path:
    return _SRC_PATH


def get_data_path() -> Path:
    return _DATA_PATH


def get_test_data_path() -> Path:
    return _TEST_DATA_PATH


def get_cwp_data_path() -> Path:
    return _CWP_DATA_PATH


def get_fwp_data_path() -> Path:
    return _FWP_DATA_PATH


def get_serialized_models_path() -> Path:
    return _SERIALIZED_MODELS_PATH


def get_models_path() -> Path:
    return _MODELS_PATH


def get_figures_path() -> Path:
    return _FIGURES_PATH


# consider removing this
def get_cleaning_figures_path() -> Path:
    return _CLEANING_FIGURES_PATH


def get_reports_path() -> Path:
    return _REPORTS_PATH


def get_experiment_logs_path() -> Path:
    return _EXPERIMENT_LOG_PATH
