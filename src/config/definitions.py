from pathlib import Path, WindowsPath

_ROOT_PATH = Path(__file__).parents[2]
_SRC_PATH = _ROOT_PATH / "src"
_DATA_PATH = _SRC_PATH / "data"
_TEST_DATA_PATH = _ROOT_PATH / "tests" / "test_data"
_PUMP_DATA_PATH = _DATA_PATH / "ssv_cooling_water_pump"
_MODELS_ARCHIVE_PATH = _ROOT_PATH / "archive"
_AAE_PATH = _SRC_PATH / "models" / "AAE"
_ANOGEN_PATH = _SRC_PATH / "models" / "AnoGen"
_DOPING_PATH = _SRC_PATH / "models" / "Doping"
_FIXED_CYCLE_PATH = _SRC_PATH / "models" / "FixedCycleTest"
_FIXED_CYCLE_FIGURES_PATH = _FIXED_CYCLE_PATH / "figures"

# everything related to ARGUE
_ARGUE_PATH = _SRC_PATH / "models" / "ARGUE"
_ARGUE_FIGURES_PATH = _ARGUE_PATH / "figures"
_ARGUE_CLEANING_FIGURES_PATH = _ARGUE_PATH / "figures" / "data_cleaning"

def get_root_path() -> WindowsPath:
    return _ROOT_PATH


def get_src_path() -> WindowsPath:
    return _SRC_PATH


def get_data_path() -> WindowsPath:
    return _DATA_PATH


def get_test_data_path() -> WindowsPath:
    return _TEST_DATA_PATH


def get_pump_data_path() -> WindowsPath:
    return _PUMP_DATA_PATH


def get_model_archive_path() -> WindowsPath:
    return _MODELS_ARCHIVE_PATH


def get_aae_path() -> WindowsPath:
    return _AAE_PATH


def get_anogen_path() -> WindowsPath:
    return _ANOGEN_PATH


def get_doping_path() -> WindowsPath:
    return _DOPING_PATH


def get_fixed_cycle_path() -> WindowsPath:
    return _FIXED_CYCLE_PATH


def get_fixed_cycle_figures_path() -> WindowsPath:
    return _FIXED_CYCLE_FIGURES_PATH

def get_ARGUE_path() -> WindowsPath:
    return _ARGUE_PATH

def get_ARGUE_figures_path() -> WindowsPath:
    return _ARGUE_FIGURES_PATH

def get_ARGUE_cleaning_figures_path() -> WindowsPath:
    return _ARGUE_CLEANING_FIGURES_PATH
