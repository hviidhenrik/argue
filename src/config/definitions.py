import os

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__ + "..//..//..//"))
_MODELS_PATH = os.path.join(_ROOT_DIR, "src","models")
_MLOPS_PATH = os.path.join(_ROOT_DIR, "src","deployment")
_ARCHIVE_MODELS_PATH = os.path.join(_ROOT_DIR, "archive")
_PDM_DIST_PATH = os.path.join(_ROOT_DIR, "dist")

def get_project_root() -> str:
    """Return path to project root"""
    return _ROOT_DIR

def get_archive_folder_path() -> str:
    """Return archive folder path"""
    return _ARCHIVE_MODELS_PATH

def get_pdm_dist_folder_path() -> str:
    """Return dist folder path"""
    return _PDM_DIST_PATH

def get_model_folder_path() -> str:
    """Return model folder path"""
    return _MODELS_PATH

def get_model_deploy_folder_path() -> str:
    """Return model folder path"""
    return _MLOPS_PATH
