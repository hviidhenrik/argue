import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silences excessive warning messages from tensorflow
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from src.models.ARGUE.models import ARGUE
from src.models.ARGUE.data_generation import *

if __name__ == "__main__":
    pass
