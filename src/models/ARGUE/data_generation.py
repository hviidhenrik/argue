import numpy as np
import pandas as pd


def make_class_labels(classes: int, N: int):
    labels = []
    for i in range(1, classes+1):
        labels += [i for _ in range(N)]
    return labels


def make_custom_test_data(N1, N2, N3):
    df = pd.concat([
        pd.DataFrame({"x1": 4 + np.sin(np.linspace(0, 10, N1) + np.random.normal(0, 0.1, N1)),
                      "x2": 4 + np.cos(np.linspace(0, 10, N1) + np.random.normal(0, 0.1, N1)),
                      "x3": 4 + np.cos(3.14 + np.linspace(0, 10, N1) + np.random.normal(0, 0.1, N1)),
                      "class": 1
                      }),
        pd.DataFrame({"x1": 500 + np.sin(np.linspace(0, 10, N2) + np.random.normal(0, 0.1, N2)),
                      "x2": 500 + np.cos(np.linspace(0, 10, N2) + np.random.normal(0, 0.1, N2)),
                      "x3": 500 + np.cos(3.14 + np.linspace(0, 10, N2) + np.random.normal(0, 0.1, N2)),
                      "class": 2
                      }),
        pd.DataFrame({"x1": -100 - 2 * np.linspace(0, 10, N3) + np.random.normal(0, 0.1, N3),
                      "x2": -100 - 3 * np.linspace(0, 10, N3) + np.random.normal(0, 0.1, N3),
                      "x3": -100 - 1 * np.linspace(0, 10, N3) + np.random.normal(0, 0.1, N3),
                      "class": 3
                      })
    ]).reset_index(drop=True)
    return df


