import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.data.data_utils import *


plt.style.use('seaborn')


class MarkovChain:
    """
    A discrete time MarkovChain class to model a power plant component's health state over time
    """

    def __init__(self, starting_state: int, probability_matrix: np.array):
        self.current_state = starting_state
        self.states = [0, 1]
        self.p_matrix = probability_matrix
        self.jumps_done = 0
        self.jump_sequence = [starting_state]

    def jump(self, n_jumps: int = 1):
        for _ in range(n_jumps):
            u = np.random.uniform()
            jump_probability = 1 - self.p_matrix[self.current_state, self.current_state]
            if u <= jump_probability:
                self.current_state = 1 - self.current_state
            self.jumps_done += 1
            self.jump_sequence.append(self.current_state)

    def get_jump_sequence_as_df(self):
        return pd.DataFrame({"state": self.jump_sequence})


if __name__ == "__main__":
    # np.random.seed(1234)

    chain = MarkovChain(0, np.array([[0.999, 0.001],
                                     [0.01, 0.99]]))
    chain.jump(10000)

    df = chain.get_jump_sequence_as_df()
    df.plot()
    plt.show()

