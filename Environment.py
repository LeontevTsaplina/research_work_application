from abc import ABC

import gym
import numpy as np
import pandas as pd

from Dataset import Dataset


class Environment(gym.Env, ABC):
    """
    Class of model Environment
    """

    def __init__(self, dataset: Dataset) -> None:
        """
        Function to init environment of model by dataset

        :param dataset: dataset object
        """

        # Dataset object
        self.__dataset = dataset

        # Dataset like DataFrame
        self.__dataset_lines = dataset.return_dataset()

        # Dataset states
        self.__states = dataset.return_states_and_actions()[0]

        # Dataset unique cases list
        self.__cases = dataset.return_dataset().index.unique()

        # Environment parameters
        self.__current_case = None
        self.__current_case_num = -1
        self.__current_step = 0

    def reset(self, **kwargs) -> pd.DataFrame:
        """
        Function of reset of environment

        :return:
        """

        self.__current_case_num += 1
        self.__current_step = 0
        self.__current_case = self.__cases[self.__current_case_num]

        return self.__states.loc[[self.__current_case]].iloc[0]

    def step(self, action: list or np.array) -> tuple[list or np.array, float, bool]:
        """
        Function of environment step by action

        :param action: action
        :return: tuple of (next_action, reward, done)
        """

        done = self.__dataset_lines.loc[[self.__current_case]].iloc[self.__current_step].end_epizode
        reward = -100 * self.__dataset_lines.loc[[self.__current_case]].iloc[self.__current_step].outcome_tar - \
                 self.__dataset_lines.loc[[self.__current_case]].iloc[self.__current_step].current_process_duration + 100

        if done:
            next_state = self.__states.loc[[self.__current_case]].iloc[self.__current_step]
        else:
            self.__current_step += 1
            self.__current_case = self.__cases[self.__current_case_num]
            next_state = self.__states.loc[[self.__current_case]].iloc[self.__current_step]

        return next_state, reward, done

    def return_dataset(self) -> Dataset:
        """
        Function to return environment input dataset

        :return: environment input dataset
        """

        return self.__dataset
