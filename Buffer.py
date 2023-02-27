import random
import numpy as np
from collections import deque


class Buffer:
    """
    Class of model experience buffer
    """

    def __init__(self, max_size: int) -> None:
        """
        Function to init buffer

        :param max_size: maximum size of buffer
        """

        self.__max_size = max_size
        self.__buffer = deque(maxlen=max_size)

    def add(self, state: list or np.array, action: list or np.array, reward: float, next_state: list or np.array,
            done: bool) -> None:
        """
        Function to append new buffer element

        :param state: current state
        :param action: action
        :param reward: reward for action
        :param next_state: next state
        :param done: done or no
        :return: None
        """

        self.__buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> zip:
        """
        Function to return samples of buffer by batch size

        :param batch_size: batch size
        :return: samples
        """

        # If batch size bigger then maximum buffer size
        if self.__max_size < batch_size:
            batch_size = self.__max_size

        return zip(*random.sample(self.__buffer, batch_size))

    def __len__(self) -> int:
        """
        Function to return length of buffer

        :return: length of buffer
        """

        return len(self.__buffer)

    def return_deque(self) -> deque:
        """
        Function to return deque of buffer

        :return: deque of buffer
        """

        return self.__buffer
