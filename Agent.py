import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim

from QNetwork import QNetwork
from Buffer import Buffer
from Environment import Environment


class Agent:
    """
    Class of model agent
    """

    def __init__(self, env: Environment, buffer_size: int = 100000, batch_size: int = 64, gamma: float = 0.99,
                 lr: float = 0.0001) -> None:
        """
        Function to init agent

        :param env: model environment
        :param buffer_size: model buffer size
        :param batch_size: model batch size
        :param gamma: gamma
        :param lr: learning rate
        """

        # Environment
        self.__env = env

        # State and action dimensions
        self.__state_dim = env.return_dataset().state_dim()
        self.__action_dim = env.return_dataset().action_dim()

        # Model parameters
        self.__batch_size = batch_size
        self.__gamma = gamma
        self.__lr = lr

        # QNetwork parameters
        self.__q_network = QNetwork(self.__state_dim, self.__action_dim)
        self.__optimizer = optim.Adam(self.__q_network.parameters(), lr=lr)

        # Buffer of agent
        self.__buffer = Buffer(buffer_size)

        # Function of loss for training
        self.__loss = nn.CrossEntropyLoss()

    def act(self, state: list or np.array, epsilon: float) -> int:
        """
        Function to return action by state by agent

        :param state: state
        :param epsilon: epsilon
        :return: action
        """

        # Epsilon-greedy exploration strategy
        if random.random() < epsilon:
            return random.choice(range(self.__action_dim))
        else:
            with torch.no_grad():
                q_values = self.__q_network.forward(torch.tensor(state).float())
                return q_values.argmax().item()

    def train(self) -> None:
        """
        Function to train agent

        :return: None
        """

        # Take buffer sample
        states, actions, rewards, next_states, dones = self.__buffer.sample(self.__batch_size)

        # Samples to tensors
        states = torch.tensor(states).float()
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards).unsqueeze(1)
        next_states = torch.tensor(next_states).float()
        dones = torch.tensor(dones).unsqueeze(1)

        # Forward and backward Q network
        q_values = self.__q_network.forward(states).gather(1, actions)

        next_q_values = self.__q_network.forward(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + self.__gamma * next_q_values * (1 - dones)

        loss = self.__loss(q_values, target_q_values)

        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()

    def run_episode(self, epsilon: float) -> list:
        """
        Function to run one episode of dataset

        :param epsilon: epsilon
        :return: list of rewards
        """

        # Reset env
        state = self.__env.reset()
        done = False
        rewards = []

        # Train for every case in episode
        while not done:
            action = self.act(state, epsilon)
            next_state, reward, done = self.__env.step(action)
            rewards.append(reward)
            self.__buffer.add(state, action, reward, next_state, done)
            state = next_state
            if len(self.__buffer) >= self.__batch_size:
                self.train()

        return rewards

    def buffer_append(self, state: list or np.array, action: list or np.array, reward: float,
                      next_state: list or np.array, done: bool) -> None:
        """
        Function to add new element in agent's buffer

        :param state: current state
        :param action: action
        :param reward: reward for this action
        :param next_state: next state
        :param done: is it done
        :return: None
        """

        self.__buffer.add(state, action, reward, next_state, done)

    def return_predict(self, state: list or np.array) -> int:
        """
        Function to return predict by agent for input state

        :param state: state
        :param device: device
        :return: action
        """

        return [column.replace('_dinam_control', '') for column in self.return_dataset().columns if column.endswith('_dinam_control')][self.__q_network.predict(state)]

    def return_dataset(self) -> pd.DataFrame:
        """
        Function to return agent dataset

        :return: dataset
        """

        return self.__env.return_dataset().return_dataset()
