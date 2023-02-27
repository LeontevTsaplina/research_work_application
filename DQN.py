import matplotlib.pyplot as plt
import numpy as np

from Environment import Environment
from Agent import Agent
from Dataset import Dataset


def dqn(dataset: Dataset) -> Agent:
    """
    Function of dqn learning by input dataset

    :param dataset: input dataset
    :return: trained agent
    """

    # Model parameters
    BUFFER_SIZE = dataset.size()
    START_EPSILON = 1.0
    END_EPSILON = 0.01
    EPSILON_DECAY = 0.99
    rewards_list = []
    average_episode_rewards_list = []

    # Environment object
    env = Environment(dataset)

    # Agent object
    agent = Agent(env, buffer_size=BUFFER_SIZE)

    # Loop parameters
    num_episodes = dataset.episodes_count()
    epsilon = START_EPSILON

    # Loop for every episode
    for episode in range(num_episodes):

        # Rewards for batch
        rewards = agent.run_episode(epsilon)
        rewards_list.append(rewards)

        # Average reward for episode
        average_episode_rewards_list.append(np.mean(rewards_list[-1]))

        # Epsilon update
        epsilon = max(END_EPSILON, epsilon * EPSILON_DECAY)

        yield episode + 1

        '''
        # Printing average reward for last iteration for every 10 (deleted for application)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Average Rewards = {average_episode_rewards_list[-1]}")
        '''

    '''
    # Plotting results (deleted for application)
    plt.plot(average_episode_rewards_list)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per episode")
    plt.show()
    '''

    return agent
