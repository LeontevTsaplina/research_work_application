import pandas as pd
import os


class Dataset:
    """
    Class of methods for input dataset
    """

    def __init__(self, file_path: str or os.path) -> None:
        """
        Function to init object of class

        :param file_path: path to file with data
        """

        # Read file
        dataset = pd.read_pickle(file_path)

        # Means of columns by cases
        grouped_means = dataset.groupby(dataset.index).mean(numeric_only=True).round(2)

        # Replace na by mean by group
        dataset = dataset.combine_first(grouped_means)

        # Drop na's from dataset
        dataset.dropna(how='any', inplace=True)

        # Sort dataset by case number and t_point
        dataset = dataset.sort_values(by=['case', 't_point'])

        # Making a columns order
        dataset = dataset[['t_point', 'current_process_duration', 'long_observation_tar'] +
                          [column for column in dataset.columns if column.endswith('_stat_control')] +
                          [column for column in dataset.columns if column.endswith('_stat_fact')] +
                          [column for column in dataset.columns if column.endswith('_dinam_fact')] +
                          [column for column in dataset.columns if column.endswith('_dinam_control')] +
                          ['end_epizode', 'outcome_tar']]

        self.__dataset = dataset

    def return_dataset(self) -> pd.DataFrame:
        """
        Function of returning dataset

        :return: object's dataframe
        """

        return self.__dataset

    def return_states_and_actions(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Function of returning states and actions of dataset

        :return: tuple of states and actions dataset
        """

        return (self.__dataset[[column for column in self.__dataset.columns if column.endswith('_stat_control')] +
                               [column for column in self.__dataset.columns if column.endswith('_stat_fact')] +
                               [column for column in self.__dataset.columns if column.endswith('_dinam_fact')]],
                self.__dataset[[column for column in self.__dataset.columns if column.endswith('_dinam_control')]])

    def size(self) -> int:
        """
        Function of returning dataset size

        :return: dataset size
        """

        return len(self.__dataset)

    def state_dim(self) -> int:
        """
        Function of returning dataset state dimensional

        :return: dataset state dimensional
        """

        return len([column for column in self.__dataset.columns if column.endswith('_stat_control')] +
                   [column for column in self.__dataset.columns if column.endswith('_stat_fact')] +
                   [column for column in self.__dataset.columns if column.endswith('_dinam_fact')])

    def action_dim(self) -> int:
        """
        Function of returning dataset action dimensional

        :return: dataset action dimensional
        """

        return len([column for column in self.__dataset.columns if column.endswith('_dinam_control')])

    def episodes_count(self) -> int:
        """
        Function to return count of episodes

        :return: count of episodes
        """

        return len(self.__dataset.index.unique())

