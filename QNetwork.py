import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Class of QNetwork
    """

    def __init__(self, n_input_neurons: int, n_output_neurons: int, n_hidden_neurons: int = 128):
        """
        Function to init Q network

        :param n_input_neurons: number of input neurons
        :param n_output_neurons: number of output neurons
        :param n_hidden_neurons: number of hidden neurons
        """

        super(QNetwork, self).__init__()

        self.f = nn.ReLU()
        self.layer_1 = nn.Linear(n_input_neurons, n_hidden_neurons)
        self.layer_2 = nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.layer_3 = nn.Linear(n_hidden_neurons, n_output_neurons)

    def forward(self, x: list or np.array) -> np.array:
        """
        Function to make forward of Q network

        :param x: input state
        :return: output state
        """

        x = self.layer_1(x)
        x = self.f(x)
        x = self.layer_2(x)
        x = self.f(x)
        x = self.layer_3(x)

        return x

    def predict(self, x: list or np.array) -> int:
        """
        Function to make predict by network

        :param x: input state
        :param device: torch device
        :return: output state
        """

        device = ('cuda' if torch.cuda.is_available() else 'cpu')

        x = torch.tensor(x, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            x = self.forward(x)
        x = x.cpu().argmax(dim=-1)
        x = x.data.numpy()

        return x
