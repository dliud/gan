import torch
import torch.nn as nn


class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, output_dim=1, dropout_rate=.1, alpha=4):
        """
        right now only doing binary classification, so output_dim = 1
        3 hidden layers
        alpha = model is proportional to (1+alpha)
        """
        super(Discriminator, self).__init__()

        self.hidden1 = nn.Sequential(
            nn.Linear(input_dim, int(128 * alpha)),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(int(128 * alpha), int(32 * alpha)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(int(32 * alpha), int(8 *alpha)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate)
        )
        self.output = nn.Sequential(
            nn.Linear(int(8 * alpha), output_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        input has dimension (input_dim)
        """
        input = self.hidden1(input)
        input = self.hidden2(input)
        input = self.hidden3(input)
        input = self.output(input)
        return input


class Generator(torch.nn.Module):
    def __init__(self, input_dim, output_dim=784, dropout_rate=.1, alpha=4):
        """
        no dropout
        3 hidden layers
        """
        super(Generator, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(input_dim, int(32*alpha)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(int(32*alpha), int(64*alpha)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(int(64*alpha), int(128*alpha)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate)
        )
        self.output = nn.Sequential(
            nn.Linear(int(128*alpha), output_dim),
            nn.Tanh()
        )

    def forward(self, input):
        input = self.hidden1(input)
        input = self.hidden2(input)
        input = self.hidden3(input)
        input = self.output(input)
        return input
