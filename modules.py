import torch
import torch.nn as nn

class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, output_dim=1, dropout_rate = .1):
        """
        right now only doing binary classification, so output_dim = 1
        3 hidden layers
        """
        super(Discriminator, self).__init__()

        self.hidden1 = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(768, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate)
        )
        self.output = nn.Sequential(
            nn.Linear(64, output_dim), 
            nn.Sigmoid()
        )
        # could be 10 depending on task
        # self.softmax = nn.Softmax()

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
    def __init__(self, input_dim, output_dim=784, dropout_rate = .1):
        """
        no dropout
        3 hidden layers
        """
        super(Generator, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(256, 768),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate)
        )
        self.output = nn.Sequential(
            nn.Linear(768, output_dim),
            nn.Tanh()  
        )  # transformation: (coords[i] - 128) / 128
    
    def forward(self, input):
        input = self.hidden1(input)
        input = self.hidden2(input)
        input = self.hidden3(input)
        input = self.output(input)
        return input

    