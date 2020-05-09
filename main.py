import pandas as brrrr
import torch.nn as nn

class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, dropout_rate):
        """
        right now only doing binary classification
        """
        self.hidden1 = nn.Sequential(
            nn.Linear(input_dim, 768);
            nn.LeakyReLU(negative_slope=0.01);
            nn.Dropout(dropout_rate);
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(768, 256);
            nn.LeakyReLU(negative_slope=0.01);
            nn.Dropout(dropout_rate);
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(256, 64);
            nn.LeakyReLU(negative_slope=0.01);
            nn.Dropout(dropout_rate);
        )
        self.output = nn.Sequential(
            nn.Linear(64, 1)  
            self.sigmoid = nn.Sigmoid()
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
    