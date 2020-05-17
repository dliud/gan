import torch
import torch.nn as nn


class Classifier(torch.nn.Module):
    def __init__(self, lr, input_dim = 784, output_dim = 10, 
                nClasses=10, dropout_rate=.1):
        """
        3 hidden layers
        """
        super(Classifier, self).__init__()

        self.hidden1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate)
        )
        self.output = nn.Linear(32, output_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, input):
        """
        input has dimension (input_dim)
        """
        input = self.hidden1(input)
        input = self.hidden2(input)
        input = self.hidden3(input)
        input = self.output(input)
        return input

    def train(self, data_loader, num_epoch):
        lossfn = torch.nn.CrossEntropyLoss()
        for _ in range(num_epoch):
            for n_batch, data in enumerate(data_loader): 
                #n_batch 0,1,...
                #data = (batch_size, 784 + 1) 
                pred = self.forward(data[:, :self.input_dim])
                target = data[-1]
                loss = lossfn(pred, target)
                loss.backward()
                self.optimizer.step()

    def predict(self, data)
        #data = (m, input_size)
        #returns array of input size of labels 0-9
        weights = self.forward(data)
        _, pred = weights.max(1)
        return pred
