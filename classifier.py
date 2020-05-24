import torch
import torch.nn as nn


class Classifier(torch.nn.Module):
    def __init__(self, lr=.0002, input_dim=784, output_dim=10, 
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

        self.lst_epochs = []
        self.lst_loss = []

    def forward(self, input):
        """
        input has dimension (input_dim)
        """
        input = self.hidden1(input)
        input = self.hidden2(input)
        input = self.hidden3(input)
        input = self.output(input)
        return input

    def train(self, data_loader, num_epoch=100, synth=False):
        lossfn = torch.nn.CrossEntropyLoss()
        for epoch in range(num_epoch):
            if (epoch % 10 == 0):
                print(epoch)
            for n_batch, data in enumerate(data_loader): 
                pred = self.forward(data[:, :-1])
                target = data[:, -1].long()
                loss = lossfn(pred, target)
                loss.backward()
                self.optimizer.step()
            
            self.lst_epochs.append(epoch)
            self.lst_loss.append(loss)
        utils.plot_loss_2(self.lst_epochs, self.lst_loss, "classifier_loss_synth" if synth else "classifier_loss_orig")


    def predict(self, data):
        #data = (m, input_size)
        #returns array of input size of labels 0-9
        weights = self.forward(data)
        _, pred = weights.max(1)
        return pred


