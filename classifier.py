import torch
import torch.nn as nn
import utils


class DeepClassifier(torch.nn.Module):
    def __init__(self, lr=.0002, input_dim=784, output_dim=10,
                 dropout_rate=.1):
        """
        3 hidden layers
        output_dim = number of classes
        """
        super(DeepClassifier, self).__init__()

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
        self.lst_dev_accuracy = []

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
            total_loss = 0
           
            for _, data in enumerate(data_loader):
                pred = self.forward(data[:, :-1])
                target = data[:, -1].long()
                loss = lossfn(pred, target)
                loss.backward()
                self.optimizer.step()

            self.lst_epochs.append(epoch)
            self.lst_loss.append(total_loss)
            if (epoch % 10 == 0):
                print(epoch, total_loss)
        utils.plot_loss_2(self.lst_epochs, self.lst_loss, "classifier_loss_synth" if synth else "classifier_loss_orig")

    def predict(self, data):
        weights = self.forward(data)
        _, pred = weights.max(1)
        return pred


class SimpleClassifier(torch.nn.Module):
    def __init__(self, lr=.0002, input_dim=784, output_dim=10, hidden_dim = 300, reg = .0001,
                 dropout_rate=.1):
        """
        3 hidden layers
        output_dim = number of classes
        """
        super(SimpleClassifier, self).__init__()

        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )
        
        self.output = nn.Linear(hidden_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay = reg)

        self.lst_epochs = []
        self.lst_loss = []
        self.lst_dev_accuracy = []

    def forward(self, input):
        """
        input has dimension (input_dim)
        """
        input = self.hidden(input)
        input = self.output(input)
        return input

    def train(self, data_loader, name, num_epoch=100):
        lossfn = torch.nn.CrossEntropyLoss()
        best_accuracy = 0
        best_params = None
        prev_accuracy = 0
        prev_prev_accuracy = 0
        for epoch in range(num_epoch):
            total_loss = 0
            for _, data in enumerate(data_loader):
                pred = self.forward(data[:, :-1])
                target = data[:, -1].long()
                loss = lossfn(pred, target)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            total_loss = total_loss/len(data_loader.dataset)
            self.lst_epochs.append(epoch)
            self.lst_loss.append(total_loss)
            curr_accuracy = utils.get_dev_accuracy(self)
            if curr_accuracy > best_accuracy:
                best_accuracy = curr_accuracy
                best_params = self.state_dict()
            self.lst_dev_accuracy.append(curr_accuracy)
            if (epoch % 10 == 0):
                print("Epoch: {}, Loss: {}".format(epoch, total_loss))
                print("Accuracy on the dev set: ", curr_accuracy)
                if (curr_accuracy > min(prev_accuracy, prev_prev_accuracy)):
                    prev_prev_accuracy = prev_accuracy
                    prev_accuracy = curr_accuracy
                else:
                    print("No improvement in accuracy.  Breaking at epoch ", epoch)
                    break
        utils.plot_loss_2(self.lst_epochs, self.lst_loss, name+"_loss")
        utils.plot_devset_accuracy(self.lst_epochs, self.lst_dev_accuracy, name + "_acc")
        self.load_state_dict(best_params)

    def predict(self, data):
        weights = self.forward(data)
        _, pred = weights.max(1)
        return pred
