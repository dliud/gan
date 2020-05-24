from matplotlib import pyplot as plt
from mlxtend.data import loadlocal_mnist
from PIL import Image
import numpy as np
import modules
import torch
import random
import torch.utils.data
import gan
import os
import errno

#converts, displays, and saves images
def vector_to_img( vect, filename, display = False):
    #input: tensor of len 784 of floats form -1.0 to 1.0
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    vect = vect.detach().numpy()
    vect = vect.reshape(-1, 28)
    img = Image.fromarray((vect+1)*128)
    img = img.convert("L")
    if display: img.show()
    img.save(filename)


def plot_loss(lst_epochs, lst_disc_loss, lst_gen_loss, title):
    """
    lst_epochs: List of epoch numbers
    lst_disc_loss:  List of discriminator losses
    lst_gen_loss: List of generator losses
    precondition: len(lst_epochs) == len(lst_disc_loss) == len(lst_gen_loss)
    """
    plt.plot(lst_epochs, lst_disc_loss, '-b', label='discriminator loss')
    plt.plot(lst_epochs, lst_gen_loss, '-r', label='generator loss')

    plt.xlabel('epoch')
    plt.legend(loc = 'upper right')
    plt.title(title)

    plt.savefig(title + ".png")
    plt.show()


def save_model(gan, trial, ID, num_epoch):
    path = './models/trial{}/gan{}-epoch{}.pkl'.format(trial, ID,  num_epoch)
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    torch.save(gan.state_dict(), path)


def load_model(gan, trial, ID, num_epoch):
    path = './models/trial{}/gan{}-epoch{}.pkl'.format(trial, ID, num_epoch)
    gan.load_state_dict(torch.load(path))


def loadDataset(train_size=1000, batch_size=100, randSeed = 17,
                image_path='./mnist/train-images-idx3-ubyte', 
                label_path='./mnist/train-labels-idx1-ubyte'):
    """
    return: list of dataloaders, each containing train-size images of each number with batch size 
    """
    random.seed(randSeed)
    train_images, train_labels = loadlocal_mnist(
                                images_path=image_path, 
                                labels_path=label_path)

    # for i range(len(train_images)):
    #     train_images[i] = torch.tensor((train_images[i] - 128.)/128)

    sortedImages = [[] for _ in range(10)]
    for i in range(len(train_labels)):
        sortedImages[train_labels[i]].append(train_images[i])

    

    for images in sortedImages: 
        random.shuffle(images)
 
    allData = torch.zeros((0, 785))
    dataLoaders = []
    for i in range(10):
        data = (torch.tensor(sortedImages[i][:train_size]) - 128.)/128
        labeled = torch.cat((data, i*torch.ones((data.shape[0], 1))), 1)
        allData = torch.cat((allData, labeled), 0)
        dataLoaders.append(torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True))
    
    labeledDataLoader = torch.utils.data.DataLoader(allData, batch_size=batch_size, shuffle=True)
    return dataLoaders, labeledDataLoader


