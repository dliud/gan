from matplotlib import pyplot as plt
import torch
import numpy as np
from PIL import Image
from mlxtend.data import loadlocal_mnist
from PIL import Image
import numpy as np
import modules
import torch
import random
import torch.utils.data
import gan
import utils

#converts, displays, and saves images
def vector_to_img( vect, filename, display = False):
    #input: tensor of len 784 of floats form -1.0 to 1.0
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


def save_model(model, model_id, epoch):
    path = './models/gan{}-epoch{}.pkl'.format(model_id, epoch)
    torch.save(model.state_dict(), path)


def load_model(model_id, epoch):
    path = './models/gan{}-epoch{}.pkl'.format(model_id, epoch)
    return torch.load(path)


def loadDataset(train_size=1000, batch_size=100, 
                image_path='./mnist/train-images-idx3-ubyte', 
                label_path='./mnist/train-labels-idx1-ubyte'):
    """
    return: list of dataloaders, each containing train-size images of each number with batch size 
    """

    train_images, train_labels = loadlocal_mnist(
                                images_path=image_path, 
                                labels_path=label_path)

    # for i range(len(train_images)):
    #     train_images[i] = torch.tensor((train_images[i] - 128.)/128)

    sortedImages = [[] for _ in range(10)]
    for i in range(len(train_labels)):
        sortedImages[train_labels[i]].append(train_images[i])

    random.seed(17)

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


