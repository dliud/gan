from matplotlib import pyplot as plt
import torch
import numpy as np
from PIL import Image

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