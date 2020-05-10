from matplotlib import pyplot as plt
import torch
import numpy as np
from PIL import Image

def vector_to_img(vect, filename):
    #input: tensor of len 784 or floats form -1.0 to 1.0
    vect = vect.detach().numpy()
    vect = vect.reshape(-1, 28)
    img = Image.fromarray((vect+1)*128)
    img.show()
    #img.save(filename)



"""
should not plot discriminator and generator losses on the same graph unlsss normalize
this method would only work if discriminator_steps = 1
"""
def plot_loss(lst_epochs, lst_disc_loss, lst_gen_loss, title):
    """
    lst_epochs is just the epoch
    lst_disc_loss should be List of discriminator losses
    lst_gen_loss should be List of generator losses
    precondition: len(lst_epochs) == len(lst_disc_loss) == len(lst_gen_loss)
    """
    plt.plot(lst_epochs, lst_disc_loss, '-b', label='discriminator loss')
    plt.plot(lst_epochs, lst_gen_loss, '-r', label='generator loss')

    plt.xlabel('epoch')
    plt.legend(loc = 'upper right')
    plt.title(title)

    plt.savefig(title + ".png")
    plt.show()