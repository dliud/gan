import numpy as np
import modules
import torch
import random
import torch.utils.data
import gan
import utils

# all tweaking should be done in main below
def train_gans(lst_saved_models, dataloaders, num_gans=10, num_epochs=2000):
    """
    lst_saved_models: List of Tuples(model_id, epoch)
    """
    gans = []
    for i in range(num_gans):
        gans.append(gan.GAN(2.2, discriminator_steps=1, generator_steps=2, disc_input_dim=784,
                gen_input_dim=100, batch_size=10, lr_disc=.0001, lr_gen=.00015))
    
    for i in range(num_gans):
        epoch = 0
        if lst_saved_models is not None:
            print("Loading GAN ", i ," from a previously saved model...")
            (model_id, epoch) = lst_saved_models[i]
            gans[i] = load_model(model_id, epoch)
            assert (gans[i] is not None), "Model didn't exist!"
        
        if epoch < num_epochs - 1:
            print("Training GAN ", i, "...")
            gans[i].train(dataLoaders[i], num_epochs, epoch)
        else:
            print("GAN ", i, " was already fully trained to ", epoch, " epochs.")
    
    return gans


def main():
    num_gans = 10
    dataloaders = utils.loadDataset(image_path='./mnist/train-images-idx3-ubyte', 
                                    label_path='./mnist/train-labels-idx1-ubyte')
    lst_saved_models = []
    for i in range(num_gans):
        lst_saved_models[i] = None
    """
    can overwrite to a saved model in this space, e.g.
    lst_saved_models[2] = (2, 1999)
    """
    train_gans(lst_saved_models, dataloaders, num_gans=num_gans, num_epochs=50)


if __name__ == "__main__":
    main()