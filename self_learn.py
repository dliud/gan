import numpy as np
import modules
import torch
import random
import torch.utils.data
import gan
import utils
import train_gans


def self_learn(dataloaders, num_iters=15, trial=20, alpha=1.1, train_size=250, batch_size=25, start=0,
               epoch_per_iter=600, num_gans=10):
    """
    alpha is "factor" to grow by; e.g. dataset size increases by alpha every iter; num_hidden_layers increases
    by alpha every iter

    trial inputted as a param should be >= 10, since we want trial numbers to start at 100 for these models
    """
    assert (trial >= 10)
    lst_saved_models = [None for _ in range(num_gans)]
    cur_dataloaders = dataloaders
    curr_size = train_size
    for i in range(start, num_iters):
        n_epoch = int(epoch_per_iter * (alpha ** i))
        n_trial = int(100 * trial + i)
        gans = train_gans.train_gans(lst_saved_models, cur_dataloaders, num_gans=num_gans, num_epochs=n_epoch,
                                     trial=n_trial,
                                     printProgress=True, updateEvery=50, alpha=alpha ** i)
        num_new_images = int((alpha - 1) * curr_size)
        curr_size += num_new_images
        # synth_images = utils.gen_synth_data(gans, n_entries=num_new_images, batch_size=25)
        next_dataloaders = [None for j in range(num_gans)]
        for j in range(num_gans):
            new_data = gans[j].generator(torch.randn(num_new_images, gans[j].gen_input_dim)).detach()
            next_dataloaders[j] = torch.utils.data.DataLoader(torch.cat((cur_dataloaders[j].dataset, new_data), 0),
                                                              batch_size=batch_size, shuffle=True, drop_last=True)
        cur_dataloaders = next_dataloaders


def main():
    real_size = 100  # number of real images per Dataloader to use at the beginning
    dataloaders, _ = utils.loadDataset(train_size=real_size, batch_size=25,
                                       image_path='./mnist/train-images-idx3-ubyte',
                                       label_path='./mnist/train-labels-idx1-ubyte')
    self_learn(dataloaders, train_size=real_size)


if __name__ == '__main__':
    main()
