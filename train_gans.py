import numpy as np
import modules
import torch
import random
import torch.utils.data
import gan
import utils


def train_gans(lst_saved_models, dataloaders, trial, alpha, num_gans=10, num_epochs=2000, 
                printProgress=True, updateEvery=50):
    """
    lst_saved_models: List of Tuples(ID, num_epoch)
    where ID is {trial}.{numGAN} and num_epoch is the epoch of the model that you want to restore

    NAMING CONVENTION: {trial}.{number that GAN is supposed to work on}
    """
    gans = []
    for i in range(num_gans):
        name = '{}.{}'.format(trial, i)
        gans.append(gan.GAN(trial, name, discriminator_steps=1, generator_steps=2, disc_input_dim=784,
                            gen_input_dim=100, lr_disc=.00075, lr_gen=.00015, label_smooth=True, alpha = alpha))

    for i in range(num_gans):
        epoch = 0
        if lst_saved_models[i] is not None:
            print("--------Loading GAN", i, " from a previously saved model--------")
            (ID, epoch) = lst_saved_models[i]
            utils.load_model(gans[i], trial, ID, epoch)
            assert (gans[i] is not None), "Model didn't exist!"

        if epoch < num_epochs - 1:
            print("--------Training GAN", i, "--------")
            gans[i].train(dataloaders[i], num_epochs, start_epoch=epoch, 
                            printProgress=printProgress, updateEvery=updateEvery)
        else:
            print("GAN", i, " was already fully trained to ", epoch, " epochs.")

    return gans


def repeatTrain(dataloaders, trial, epoch_len, alpha,  end, start=0):
    """
    dataloaders: list of 10 dataloaders which have data
    trial: trial number to save things
    epoch_len: train each gan for epoch_len epochs before training the next can
    """
    lst_saved_models = [None for _ in range(10)]
    
    
    prev_stop = start
    next_stop = prev_stop + epoch_len
    
    while prev_stop < end:
        if prev_stop != 0:
            for i in range(10):
                ID = '{}.{}'.format(trial, i)
                lst_saved_models[i] = (ID, prev_stop)
        
        train_gans(lst_saved_models, dataloaders, num_gans=10, num_epochs=next_stop, trial=trial, 
                printProgress=True, updateEvery=50, alpha = alpha)
        prev_stop = next_stop   
        next_stop = prev_stop + epoch_len
        print("Done with {} epochs".format(prev_stop))


def main(train_size, model_size, trial):
    num_gans = 10
    dataloaders, labeledDataLoader = utils.loadDataset(train_size=train_size, batch_size=25,
                                                        image_path='./mnist/train-images-idx3-ubyte',
                                                        label_path='./mnist/train-labels-idx1-ubyte')
    
    repeatTrain(dataloaders, trial = trial, epoch_len = 500, end = 2000, alpha = model_size)
    """
    lst_saved_models = [None for _ in range(num_gans)]
    
    trial = 1, saved_epoch = 30
    for i in range(num_gans):
        ID = '{}.{}'.format(trial, i)
        lst_saved_models[i] = (ID, saved_epoch)
    
    train_gans(lst_saved_models, dataloaders, num_gans=num_gans, num_epochs=60, trial=1, 
                printProgress=True, updateEvery=10)
    
    """


if __name__ == "__main__":
    main()
