import numpy as np
import modules
import torch
import random
import torch.utils.data
import gan
import utils
import classifier


def main():
    _, orig_data_loader = utils.loadDataset(train_size=1000, batch_size=100,
                                            image_path='./mnist/train-images-idx3-ubyte',
                                            label_path='./mnist/train-labels-idx1-ubyte')

    gans = [gan.GAN(None, None) for _ in range(10)]
    trial = 2
    for i in range(len(gans)):
        ID = '{}.{}'.format(trial, i)
        utils.load_model(gans[i], trial, ID, 7500)
    synth_data_loader1 = utils.gen_synth_data(gans, n_entries=1000, batch_size=100)
    synth_data_loader2 = utils.gen_synth_data(gans, n_entries=2000, batch_size=100)
    synth_data_loader4 = utils.gen_synth_data(gans, n_entries=4000, batch_size=100)
    # select from {SimpleClassifier, DeepClassifier}
    c_orig = classifier.SimpleClassifier()
    c_synth1 = classifier.SimpleClassifier()
    c_synth2 = classifier.SimpleClassifier()
    c_synth4 = classifier.SimpleClassifier()
    c_orig.train(orig_data_loader, num_epoch=100, name = 'Orig (1000)')
    c_synth1.train(synth_data_loader1, num_epoch=100, name = 'Synth1 (1000)')
    c_synth2.train(synth_data_loader2, num_epoch=50, name = 'Synth2 (2000)')
    c_synth4.train(synth_data_loader4, num_epoch=25, name = 'Synth4 (4000)')
    print("Orig accuracy: {}", utils.get_accuracy(c_orig))
    print("Synthetic accuracy1: {}", utils.get_accuracy(c_synth1))
    print("Synthetic accuracy2: {}", utils.get_accuracy(c_synth2))
    print("Synthetic accuracy4: {}", utils.get_accuracy(c_synth4))


if __name__ == "__main__":
    main()
