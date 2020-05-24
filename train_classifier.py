import numpy as np
import modules
import torch
import random
import torch.utils.data
import gan
import utils
import classifier

def main():
    _, orig_data_loader= utils.loadDataset(train_size=1000, batch_size=100, 
                                                        image_path='./mnist/train-images-idx3-ubyte',
                                                        label_path='./mnist/train-labels-idx1-ubyte')
    
    gans = [gan.GAN(None, None) for _ in range(10)]
    trial = 1
    for i in range(len(gans)):
        ID = '{}.{}'.format(trial, i)
        utils.load_model(gans[i], trial, ID, 20000)
    synth_data_loader = utils.gen_synth_data(gans, n_entries=1000, batch_size=100)
    

    c_orig = classifier.Classifier()
    c_synth = classifier.Classifier()
    c_orig.train(orig_data_loader, num_epoch=100, synth=False)
    c_synth.train(synth_data_loader, num_epoch=100, synth=True)
    print("Orig accuracy: {}", utils.get_accuracy(c_orig))
    print("Synth accuracy: {}", utils.get_accuracy(c_synth))

if __name__ == "__main__":
    main()