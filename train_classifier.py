import numpy as np
import modules
import torch
import random
import torch.utils.data
import gan
import utils
import classifier

real_data_to_test = [0, 500]
synth_data_to_test = [0, 500, 1000, 2000]


def mix(gans, num_real, num_synth):
    """
    return a dataloader
    """
    if num_real <= 0:
        return utils.gen_synth_data(gans, n_entries=num_synth)
    elif num_synth <= 0:
        return (utils.loadDataset(train_size=num_real, batch_size=100))[1]
    else:
        synth_data_loader = utils.gen_synth_data(gans, n_entries=num_synth)
        _, orig_data_loader = utils.loadDataset(train_size=num_real, batch_size=100)
        mixed_data_loader = torch.utils.data.DataLoader(torch.cat((orig_data_loader.dataset, synth_data_loader1.dataset), 0), 
                                                        batch_size=100, shuffle=True)
        return mixed_data_loader


def train_and_test(gans, trial, num_real, num_synth, fin, num_epoch=200):
    name = "./classifier_results/trial{}/{}-{}".format(trial, num_real, num_synth)
    data_loader = mix(gans, num_real, num_synth)
    # select from {SimpleClassifier, DeepClassifier}
    c = classifier.SimpleClassifier()
    c.train(data_loader, name, num_epoch=num_epoch)
    fin.write("Model: {}; Accuracy: {}\n".format(name, utils.get_test_accuracy(c)))


def main():
    gans = [gan.GAN(None, None, alpha = 4) for _ in range(10)]
    trial = 5
    for i in range(len(gans)):
        ID = '{}.{}'.format(trial, i)
        utils.load_model(gans[i], trial, ID, 2000)
    filename = "./classifier_results/trial{}/accuracies".format(trial)
    utils.make_folder(filename)
    fin = open(filename, "a")

    for num_real in real_data_to_test:
        for num_synth in synth_data_to_test:
            if num_real == 0 and num_synth == 0:
                continue
            train_and_test(gans, trial, num_real, num_synth, fin)

    fin.close()


    # synth_data_loader1 = utils.gen_synth_data(gans, n_entries=500, batch_size=100)
    # synth_data_loader2 = utils.gen_synth_data(gans, n_entries=1000, batch_size=100)
    # synth_data_loader4 = utils.gen_synth_data(gans, n_entries=2000, batch_size=100)
    # mixed_data_loader = torch.utils.data.DataLoader(torch.cat((orig_data_loader.dataset, synth_data_loader1.dataset), 0), batch_size=100, shuffle=True)
    # hidden_dim = 300
    # c_orig = classifier.SimpleClassifier(hidden_dim=hidden_dim)
    # c_synth1 = classifier.SimpleClassifier(hidden_dim=hidden_dim)
    # c_synth2 = classifier.SimpleClassifier(hidden_dim=hidden_dim)
    # c_synth4 = classifier.SimpleClassifier(hidden_dim=hidden_dim)
    # c_mixed = classifier.SimpleClassifier(hidden_dim=hidden_dim)
   
    #c_orig.train(orig_data_loader, num_epoch=200, name = './ClassifierResults2{}/Orig(500)'.format(trial))
    #c_synth1.train(synth_data_loader1, num_epoch=200, name = './ClassifierResults{}/Synth1(500)'.format(trial))
    #c_synth2.train(synth_data_loader2, num_epoch=200, name = './ClassifierResults{}/Synth2(1000)'.format(trial))
    #c_synth4.train(synth_data_loader4, num_epoch=200, name = './ClassifierResults{}/Synth4(2000)'.format(trial))
    #c_mixed.train(mixed_data_loader, num_epoch=200, name = './ClassifierResults{}/Mixed(500+500)'.format(trial))
    
    
    # print("Synthetic accuracy1: ", utils.get_test_accuracy(c_synth1))
    # print("Synthetic accuracy2: ", utils.get_test_accuracy(c_synth2))
    # print("Synthetic accuracy4: ", utils.get_test_accuracy(c_synth4))
    # print("Mixed accuracy: ", utils.get_test_accuracy(c_mixed))


if __name__ == "__main__":
    main()
