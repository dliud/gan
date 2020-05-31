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
    :return: a Dataloader with num_real real and num_synth synthetic images
    """
    if num_real <= 0:
        return utils.gen_synth_data(gans, n_entries=num_synth)
    elif num_synth <= 0:
        return (utils.loadDataset(train_size=num_real, batch_size=100))[1]
    else:
        synth_data_loader = utils.gen_synth_data(gans, n_entries=num_synth)
        _, orig_data_loader = utils.loadDataset(train_size=num_real, batch_size=100)
        mixed_data_loader = torch.utils.data.DataLoader(torch.cat((orig_data_loader.dataset, synth_data_loader.dataset), 0),
                                                        batch_size=100, shuffle=True)
        return mixed_data_loader


def train_and_test(gans, trial, num_real, num_synth, fin, num_epoch=200):
    name = "./classifier_results/trial{}/{}-{}".format(trial, num_real, num_synth)
    data_loader = mix(gans, num_real, num_synth)
    c = classifier.SimpleClassifier()  # select from {SimpleClassifier, DeepClassifier}
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

if __name__ == "__main__":
    main()
