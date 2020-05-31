from matplotlib import pyplot as plt
from mlxtend.data import loadlocal_mnist
from PIL import Image
import torch
import random
import torch.utils.data
import os
import classifier
import errno


def gen_synth_data(gans, n_entries=20, batch_size=100):
    """
    params: gans: list of gans
        n_entries: number of fake images of each number to generate
        batch_size: batch size of the returned dataloader
    Returns a dataloader with each element being a synthetic image + label in the format (785,) tensor
    n_entries images of each number are generated, so n_entries * len(gans) total images
    """
    all_data = torch.zeros((0, 785))
    for i in range(len(gans)):
        noise = torch.randn(n_entries, gans[i].gen_input_dim)
        data = gans[i].generator(noise).detach()
        labeled = torch.cat((data, i * torch.ones((data.shape[0], 1))), 1)
        all_data = torch.cat((all_data, labeled), 0)
    return torch.utils.data.DataLoader(all_data, batch_size=batch_size, shuffle=True)


def vector_to_img(vect, filename, display=False):
    """
    Converts, displays, and saves images
    Input: tensor of len 784 of floats from -1.0 to 1.0
    """
    make_folder(filename)
    vect = vect.detach().numpy()
    vect = vect.reshape(-1, 28)
    img = Image.fromarray((vect + 1) * 128)
    img = img.convert("L")
    if display: img.show()
    img.save(filename)


def plot_loss(lst_epochs, lst_disc_loss, lst_gen_loss, title):
    """
    ~~~~~~~~for the GAN~~~~~~~~
    lst_epochs: List of epoch numbers
    lst_disc_loss:  List of discriminator losses
    lst_gen_loss: List of generator losses
    precondition: len(lst_epochs) == len(lst_disc_loss) == len(lst_gen_loss)
    """
    plt.plot(lst_epochs, lst_disc_loss, '-b', label='discriminator loss')
    plt.plot(lst_epochs, lst_gen_loss, '-r', label='generator loss')

    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.title(title)
    filename = title + ".png"
    make_folder(filename)
    plt.savefig(filename)
    # plt.show()

    plt.clf()
    plt.cla()
    plt.close()


def plot_loss_2(lst_epochs, lst_loss, title):
    """
    ~~~~~~~~for the classifier~~~~~~~~
    lst_epochs: List of epoch numbers
    lst_loss: List of classifier losses
    precondition: len(lst_epochs) == len(lst_loss)
    """
    plt.plot(lst_epochs, lst_loss, '-g', label='classifier loss')

    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.title(title)

    filename = title + ".png"
    make_folder(filename)
    plt.savefig(filename)
    # plt.show()

    plt.clf()
    plt.cla()
    plt.close()


def plot_devset_accuracy(lst_epochs, lst_accuracies, title):
    plt.plot(lst_epochs, lst_accuracies, '-g', label='accuracy (dev set)')

    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.title(title)

    filename = title + ".png"
    make_folder(filename)
    plt.savefig(filename)
    # plt.show()

    plt.clf()
    plt.cla()
    plt.close()


def save_model(gan, trial, ID, num_epoch):
    path = './models/trial{}/gan{}-epoch{}.pkl'.format(trial, ID, num_epoch)
    make_folder(path)
    torch.save(gan.state_dict(), path)


def load_model(gan, trial, ID, num_epoch):
    path = './models/trial{}/gan{}-epoch{}.pkl'.format(trial, ID, num_epoch)
    gan.load_state_dict(torch.load(path))


def loadDataset(train_size=1000, batch_size=100, randSeed=17,
                image_path='./mnist/train-images-idx3-ubyte',
                label_path='./mnist/train-labels-idx1-ubyte'):
    """
    return: list of dataloaders, each containing train-size images of each number with batch size 
    """
    random.seed(randSeed)
    train_images, train_labels = loadlocal_mnist(
        images_path=image_path,
        labels_path=label_path)

    sortedImages = [[] for _ in range(10)]
    for i in range(len(train_labels)):
        sortedImages[train_labels[i]].append(train_images[i])

    for images in sortedImages:
        random.shuffle(images)

    allData = torch.zeros((0, 785))
    dataLoaders = []
    for i in range(10):
        data = (torch.tensor(sortedImages[i][:train_size]) - 128.) / 128
        labeled = torch.cat((data, i * torch.ones((data.shape[0], 1))), 1)
        allData = torch.cat((allData, labeled), 0)
        dataLoaders.append(torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True))

    labeledDataLoader = torch.utils.data.DataLoader(allData, batch_size=batch_size, shuffle=True)
    return dataLoaders, labeledDataLoader


def get_dev_accuracy(classifier, dev_size=1000, randSeed=17,
                     image_path='./mnist/train-images-idx3-ubyte',
                     label_path='./mnist/train-labels-idx1-ubyte'):
    random.seed(randSeed)
    train_images, train_labels = loadlocal_mnist(images_path=image_path, labels_path=label_path)

    sortedImages = [[] for _ in range(10)]
    for i in range(len(train_labels)):
        sortedImages[train_labels[i]].append(train_images[i])

    for images in sortedImages:
        random.shuffle(images)

    test_images = []
    test_labels = []
    for i in range(10):
        test_images += sortedImages[i][-dev_size:]
        test_labels += [i for j in range(dev_size)]

    test = (torch.tensor(test_images) - 128.) / 128
    test_labels = torch.tensor(test_labels)
    predictions = classifier.predict(test)
    predictions = predictions.type(torch.uint8)
    return torch.mean(torch.eq(predictions, test_labels).float()).item()


def get_test_accuracy(classifier, image_path='./mnist/t10k-images-idx3-ubyte',
                      label_path='./mnist/t10k-labels-idx1-ubyte'):
    test_images, test_labels = loadlocal_mnist(images_path=image_path, labels_path=label_path)
    test = (torch.tensor(test_images) - 128.) / 128
    test_labels = torch.tensor(test_labels)
    predictions = classifier.predict(test)
    predictions = predictions.type(torch.uint8)
    return torch.mean(torch.eq(predictions, test_labels).float()).item()


# def test_discriminators(classifier, image_path='./mnist/t10k-images-idx3-ubyte',
# label_path='./mnist/t10k-labels-idx1-ubyte'):

def make_folder(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
