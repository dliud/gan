from mlxtend.data import loadlocal_mnist
from PIL import Image
import numpy as np
import modules
import torch
import random
import torch.utils.data
import train

train_images, train_labels = loadlocal_mnist(
        images_path='./mnist/train-images-idx3-ubyte', 
        labels_path='./mnist/train-labels-idx1-ubyte')

# for i range(len(train_images)):
#     train_images[i] = torch.tensor((train_images[i] - 128.)/128)

sortedImages = [[] for _ in range(10)]
for i in range(len(train_labels)):
    sortedImages[train_labels[i]].append(train_images[i])

random.seed(17)

for images in sortedImages: 
    random.shuffle(images)

train_size = 1000
batch_size = 20
dataLoaders = []
for i in range(10):
    data = (torch.tensor(sortedImages[i][:train_size]) - 128.)/128
    dataLoaders.append(torch.utils.data.DataLoader(data, batch_size=100, shuffle=True))

gan_0 = train.GAN(discriminator_steps=1, disc_input_dim=784, 
                gen_input_dim=100, batch_size=10, lr_disc=.0002, 
                lr_gen=.0002)

gan_0.train(dataLoaders[0], 20)

"""
disc = modules.Discriminator(imageDim)
for i in range(5):
    data = torch.tensor((X[i] - 128.)/128)
    print(disc(data) )

noiseDim = 100
gen = modules.Generator(noiseDim)
for i in range(5):
    image = gen(torch.randn(noiseDim))
    image = image.detach().numpy()
    image = image.reshape(28,28)
    img = Image.fromarray(image*256)
    img.show()
"""
