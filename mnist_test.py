from mlxtend.data import loadlocal_mnist
from PIL import Image
import numpy as np
import modules
import torch

X, y = loadlocal_mnist(
        images_path='./mnist/train-images-idx3-ubyte', 
        labels_path='./mnist/train-labels-idx1-ubyte')

imageDim = X[0].size
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
