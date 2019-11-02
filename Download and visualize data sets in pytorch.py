#Download and visualize data sets in pytorch
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

"""#MNIST"""

train = MNIST('./mnist_folder', train=True, download=True)

train

train[0]

image,tag = train[0]

image

tag

from PIL import Image

new_iamge = image.resize((200,200), Image.ANTIALIAS)

new_iamge

"""#FashionMNIST"""

from torchvision.datasets import FashionMNIST
train = FashionMNIST('./fansionmnist_folder', train=True, download=True)

train

train[0]

image,label = train[1]

image

from PIL import Image
new_iamge = image.resize((200,200), Image.ANTIALIAS)
new_iamge

label



from torchvision.datasets import KMNIST
train = KMNIST('./kmnist_folder', train=True, download=True)

train

train[0]

image,label = train[0]

image

from PIL import Image
new_iamge = image.resize((200,200), Image.ANTIALIAS)
new_iamge

import torchvision.datasets as dsets
import torchvision.transforms as transforms

train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

batch_size = 256
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

train_loader

for image,label in train_loader:
  print(image)
  print(label)
  print(len(image))

