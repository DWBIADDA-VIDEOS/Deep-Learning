# -*- coding: utf-8 -*-
"""Linear Regression in Pytorch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WyeGTTyCt2fRZnQAh7KBHgFvHYKfMXNT

<img style="float: centre;" src="https://drive.google.com/uc?id=1ZVzMRu99qZNwgAM0myg0Ub0-I9hVcT9F">

# We create our own dataset

## We use numpy to to create the the x values and the corresponding y values
"""

import numpy as np
import matplotlib.pyplot as plt

"""# Lets take first 10 positive integers and fit them in a linear equation of $y=1.5x + 13$"""

x_values = [i for i in range(1,11)]

x_train = np.array(x_values, dtype=np.float32)
print(x_train)
print(x_train.shape)
x_train = x_train.reshape(-1, 1)
x_train.shape

x_train

y_values = [1.5*i + 13 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
print(y_train)
print(y_train.shape)
y_train = y_train.reshape(-1, 1)
y_train.shape

y_train

plt.scatter(x_train, y_train,alpha=0.5)
plt.show()

"""# Write the basic Linear Regression module"""

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import SGD

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  
    
    def forward(self, x):
        out = self.linear(x)
        return out

linear_model = LinearRegressionModel()

linear_model.cuda()

linear_model.parameters()

"""# Fine tuning our Linear Model:
1. Loss Function
2. Optimizer

## Loss function -> Mean Squared Error
"""

criterion = nn.MSELoss()

"""## Optimzer -> Stochastic Gradient Descent"""

optimizer = SGD(linear_model.parameters(), lr=0.01)

"""# How many times do you want to fine tune?
Answer : Your epoch number
"""

number_of_epoch = 1000
loss_in_each_epoch = []

for epoch in range(number_of_epoch):
  epoch += 1
  inputs = Variable(torch.from_numpy(x_train).cuda())
  
  labels = Variable(torch.from_numpy(y_train).cuda())
  
  optimizer.zero_grad() 
  outputs = linear_model(inputs)
  loss = criterion(outputs, labels)
  loss.backward()
  optimizer.step()
  loss_in_each_epoch.append(loss.item())
  print('epoch {} resulted in a loss of {}'.format(epoch, loss.item()))

predicted = linear_model.forward(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()

plt.plot(x_train, y_train, 'ro', label = 'true value', alpha = 0.5)
plt.plot(x_train, predicted, label = 'predicted value', alpha = 0.5)
plt.legend()
plt.show()

epoch=[i for i in range(1,1001)]
plt.plot(epoch,loss_in_each_epoch)
plt.title("Loss over epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

"""# Testing performance of the model"""

x_test = torch.tensor([[100.0],[200.0],[300.0],[400.0]])

x_test.shape

real_values = x_test*1.5+13

real_values

predicted = linear_model.forward(Variable((x_test).cuda()))

predicted
