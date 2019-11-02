

import torch
import torchvision 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time

trainset = dsets.MNIST(root='./my_data', train=True,
                                        download=True, transform=transforms.ToTensor())


testset = dsets.MNIST(root='./my_data', train=False,
                                       download=True, transform=transforms.ToTensor())

batch_size = 1 # min(1) to maximum(60,000 -> 10,000)
num_epochs = 10
print("Number of epochs : ",num_epochs)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)

len(trainloader) #60,000 / 100

len(testloader)

# SIMPLE MODEL DEFINITION
class Simple_MLP(nn.Module):
    def __init__(self, size_list):
        super(Simple_MLP, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.size_list[0]) # Flatten the input
        return self.net(x)

def train_epoch(model,train_loader,optimizer, criterion):
    model.train()
    model.to(device)
    running_loss = 0.0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):   
        optimizer.zero_grad()   
        data = data.to(device)
        target = target.long().to(device)

        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    return running_loss,(end_time - start_time)

def test_model(model, test_loader, criterion):
    with torch.no_grad():
        model.eval()
        model.to(device)

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(test_loader):   
            data = data.to(device)
            target = target.long().to(device)

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            loss = criterion(outputs, target).detach()
            running_loss += loss.item()


        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc

model = Simple_MLP([784, 100, 10])

criterion = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

"""# BGD"""

loss_train_time_bgd = []
loss_acc_test_bgd = []

for i in range(num_epochs):
  trep = train_epoch(model, trainloader, optimizer, criterion)
  temo = test_model(model, testloader, criterion)
  loss_train_time_bgd.append(trep)
  loss_acc_test_bgd.append(temo)

test_loss_bgd = []
test_accuracy_bgd = []

for i in loss_acc_test_bgd:
  test_loss_bgd.append(i[0])
  test_accuracy_bgd.append(i[1])

train_loss_bgd = []
train_time_bgd =[]

for i in loss_train_time_bgd:
  train_loss_bgd.append(i[0])
  train_time_bgd.append(i[1])

print(test_loss_bgd)
print(test_accuracy_bgd)
print(train_loss_bgd)
print(train_time_bgd)

"""# SGD"""

loss_train_time_sgd = []
loss_acc_test_sgd = []

for i in range(num_epochs):
  trep = train_epoch(model, trainloader, optimizer, criterion)
  temo = test_model(model, testloader, criterion)
  loss_train_time_sgd.append(trep)
  loss_acc_test_sgd.append(temo)

test_loss_sgd = []
test_accuracy_sgd = []

for i in loss_acc_test_sgd:
  test_loss_sgd.append(i[0])
  test_accuracy_sgd.append(i[1])

train_loss_sgd = []
train_time_sgd =[]

for i in loss_train_time_sgd:
  train_loss_sgd.append(i[0])
  train_time_sgd.append(i[1])

print(test_loss_sgd)
print(test_accuracy_sgd)
print(train_loss_sgd)
print(train_time_sgd)

"""# GD"""

loss_train_time_gd = []
loss_acc_test_gd = []

for i in range(num_epochs):
  trep = train_epoch(model, trainloader, optimizer, criterion)
  temo = test_model(model, testloader, criterion)
  loss_train_time_gd.append(trep)
  loss_acc_test_gd.append(temo)

test_loss_gd = []
test_accuracy_gd = []

for i in loss_acc_test_gd:
  test_loss_gd.append(i[0])
  test_accuracy_gd.append(i[1])

train_loss_gd = []
train_time_gd =[]

for i in loss_train_time_gd:
  train_loss_gd.append(i[0])
  train_time_gd.append(i[1])

print(test_loss_gd)
print(test_accuracy_gd)
print(train_loss_gd)
print(train_time_gd)

"""# Plotting Test Accuracies"""

epoch=list(range(1,num_epochs+1))
plt.figure(figsize=(9,9))


plt.plot(epoch,test_accuracy_sgd, label = "Stoachastic Gradient Descent", alpha = 0.6)
plt.plot(epoch,test_accuracy_gd, label = "Gradient Descent", alpha = 0.6)
plt.plot(epoch,test_accuracy_bgd, label = "Batch Gradient Descent", alpha = 0.6)
plt.title('Testing Dataset Performance')
plt.ylabel('Test Accuracy')
plt.xlabel('epoch')

plt.legend()
plt.show()

"""# Plotting Run Time"""

#epoch=list(range(1,num_epochs+1))
plt.figure(figsize=(9,9))


plt.plot(epoch,train_time_sgd, label = "Stoachastic Gradient Descent", alpha = 0.6)
plt.plot(epoch,train_time_gd, label = "Gradient Descent", alpha = 0.6)
plt.plot(epoch,train_time_bgd, label = "Batch Gradient Descent", alpha = 0.6)
plt.title('Testing Dataset Performance')
plt.ylabel('Run Time')
plt.xlabel('Epoch')

plt.legend()
plt.show()

"""# Plotting Traiining Loss"""

plt.figure(figsize=(9,9))

plt.plot(epoch,train_loss_sgd, label = "Stoachastic Gradient Descent", alpha = 0.6)
plt.plot(epoch,train_loss_gd, label = "Gradient Descent", alpha = 0.6)
plt.plot(epoch,train_loss_bgd, label = "Batch Gradient Descent", alpha = 0.6)
plt.title('Training Dataset Performance')
plt.ylabel('Training Loss')
plt.xlabel('Epoch')

plt.legend()
plt.show()

