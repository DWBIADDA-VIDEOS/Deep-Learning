#converting numpy to tensor using pytorch

!pip3 install torch torchvision

import torch
print(torch.__version__)

torch.cuda.is_available()

torch.version.cuda

import torch
x = torch.rand(3, 3)
print(x)

x = torch.zeros(3, 3)
print(x)

a = torch.ones(5)
print(a)

x = torch.Tensor(3, 3)
print(x)

x = torch.tensor([[1,3], [5, 7]])
x

"""#Tensor vs numpy"""

import numpy as np
import time

n = 1000

m1 = np.random.rand(n,n).astype(np.float32)
m2 = np.random.rand(n,n).astype(np.float32)

start = time.time()
result = m1.dot(m2)
end = time.time()

print("Time for numpy operation  : ",(end-start))


m3 = torch.rand(n,n).cuda()
m4 = torch.rand(n,n).cuda()

start = time.time()
result = torch.mm(m3,m4)
end = time.time()

print("Time for tensor operation : ",(end-start))

type(m1)

type(m2)

type(m3)

type(m4)

m1 = torch.from_numpy(m1)
m2 = torch.from_numpy(m2)

start = time.time()
result = torch.mm(m1,m2)
end = time.time()

print("Time for tensor operation : ",(end-start))

type(m1)

type(m2)

m3 = m3.cpu().data.numpy()
m4 = m4.cpu().data.numpy()

start = time.time()
result = m3.dot(m4)
end = time.time()

print("Time for tensor operation : ",(end-start))

type(m3)

type(m4)

