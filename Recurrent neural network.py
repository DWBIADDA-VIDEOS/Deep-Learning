
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SingleRNN(nn.Module):
    def __init__(self, input_size, neurons):
        super(SingleRNN, self).__init__()
        
        self.Wx = torch.randn(input_size, neurons) 
        self.Wy = torch.randn(neurons, neurons) 
        
        self.b = torch.zeros(1, neurons) 
    
    def forward(self, X0, X1):
        self.Y0 = torch.tanh(torch.mm(X0, self.Wx) + self.b)
        
        self.Y1 = torch.tanh(torch.mm(self.Y0, self.Wy) +
                            torch.mm(X1, self.Wx) + self.b)
        
        return self.Y0, self.Y1

input_size  = 2
neurons = 1

X0_batch = torch.tensor([[1,2], 
                         [3,4], 
                         [5,6], 
                         [7,8]],
                        dtype = torch.float) 

X1_batch = torch.tensor([[1.4,5.6], 
                         [2.3,4.5], 
                         [2.6,7.8], 
                         [2,4]],
                        dtype = torch.float) 

model = SingleRNN(input_size, neurons)

Y0_val, Y1_val = model(X0_batch, X1_batch)

Y0_val

Y1_val

