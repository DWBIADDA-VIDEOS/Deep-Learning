#Autograd
import torch

x = torch.tensor(30.)
w = torch.tensor(40., requires_grad=True)
b = torch.tensor(50., requires_grad=True)

x.is_leaf


z = w * x + b
z

z.is_leaf

"""# Taking gradient"""

z.backward()

"""#$\frac{\delta z}{\delta w}$"""

w.grad

"""#$\frac{\delta z}{\delta b}$"""

b.grad

