import torch
from torch.autograd import Variable

""" below is Variable hook """
# 1
'''
v = Variable(torch.Tensor([0, 0, 0]), requires_grad=True)
v.backward(torch.Tensor([1, 1, 1]))
print(v.grad.data)
'''

# 2 (hook is like a plugin to modify gradient of variables)
'''
v = Variable(torch.Tensor([0, 0, 0]), requires_grad=True)
u = Variable(torch.Tensor([0, 0, 1]), requires_grad=True)
# h = v.register_hook(lambda grad: grad * 2)  # double the gradient
h = u.register_hook(lambda x: x * 2)  # double the gradient
v.backward(torch.Tensor([1, 1, 1]))
print(v.grad.data)
h.remove()  # removes the hook
'''
