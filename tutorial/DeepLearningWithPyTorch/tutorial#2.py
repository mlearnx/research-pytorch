"""
Tutorial #2 
Autograd: automatic differentiation
"""

from __future__ import division;
from __future__ import print_function;
from __future__ import absolute_import;

import torch; 
import numpy as np;

#######################
## autograd.Variable ##
#######################
# You can access the raw tensor through the .data attribute, while the gradient w.r.t. this variable is accumulated into .grad.
from torch.autograd import Variable

# Create a variable:
x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)

# Addition: y was created as a result of an operation, so it has a grad_fn.
y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean(); # out = 1/4 sum(z), where z = 3(x+2)^2
print(z, out)

###############
## Gradients ##
###############
out.backward()
print(x.grad); # dout/dx

# example
x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)
print(x.grad)

