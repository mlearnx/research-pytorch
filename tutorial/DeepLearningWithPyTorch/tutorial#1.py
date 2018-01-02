"""
Tutorial #1
"""

from __future__ import division;
from __future__ import print_function;
from __future__ import absolute_import;

import torch; 
import numpy as np;

######################
## Construct tensor ##
######################
# Construct a 5x3 matrix, uninitialized:
x = torch.Tensor(5, 3)
print(x)

# Construct a randomly initialized matrix
x = torch.rand(5, 3)
print(x)

# Get its size
print(x.size())

##############
## Addition ##
##############
# Addition
y = torch.rand(5, 3)
print(x + y)
# OR
print(torch.add(x, y))

# Addition: giving an output tensor
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)

# Addition: in-place
# adds x to y
y.add_(x)
print(y)

############################################
## Converting torch Tensor to numpy Array ##
############################################
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

############################################
## Converting numpy Array to torch Tensor ##
############################################
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

##################
## CUDA Tensors ##
##################
# let us run this cell only if CUDA is available
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x + y




