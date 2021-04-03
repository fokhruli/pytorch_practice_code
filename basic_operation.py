import torch

############################ tensor initialization #####################################

device = "cuda" if torch.cuda.is_available() else "cpu"

tensorr = torch.tensor([[1,2,3], [7,8,9]],
                       dtype=torch.float32, device=device)

#print(tensorr.device)

############################ tensor other operation #####################################

z = torch.empty(size=(3,3))
z = torch.zeros((3,3))
z = torch.rand((3,3))
z = torch.ones((3,3))
z = torch.eye(5,5)
z = torch.arange(start=1, end=100, step=6)
z = torch.linspace(start=0.1, end=2, steps=45)
z = torch.diag(torch.tensor([1,2,3]))
z = torch.empty(size=(1,7)).normal_(mean=0, std=1)
z = torch.empty(size=(1,6)).uniform_(0, 1)
#print(z)

#################################### tensor --> numpy array #################################

import numpy as np

array = np.array([[1,2,3], [4,5,6]])
tensor = torch.from_numpy(array)

array_back = tensor.numpy()


#################################### tensor math and inplace #########################################


# add

x= torch.tensor([1,2,3])
y= torch.tensor([3,2,1])

z= x+y


# subtract

z = x-y
# devide

z= x/y

# inplace operation
y = torch.empty(3)
y.add_(z)

# exponnetial

y= x.pow(2)

y = x ** 2
#print(y)

################## mat mul ###########################

# normal matrix multification

x = torch.rand(size=(2,5))
y = torch.rand(size=(5,2))

z = torch.mm(x,y)
z = x.mm(y)

# matrix exponent

z = z.matrix_power(3)

# elementwise mul.
x= torch.tensor([1,2,3])
y= torch.tensor([3,2,1])
z = x*y

# dot product
z = torch.dot(x,y)

# Batch mat mut

batch = 32
m = 50
n = 40
p = 30
x = torch.rand(size=(batch,m,n))
y = torch.rand(size=(batch,n,p))

z = torch.bmm(x,y)
# ex

# "  sum, max, min, argmax, argmin, abs, eq, sort " function also work

# clamp a tensor

y= torch.tensor([[-0.5,0.2,1], [3,2,-0.1], [30,20,-1]])

y = torch.clamp(y, min=0, max=10) # important op.

# tensor .all and .any function

y = torch.tensor([1,0,1,1], dtype=torch.bool)
# print(torch.any(y)) # true
# print(torch.all(y)) # False


################################# tensor indexing #############################

x =  torch.arange(10)
index = torch.tensor([2,3,5])
x = torch.rand(5,6)
row = torch.tensor([4,0])
col = torch.tensor([4,0])
x =  torch.arange(10)
#print(x[(x<2) | (x>7)])
#print(x[x.remainder(2) == 0])
y = torch.rand(size=(batch,n,p))
print(y.shape)
z = y.reshape(-1, batch,n,p)
print(z.shape)