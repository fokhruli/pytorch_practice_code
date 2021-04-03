import torch

# gradient of eqn. y = 9x^4 + 2x^3 + 3x^3 + 6x +1

x = torch.tensor(2.0, requires_grad=True) # must put require_grad = True
y = 9*x**4 + 2*x**3 + 3*x**3 + 6*x +1
y.backward() # calculate backprop
#print(x.grad) # printed gradient in term of x

# what about two variable?

x = torch.tensor(1.0, requires_grad=True)
z = torch.tensor(2.0, requires_grad=True)
y = x**2 + z**3

y.backward()
print(x.grad) # 2
print(z.grad) # 12