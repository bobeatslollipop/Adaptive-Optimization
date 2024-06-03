import torch
import torch.autograd as autograd

a = torch.tensor([2., 3.], requires_grad=True)

Q = 3 * (a**3)
Q_sum = Q.sum()

print(Q)
print(Q_sum)

# Compute the first derivative of Q_sum with respect to a
first_derivative = autograd.grad(Q_sum, a, create_graph=True)[0]
print(first_derivative)

# Compute the second derivative of Q_sum with respect to a
second_derivative = autograd.grad(first_derivative, a, grad_outputs=torch.ones_like(a))[0]
print(second_derivative)

