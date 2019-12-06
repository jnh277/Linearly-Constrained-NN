import torch
import torch.autograd as ag
import torch.nn as nn

def func(x, c0=1.0, c1=1.0, c2=1.0):
    return c2*x.pow(2) + c1*x + c0

x = torch.ones(5,2)

c0 = torch.tensor([[1.0]], requires_grad=True)
c1 = torch.tensor([[2.0]], requires_grad=True)
c2 = torch.tensor([[3.0]], requires_grad=True)
c3 = torch.tensor([[1.5]], requires_grad=True)
c4 = torch.tensor([[1.8]], requires_grad=True)
x.requires_grad = True
y = func(x, c0=c0,c1=c1, c2=c2)






test = ag.grad(outputs=y, inputs=x, create_graph=True, grad_outputs=torch.ones(y.size()),
               retain_graph=True, only_inputs=True)[0]


def func2(x, c0, c1, c2, c3, c4):
    return c4*x[:, 0]*x[:, 0] + c3*x[:, 1]*x[:, 1] + c2*x[:, 0]*x[:, 1] + c1 * x[:, 1] + c0 * x[:, 0]

f = func2(x, c0, c1, c2, c3, c4)

g = ag.grad(outputs=f, inputs=x, create_graph=True, grad_outputs=torch.ones(f.size()),
               retain_graph=True, only_inputs=True)[0]

hx = ag.grad(outputs=g[:,0], inputs=x, create_graph=True, grad_outputs=torch.ones(g[:,0].size()),
               retain_graph=True, only_inputs=True)[0]
hy = ag.grad(outputs=g[:,1], inputs=x, create_graph=True, grad_outputs=torch.ones(g[:,1].size()),
               retain_graph=True, only_inputs=True)[0]

h = torch.cat((hx[:,0].unsqueeze(1), hx[:,1].unsqueeze(1),hy[:,1].unsqueeze(1)),0)

loss = h.mean()
loss.backward()
dc0 = c0.grad
dc1 = c1.grad
dc2 = c2.grad
dc3 = c3.grad
dc4 = c4.grad

# gr = torch.cat((g[:,0].unsqueeze(1), g[:,1].unsqueeze(1)),0)
#
# h = ag.grad(outputs=gr, inputs=x, create_graph=True, grad_outputs=torch.ones(gr.size()),
#                retain_graph=True, only_inputs=True)[0]

# y.backward(retain_graph=True)
#
# dc0 = c0.grad
# dc1 = c1.grad
# would then need to zero gradients?

# test.backward()
# dxdc0 = c0.grad # this grad doesn't exist
# dxdc1 = c1.grad
# dxdc2 = c2.grad


# model = torch.nn.Sequential(nn.Linear(1,5),nn.Tanh(),nn.Linear(5,1))

# class DerivNet(torch.nn.Module):
#     def __init__(self, base_net):
#         super(DerivNet, self).__init__()
#         self.base_net = base_net
#
#     def forward(self, x):
#         x.requires_grad = True
#         y = self.base_net(x)
#         dydx = ag.grad(outputs=y, inputs=x, create_graph=True, grad_outputs=torch.ones(y.size()),
#                        retain_graph=True, only_inputs=True)[0]
#
#         return y, dydx
#
#
# x3 = torch.ones(5,1)
# grad_model = DerivNet(model)
# y, dydx = grad_model(x3)
#
# loss = y.sum()
# loss.backward()
