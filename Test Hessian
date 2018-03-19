### a test script (seems to work at least for the defined architecture) to compute Hessian
### achitecture and transform function need to be modified
### Hessian need to be modified according to the architecture



import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os.path
from torchvision import datasets, transforms, utils
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1=nn.Linear(784, 10,bias=False)
    def forward(self, x):
        x = x.view(-1,784)
        x = self.fc1(x)
        return x

model=Net()
model=model.cuda()

transform = transforms.Compose( [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] )

batch=64
trainset = datasets.MNIST(root='./data/MNIST_data', train=True,  download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)

testset = datasets.MNIST(root='./data/MNIST_data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)


def loss(model):
    res=0
    for i, (inputs,labels) in enumerate(trainloader, 0):
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        temp_loss = F.cross_entropy(outputs,labels,size_average=False)
        res+=temp_loss
    res.data[0]=res.data[0]/len(trainset)
    return res


def training(epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0)
    for i in range(epoch):
        for _, (inputs,labels) in enumerate(trainloader, 0):
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())       
            optimizer.zero_grad()       
            outputs = model(inputs)
            myloss = criterion(outputs, labels)
            myloss.backward()
            optimizer.step()

        
def Hessian(model):
    t0=time.time()
    free=10*784
    Hessian=np.zeros([free,free])
    model.zero_grad
    myloss=loss(model)
    dloss=torch.autograd.grad(myloss, model.parameters(), create_graph=True)
    k=0
    for v in dloss[0]:
        for i in range(len(v)):
            ddloss=torch.autograd.grad(v[i], model.parameters(),retain_graph=True)
            res=ddloss[0].view(1,-1).data
            Hessian[len(v)*k+i,:]=res
        print(time.time()-t0)
        k+=1
    return Hessian

# vp=np.linalg.svd(H,compute_uv=False)
# plt.plot(vp)
# plt.yscale('log')
# plt.show()
