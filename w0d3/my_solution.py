#%%
import torch as t
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import PIL
from PIL import Image
import json
from pathlib import Path
from typing import Union, Tuple, Callable, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import utils

#%%
import torch as t
import torch.nn as nn
import torch.nn.functional as F

#%%
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.max1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.max2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = F.relu(self.conv1(x))
        x = self.max1(x)
        x = F.relu(self.conv2(x))
        x = self.max2(x)
        x = t.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = ConvNet()
# print(model)
# %%
from tqdm.notebook import tqdm
import time

for i in tqdm(range(100)):
    time.sleep(0.01)


#%%
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
#%%
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# %%
epochs = 3
loss_fn = nn.CrossEntropyLoss()
batch_size = 128

MODEL_FILENAME = "./w1d2_convnet_mnist.pt"
device = "cuda" if t.cuda.is_available() else "cpu"


def train_convnet(trainloader: DataLoader, epochs: int, loss_fn: Callable) -> list:
    '''
    Defines a ConvNet using our previous code, and trains it on the data in trainloader.
    '''

    model = ConvNet().to(device).train()
    optimizer = t.optim.Adam(model.parameters())
    loss_list = []

    for epoch in range(epochs):

        progress_bar = tqdm(trainloader)
        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(loss.item())

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")

    print(f"Saving model to: {MODEL_FILENAME}")
    t.save(model, MODEL_FILENAME)
    return loss_list

# loss_list = train_convnet(trainloader, epochs, loss_fn)

#%%
# import plotly.express as px
# fig = px.line(y=loss_list, template="simple_white")
# fig.update_layout(title="Cross entropy loss on MNIST", yaxis_range=[0, max(loss_list)])
# fig.show()
# %%
testset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
testloader = DataLoader(testset, batch_size=64, shuffle=True)

def train_convnet_and_test(trainloader: DataLoader, testloader: DataLoader, epochs: int, loss_fn: Callable) -> Tuple[list, list]:
    '''
    Defines a ConvNet using our previous code, and trains it on the data in trainloader.

    Returns tuple of (loss_list, accuracy_list), where accuracy_list contains the fraction of accurate classifications on the test set, at the end of each epoch.
    '''
    model = ConvNet().to(device).train()
    optimizer = t.optim.Adam(model.parameters())
    loss_list = []
    accuracy_list = []

    for epoch in range(epochs):

        progress_bar = tqdm(trainloader)
        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(loss.item())

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")

        correct = 0
        total = 0
        for (x, y) in tqdm(testloader):

            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            _, pred = t.max(y_hat, 1)
            correct += (y == pred).sum().item()
            total += len(y)
        
        accuracy_list.append(correct / total)


    print(f"Saving model to: {MODEL_FILENAME}")
    t.save(model, MODEL_FILENAME)
    return loss_list, accuracy_list

loss_list, accuracy_list = train_convnet_and_test(trainloader, testloader, epochs, loss_fn)

#%%
import importlib
importlib.reload(utils)
utils.plot_loss_and_accuracy(loss_list, accuracy_list)

#### Part 2: Assembling Resnet
#%%
import torch as t
import torch.nn as nn
import utils
import numpy as np
# %%
class BatchNorm2d(nn.Module):
    running_mean: t.Tensor         # shape: (num_features,)
    running_var: t.Tensor          # shape: (num_features,)
    num_batches_tracked: t.Tensor  # shape: ()

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        super().__init__()
        self.num_features = num_features
        self.weight = (t.rand((num_features,)) * 2 - 1) / np.sqrt(num_features)
        self.bias = (t.rand((num_features,)) * 2 - 1) / np.sqrt(num_features)
        self.register_buffer('running_mean', t.zeros(num_features))
        self.register_buffer('running_var', t.zeros(num_features))
        self.momentum = momentum
        self.eps = eps


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        x_mean = x.mean(dim=[0, 2, 3])
        x_var = x.var(dim=[0, 2, 3])
        self.running_mean = self.momentum * x_mean + (1 - self.momentum) * self.running_mean
        self.running_var = self.momentum * x_var + (1 - self.momentum) * self.running_var
        x_scaled = (x - x_mean) / t.sqrt(x_var + self.eps)
        y = x_scaled * self.weight + self.bias
        return y

    def extra_repr(self) -> str:
        return f'weight={self.weight} bias={self.bias} mu={self.running_mean} var={self.running_var}'

#%%
utils.test_batchnorm2d_module(BatchNorm2d)
utils.test_batchnorm2d_forward(BatchNorm2d)
utils.test_batchnorm2d_running_mean(BatchNorm2d)

# %%
