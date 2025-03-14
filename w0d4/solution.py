#%%
import torch as t
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fancy_einsum import einsum
from typing import Union, Optional, Callable
import numpy as np
from einops import rearrange
from tqdm.notebook import tqdm
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
import wandb

#%%
def plot_results(loss_list, accuracy_list):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(y=loss_list, name="Training loss"))
    fig.update_xaxes(title_text="Num batches observed")
    fig.update_yaxes(title_text="Training loss", secondary_y=False)
    # This next bit of code plots vertical lines corresponding to the epochs
    if len(accuracy_list) > 1:
        for idx, epoch_start in enumerate(np.linspace(0, len(loss_list), len(accuracy_list), endpoint=False)):
            fig.add_vline(x=epoch_start, line_width=3, line_dash="dash", annotation_text=f"Epoch {idx}", annotation_position="top right")
        fig.add_trace(
            go.Scatter(y=accuracy_list, x=np.linspace(0, len(loss_list), len(accuracy_list)), mode="lines", name="Accuracy"),
            secondary_y=True
        )
    fig.update_layout(template="simple_white", title_text="Training loss & accuracy on CIFAR10")
    fig.show()

def show_cifar_images(trainset, rows=3, cols=5):
    
    img = trainset.data[:rows*cols]
    fig = px.imshow(img, facet_col=0, facet_col_wrap=cols)
    for i, j in enumerate(np.arange(rows*cols).reshape(rows, cols)[::-1].flatten()):
            fig.layout.annotations[i].text = trainset.classes[trainset.targets[j]]
    fig.show()

device = "cuda" if t.cuda.is_available() else "cpu"
# %%
cifar_mean = [0.485, 0.456, 0.406]
cifar_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar_mean, std=cifar_std)
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

show_cifar_images(trainset, rows=3, cols=5)
# %%
def train(trainset, testset, epochs: int, loss_fn: Callable, batch_size: int, lr: float) -> tuple[list, list]:

    model = torchvision.models.resnet34(weights="DEFAULT").to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    loss_list = []
    accuracy_list = []

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    for epoch in range(epochs):

        progress_bar = tqdm(trainloader)

        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")

        with t.inference_mode():

            accuracy = 0
            total = 0

            for (x, y) in testloader:

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_predictions = y_hat.argmax(1)
                accuracy += (y_predictions == y).sum().item()
                total += y.size(0)

            accuracy_list.append(accuracy/total)

        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}, accuracy is {accuracy}/{total}")

    filename = "./w0d3_resnet.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)

    plot_results(loss_list, accuracy_list)
    return loss_list, accuracy_list

epochs = 1
loss_fn = nn.CrossEntropyLoss()
batch_size = 8
lr = 0.001

loss_list, accuracy_list = train(
    trainset, testset, epochs, loss_fn, batch_size, lr
)
#### W and b
# %%
def train_wandb(trainset, testset, epochs: int, loss_fn: Callable, batch_size: int, lr: float) -> None:

    config_dict = {
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
    }
    wandb.init(project="w2d1_resnet", config=config_dict)

    model = torchvision.models.resnet34(weights="DEFAULT").to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    examples_seen = 0
    start_time = time.time()

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)

    for epoch in range(epochs):

        progress_bar = tqdm(trainloader)

        model.train()
        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")

            examples_seen += len(x)
            wandb.log({"train_loss": loss, "elapsed": time.time() - start_time}, step=examples_seen)

        model.eval()
        with t.inference_mode():

            accuracy = 0
            total = 0

            for (x, y) in testloader:

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_predictions = y_hat.argmax(1)
                accuracy += (y_predictions == y).sum().item()
                total += y.size(0)

            wandb.log({"test_accuracy": accuracy/total}, step=examples_seen)

    filename = f"{wandb.run.dir}/model_state_dict.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)
    wandb.save(filename)

#%%
# train_wandb(trainset, testset, epochs, loss_fn, batch_size, lr)

#### Sweep
#%%
def train() -> None:

    wandb.init()

    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr

    model = torchvision.models.resnet34(weights="DEFAULT").to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    examples_seen = 0
    start_time = time.time()

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)

    for epoch in range(epochs):

        progress_bar = tqdm(trainloader)

        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")

            examples_seen += len(x)
            wandb.log({"train_loss": loss, "elapsed": time.time() - start_time}, step=examples_seen)


        with t.inference_mode():

            accuracy = 0
            total = 0

            for (x, y) in testloader:

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_predictions = y_hat.argmax(1)
                accuracy += (y_predictions == y).sum().item()
                total += y.size(0)

            wandb.log({"test_accuracy": accuracy/total}, step=examples_seen)

        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}, accuracy is {accuracy}/{total}")

    filename = f"{wandb.run.dir}/model_state_dict.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)
    wandb.save(filename)

sweep_config = {
    'method': 'random',
    'name': 'w2d1_resnet_sweep_2',
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
    'parameters': 
    {
        'batch_size': {'values': [64, 128, 256]},
        'epochs': {'min': 1, 'max': 3},
        'lr': {'max': 0.1, 'min': 0.0001, 'distribution': 'log_uniform_values'}
     }
}

# sweep_id = wandb.sweep(sweep=sweep_config, project='w2d1_resnet')

# wandb.agent(sweep_id=sweep_id, function=train, count=2)