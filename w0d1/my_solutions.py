import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from typing import Optional, Callable
import ipywidgets as wg
from fancy_einsum import einsum
import utils


#### Part 1: Fourier Transforms


def DFT_1d(arr : np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Returns the DFT of the array `arr`, using the equation above.
    """
    N = len(arr)
    if inverse:
        left_mat = np.array(
            [
                [np.exp((2j * np.pi * k * n) / N) / N for k in range(N)]
                for n in range(N)
            ]
        )
    else:
        omega = np.exp(-2j * np.pi / N)
        left_mat = np.array(
            [
                [omega ** (r * c) for c in range(N)]
                for r in range(N)
            ]
        )
    return left_mat @ arr


#%%
utils.test_DFT_func(DFT_1d) 
# %%


def test_DFT_example(
    DFT_1d_op=DFT_1d, x=np.array([1, 2-1j, -1j, -1 + 2j]), y=np.array([2, -2-2j, -2j, 4+4j])
) -> None:
    
    y_reconstructed_actual = DFT_1d_op(x)
    np.testing.assert_allclose(
        y_reconstructed_actual, y, atol=1e-10, err_msg="DFT failed for known example"
    )

#%%
test_DFT_example()

#%%
def integrate_function(func: Callable, x0: float, x1: float, n_samples: int = 1000):
    """
    Calculates the approximation of the Riemann integral of the function `func`, 
    between the limits x0 and x1.

    You should use the Left Rectangular Approximation Method (LRAM).
    """
    area = 0
    delta_x = float(x1 - x0) / n_samples
    for i in range(n_samples):
        x_i = x0 + i * delta_x
        y_i = func(x_i)
        area += y_i * delta_x
    return area

#%%
utils.test_integrate_function(integrate_function)

#%%
def integrate_product(
    func1: Callable, func2: Callable, x0: float, x1: float, n_samples: int = 1000
):
    """
    Computes the integral of the function x -> func1(x) * func2(x).
    """
    return integrate_function(
        func=lambda x: func1(x) * func2(x),
        x0=x0,
        x1=x1,
        n_samples=n_samples
    )

#%%
utils.test_integrate_product(integrate_product)
# %%
def calculate_fourier_series(func: Callable, max_freq: int = 50):
    """
    Calculates the fourier coefficients of a function, 
    assumed periodic between [-pi, pi].

    Your function should return ((a_0, A_n, B_n), func_approx), where:
        a_0 is a float
        A_n, B_n are lists of floats, with n going up to `max_freq`
        func_approx is the fourier approximation, as described above
    """

    a_0 = integrate_function(func, -np.pi, np.pi) / np.pi
    A_n = [
        integrate_product(func, lambda x: np.cos(n_A * x), -np.pi, np.pi) / np.pi
        for n_A in range(1, max_freq)
    ]
    B_n = [
        integrate_product(func, lambda x: np.sin(n_B * x), -np.pi, np.pi) / np.pi
        for n_B in range(1, max_freq)
    ]
    
    def func_approx(x: np.ndarray):
        y = np.array([a_0 / 2] * len(x))
        for n in range(1, max_freq):
            y += A_n[n - 1] * np.cos(n * x)
            y += B_n[n - 1] * np.sin(n * x)
        return y

    return ((a_0, A_n, B_n), func_approx)


#%%
step_func = lambda x: 1 * (x > 0)
utils.create_interactive_fourier_graph(calculate_fourier_series, func = step_func)


# %%
cubic_func = lambda x: x ** 3
utils.create_interactive_fourier_graph(calculate_fourier_series, func = cubic_func)
# %%
sawtooth_func = lambda x: x
utils.create_interactive_fourier_graph(calculate_fourier_series, func = sawtooth_func)
# %%
trig_func = lambda x: np.sin(3 * x) + np.cos(17 * x)
utils.create_interactive_fourier_graph(calculate_fourier_series, func = trig_func)

#### Part 2: Basic Neural Network

#%%
NUM_FREQUENCIES = 2
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6

#### 2.1: Numpy version

x = np.linspace(-np.pi, np.pi, 2000)
y = TARGET_FUNC(x)

x_cos = np.array([np.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = np.array([np.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

a_0 = np.random.randn()
A_n = np.random.randn(NUM_FREQUENCIES)
B_n = np.random.randn(NUM_FREQUENCIES)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):
    # compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    y_pred = a_0 + np.dot(A_n, x_cos) + np.dot(B_n, x_sin)

    # compute `loss`, which is the sum of squared error between `y` and `y_pred`
    loss = np.sum((y - y_pred) ** 2)

    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0, A_n.copy(), B_n.copy()])
        y_pred_list.append(y_pred)

    # compute gradients of coeffs with respect to `loss`
    nabla_a0 = np.sum(y_pred - y)
    nabla_a = np.sum(2 * (y_pred - y) * x_cos)
    nabla_b = np.sum(2 * (y_pred - y) * x_sin)


    # update weights using gradient descent (using the parameter `LEARNING_RATE`)
    a_0 -= LEARNING_RATE * nabla_a0
    A_n -= LEARNING_RATE * nabla_a
    B_n -= LEARNING_RATE * nabla_b

#%%
utils.visualise_fourier_coeff_convergence(
    x, y, y_pred_list, coeffs_list
)

# %%
import torch
x = torch.arange(5)
y1 = torch.Tensor(x.shape)
y2 = torch.Tensor(tuple(x.shape))
y3 = torch.Tensor(list(x.shape))
print(y1, y2, y3)

#### 2.2: pytorch version
#%%
x = torch.linspace(-torch.pi, torch.pi, 2000)
y = TARGET_FUNC(x)

x_cos = torch.stack(
    [torch.cos(n*x) for n in range(1, NUM_FREQUENCIES + 1)]
)
x_sin = torch.stack(
    [torch.sin(n*x) for n in range(1, NUM_FREQUENCIES + 1)]
)

a_0 = torch.randn(1)
A_n = torch.randn(NUM_FREQUENCIES)
B_n = torch.randn(NUM_FREQUENCIES)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):
    # compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    y_pred = (
        a_0 +
        torch.einsum('f,fx->x', [A_n, x_cos]) +
        torch.einsum('f,fx->x', [B_n, x_sin])
    )
 
    # compute `loss`, which is the sum of squared error between `y` and `y_pred`
    loss = torch.sum((y - y_pred) ** 2)

    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0.item(), A_n.numpy(), B_n.numpy()])
        y_pred_list.append(y_pred)

    # compute gradients of coeffs with respect to `loss`
    nabla_a0 = torch.sum(y_pred - y)
    nabla_a = torch.sum(2 * (y_pred - y) * x_cos)
    nabla_b = torch.sum(2 * (y_pred - y) * x_sin)


    # update weights using gradient descent (using the parameter `LEARNING_RATE`)
    a_0 -= LEARNING_RATE * nabla_a0
    A_n -= LEARNING_RATE * nabla_a
    B_n -= LEARNING_RATE * nabla_b

#%%
utils.visualise_fourier_coeff_convergence(
        x, y, y_pred_list, coeffs_list
    )

#### 2.3 - Autograd
# %%
import torch

a = torch.tensor(2, dtype=torch.float, requires_grad=True)
b = torch.tensor(3, dtype=torch.float, requires_grad=True)

Q = 3*a**3 - b**2
Q.backward()

# %%
# check if collected gradients are correct
assert 9*a**2 == a.grad
assert -2*b == b.grad
# %%

x = torch.linspace(-torch.pi, torch.pi, 2000)
y = TARGET_FUNC(x)

x_cos = torch.stack(
    [torch.cos(n*x) for n in range(1, NUM_FREQUENCIES + 1)]
)
x_sin = torch.stack(
    [torch.sin(n*x) for n in range(1, NUM_FREQUENCIES + 1)]
)

a_0 = torch.randn(1, requires_grad=True)
A_n = torch.randn(NUM_FREQUENCIES, requires_grad=True)
B_n = torch.randn(NUM_FREQUENCIES, requires_grad=True)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):
    # compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    y_pred = (
        a_0 +
        torch.einsum('f,fx->x', [A_n, x_cos]) +
        torch.einsum('f,fx->x', [B_n, x_sin])
    )
 
    # compute `loss`, which is the sum of squared error between `y` and `y_pred`
    loss = torch.sum((y - y_pred) ** 2)

    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([
            a_0.detach().item(), A_n.detach().numpy(), B_n.detach().numpy()
        ])
        y_pred_list.append(y_pred.detach())

    # compute gradients of coeffs with respect to `loss`
    loss.backward()

    # update weights using gradient descent (using the parameter `LEARNING_RATE`)
    with torch.no_grad():
        a_0 -= LEARNING_RATE * a_0.grad
        A_n -= LEARNING_RATE * A_n.grad
        B_n -= LEARNING_RATE * B_n.grad

    a_0.grad = None
    A_n.grad = None
    B_n.grad = None

#%%
utils.visualise_fourier_coeff_convergence(
        x, y, y_pred_list, coeffs_list
    )

#### 2.4 - Models
# %%
import torch.nn as nn
from einops import rearrange
x = torch.linspace(-torch.pi, torch.pi, 2000)
y = TARGET_FUNC(x)

x_cos = torch.stack(
    [torch.cos(n*x) for n in range(1, NUM_FREQUENCIES + 1)]
)
x_sin = torch.stack(
    [torch.sin(n*x) for n in range(1, NUM_FREQUENCIES + 1)]
)
x_net = torch.concat([x_cos, x_sin])
x_net = rearrange(x_net, 't x -> x t')


class Net(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2 * NUM_FREQUENCIES, 1)
        self.flat = nn.Flatten()

    def forward(self, x: torch.Tensor):
        return self.flat(self.fc(x)).squeeze()


net = Net()
y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):
    # compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    y_pred = net(x_net)

 
    # compute `loss`, which is the sum of squared error between `y` and `y_pred`
    loss = torch.sum((y - y_pred) ** 2)


    if step % 100 == 0:
        print(f"{loss = :.2f}")
        for param in net.parameters():
            coeffs_list.append(param.detach().numpy())
        y_pred_list.append(y_pred.detach())

    # compute gradients of coeffs with respect to `loss`
    loss.backward()

    # update weights using gradient descent (using the parameter `LEARNING_RATE`)
    with torch.no_grad():
        for param in net.parameters():
            param -= LEARNING_RATE * param.grad
    net.zero_grad()

#### 2.5 Optimizers
# %%
import torch.optim as optim
x = torch.linspace(-torch.pi, torch.pi, 2000)
y = TARGET_FUNC(x)

x_cos = torch.stack(
    [torch.cos(n*x) for n in range(1, NUM_FREQUENCIES + 1)]
)
x_sin = torch.stack(
    [torch.sin(n*x) for n in range(1, NUM_FREQUENCIES + 1)]
)
x_net = torch.concat([x_cos, x_sin])
x_net = rearrange(x_net, 't x -> x t')


class Net(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2 * NUM_FREQUENCIES, 1)
        self.flat = nn.Flatten()

    def forward(self, x: torch.Tensor):
        return self.flat(self.fc(x)).squeeze()


net = Net()
y_pred_list = []
coeffs_list = []
optimiser = optim.SGD(net.parameters(), lr=LEARNING_RATE)

for step in range(TOTAL_STEPS):
    # compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    y_pred = net(x_net)

 
    # compute `loss`, which is the sum of squared error between `y` and `y_pred`
    loss = torch.sum((y - y_pred) ** 2)


    if step % 100 == 0:
        print(f"{loss = :.2f}")
        for param in net.parameters():
            coeffs_list.append(param.detach().numpy())
        y_pred_list.append(y_pred.detach())

    # compute gradients of coeffs with respect to `loss`
    loss.backward()

    # update weights using gradient descent (using the parameter `LEARNING_RATE`)
    optimiser.step()
    net.zero_grad()

# %%
