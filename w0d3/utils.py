from plotly.subplots import make_subplots
import plotly.graph_objects as go
import torch as t
from IPython.display import display
import torchvision
import pandas as pd

def plot_loss_and_accuracy(loss_list, accuracy_list):

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(y=loss_list, name="Loss"), secondary_y=False)

    n_batches = len(loss_list) // len(accuracy_list)
    x = list(range(n_batches-1, len(loss_list), n_batches))
    fig.add_trace(go.Scatter(y=accuracy_list, x=x, name="Accuracy"), secondary_y=True)

    fig.update_layout(
        title_text="CNN training loss & test accuracy",
        template="simple_white", 
        xaxis_range=[0, len(loss_list)], 
        yaxis2_range=[0.9, 1],
        yaxis2_tickformat=".0%", 
        hovermode="x unified"
    )

    fig.show()
    return fig

def test_batchnorm2d_module(BatchNorm2d):
    """The public API of the module should be the same as the real PyTorch version."""
    num_features = 2
    bn = BatchNorm2d(num_features)
    assert bn.num_features == num_features
    assert isinstance(bn.weight, t.nn.parameter.Parameter), f"weight has wrong type: {type(bn.weight)}"
    assert isinstance(bn.bias, t.nn.parameter.Parameter), f"bias has wrong type: {type(bn.bias)}"
    assert isinstance(bn.running_mean, t.Tensor), f"running_mean has wrong type: {type(bn.running_mean)}"
    assert isinstance(bn.running_var, t.Tensor), f"running_var has wrong type: {type(bn.running_var)}"
    assert isinstance(bn.num_batches_tracked, t.Tensor), f"num_batches_tracked has wrong type: {type(bn.num_batches_tracked)}"
    print("All tests in `test_batchnorm2d_module` passed!")

def test_batchnorm2d_forward(BatchNorm2d):
    """For each channel, mean should be very close to 0 and std kinda close to 1 (because of eps)."""
    num_features = 2
    bn = BatchNorm2d(num_features)
    assert bn.training
    x = t.randn((100, num_features, 3, 4))
    out = bn(x)
    assert x.shape == out.shape
    t.testing.assert_close(out.mean(dim=(0, 2, 3)), t.zeros(num_features))
    t.testing.assert_close(out.std(dim=(0, 2, 3)), t.ones(num_features), atol=1e-3, rtol=1e-3)
    print("All tests in `test_batchnorm2d_forward` passed!")

def test_batchnorm2d_running_mean(BatchNorm2d):
    """Over repeated forward calls with the same data in train mode, the running mean should converge to the actual mean."""
    bn = BatchNorm2d(3, momentum=0.6)
    assert bn.training
    x = t.arange(12).float().view((2, 3, 2, 1))
    mean = t.tensor([3.5000, 5.5000, 7.5000])
    num_batches = 30
    for i in range(num_batches):
        bn(x)
        expected_mean = (1 - (((1 - bn.momentum) ** (i + 1)))) * mean
        t.testing.assert_close(bn.running_mean, expected_mean)
    assert bn.num_batches_tracked.item() == num_batches

    # Large enough momentum and num_batches -> running_mean should be very close to actual mean
    bn.eval()
    actual_eval_mean = bn(x).mean((0, 2, 3))
    t.testing.assert_close(actual_eval_mean, t.zeros(3))
    print("All tests in `test_batchnorm2d_running_mean` passed!")


def compare_my_resnet_to_pytorch(myresnet):
    
    their_state = torchvision.models.resnet34().state_dict().items()
    your_state = myresnet.state_dict().items()
    
    df = pd.DataFrame.from_records(
        [(tk, tuple(tv.shape), mk, tuple(mv.shape)) for ((tk, tv), (mk, mv)) in zip(their_state, your_state)],
        columns=["their name", "their shape", "your name", "your shape"],
    )
    with pd.option_context("display.max_rows", None):  # type: ignore
        display(df)