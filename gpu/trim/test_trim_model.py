import torch
from gpu.trim.trim_model import SlimLinear, trim_model


def test_selected_columns_and_scattered_rows():
    torch.manual_seed(7)
    w = torch.randn(4608, 4608, dtype=torch.bfloat16)
    w[:, 2560:] = 0
    w[:, 2732] = torch.randn(4608, dtype=torch.bfloat16)
    w[2560:] = 0
    x = torch.randn(2, 3, 4608, dtype=torch.bfloat16)
    dense = torch.nn.functional.linear(x, w)
    slim = SlimLinear(w, select_input=True, trim_output=True)(x)
    assert torch.equal(dense, slim)


def test_trim_replaces_only_selected_layers():
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4608, 4608, bias=False)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Layer() for _ in range(13)])
    m = Model()
    stats = trim_model(m)
    assert stats["slim_linear_count"] == 1
    assert isinstance(m.model.layers[11].proj, torch.nn.Linear)
    assert isinstance(m.model.layers[12].proj, SlimLinear)
