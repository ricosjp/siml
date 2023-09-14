import pytest
import torch

from siml.siml_variables import siml_tensor_variables


@pytest.fixture
def make_default_tensors():
    vals = {
        "a": torch.tensor([1.0]),
        "b": [
            torch.tensor([2.0]),
            torch.tensor([3.0])
        ],
        "c": {
            "d": [
                torch.tensor([[2.0], [7.0]]),
                torch.tensor([3.0, 6.0, 6.0])
            ],
            "e": torch.tensor([9.0])
        }
    }
    return vals


def test__get_value(make_default_tensors):
    vals = make_default_tensors
    siml_vals = siml_tensor_variables(vals)
    out = siml_vals.get_values()

    assert type(out["a"]) == torch.Tensor
    for v in out["b"]:
        assert type(v) == torch.Tensor

    assert type(out["c"]["e"]) == torch.Tensor
    for v in out["c"]["d"]:
        assert type(v) == torch.Tensor


def test__send(make_default_tensors):
    vals = make_default_tensors
    siml_vals = siml_tensor_variables(vals)
    device = torch.device('cuda:0')
    out = siml_vals.send(device).get_values()

    assert out["a"].device == device
    for v in out["b"]:
        assert v.device == device

    assert out["c"]["e"].device == device
    for v in out["c"]["d"]:
        assert v.device == device


@pytest.mark.parametrize('slices, shapes', [
    ((slice(None), slice(1, 2), slice(None)), (3, 4, 2)),
    ((slice(2), slice(None), slice(2)), (3, 4, 2))
])
def test__get_slice(slices, shapes):
    sample0 = torch.rand(shapes)
    sample1 = torch.rand(shapes)
    sample2 = torch.rand(shapes)
    val = {
        "a": sample0,
        "b": [
            sample1,
            sample2
        ]
    }
    out = siml_tensor_variables(val).slice(slices).get_values()

    assert torch.equal(sample0[slices], out["a"])
    assert torch.equal(sample1[slices], out["b"][0])
    assert torch.equal(sample2[slices], out["b"][1])
