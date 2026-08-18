import torch

from src.models.deepsets import masked_statistics


def test_padding_does_not_affect_masked_statistics():
    real = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    padded = torch.cat([real, torch.tensor([[999.0, -999.0]])])
    actual = masked_statistics(padded, torch.tensor([True, True, False]))
    expected = masked_statistics(real)
    assert torch.allclose(actual, expected)


def test_empty_stage_returns_zero_statistics():
    output = masked_statistics(torch.zeros((0, 32)))
    assert output.shape == (96,)
    assert torch.count_nonzero(output) == 0

