import torch

from src.models import OlympicForecastModel


def _edition(year, semis=True):
    return {
        "year": year,
        "sex_id": torch.tensor(0),
        "stages": {
            "HEATS": torch.randn(12, 5),
            "SEMIFINALS": torch.randn(8, 5) if semis else torch.zeros(0, 5),
            "FINALS": torch.randn(3, 5),
        },
        "presence": torch.tensor([1.0, float(semis), 1.0]),
        "count_features": torch.zeros(25),
        "year_features": torch.tensor([(year - 1986) / 10.0, 0.2]),
    }


def test_full_model_shapes_and_size():
    model = OlympicForecastModel()
    context = [_edition(1986, False), _edition(1991, False), _edition(1994, False)]
    queries = torch.tensor([[0.0, float(rank)] for rank in range(1, 17)] +
                           [[1.0, float(rank)] for rank in range(1, 9)] +
                           [[2.0, float(rank)] for rank in range(1, 4)])
    output = model(context, queries, 1.4, 0.4, 0)
    assert output.shape == (27,)
    assert 20_000 <= model.parameter_count() <= 80_000

