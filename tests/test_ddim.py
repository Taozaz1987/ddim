from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def _make_sampler():
    diffusion = pytest.importorskip("guided_diffusion.ddim")
    DDIMSampler = diffusion.DDIMSampler
    sampler = object.__new__(DDIMSampler)
    sampler._wrap_model = lambda m: m  # noqa: SLF001
    sampler._scale_timesteps = lambda t: t  # noqa: SLF001
    return sampler


def test_get_et_accepts_single_channel_output():
    torch = pytest.importorskip("torch")
    sampler = _make_sampler()
    x = torch.zeros((2, 3, 4, 4))
    model_output = torch.ones_like(x)

    def model_fn(x_in, *_args, **_kwargs):
        return model_output

    result = sampler._get_et(model_fn, x, torch.tensor([0, 1]), {})  # noqa: SLF001
    assert torch.equal(result, model_output)


def test_get_et_splits_double_channel_output():
    torch = pytest.importorskip("torch")
    sampler = _make_sampler()
    x = torch.zeros((1, 2, 4, 4))
    first_half = torch.full_like(x, 2.0)
    second_half = torch.full_like(x, -1.0)
    model_output = torch.cat([first_half, second_half], dim=1)

    def model_fn(x_in, *_args, **_kwargs):
        return model_output

    result = sampler._get_et(model_fn, x, torch.tensor([0]), {})  # noqa: SLF001
    assert torch.equal(result, first_half)


def test_get_et_rejects_unexpected_channel_count():
    torch = pytest.importorskip("torch")
    sampler = _make_sampler()
    x = torch.zeros((1, 2, 4, 4))
    model_output = torch.zeros((1, 3, 4, 4))

    def model_fn(x_in, *_args, **_kwargs):
        return model_output

    with pytest.raises(ValueError):
        sampler._get_et(model_fn, x, torch.tensor([0]), {})  # noqa: SLF001


def test_seismic_config_defaults_to_learn_sigma_false():
    if yaml is None:
        pytest.skip("PyYAML not installed")
    conf = yaml.safe_load((ROOT / "configs" / "seismic.yaml").read_text())
    assert conf["learn_sigma"] is False
    assert conf["in_channels"] == 1
    assert conf["out_channels"] == 1


def test_seismic_config_file_exists():
    assert (ROOT / "configs" / "seismic.yaml").is_file()
