import pytest
from accelerator_bench.telemetry import integrate_power


def test_constant_power_and_clipped_interval():
    samples = [{"t_s": t, "power_w": 20} for t in [0, 1, 2, 3]]
    assert integrate_power(samples, .5, 2.5) == pytest.approx(40)


def test_linear_power_interpolation():
    samples = [{"t_s": 0, "power_w": 0}, {"t_s": 2, "power_w": 40}]
    assert integrate_power(samples, .5, 1.5) == pytest.approx(20)


def test_no_extrapolation_or_reversed_time():
    samples = [{"t_s": 1, "power_w": 20}, {"t_s": 2, "power_w": 20}]
    with pytest.raises(ValueError):
        integrate_power(samples, 0, 2)
    with pytest.raises(ValueError):
        integrate_power(samples, 2, 1)
