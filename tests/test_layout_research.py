import numpy as np
import torch

from accelerator_bench.layout_research import conv_from_definition, hello_layout


def test_definition_handles_spatial_kernel_multiple_channels_and_batches():
    rng = np.random.default_rng(3)
    x, w = rng.normal(size=(2, 3, 5, 7)), rng.normal(size=(4, 3, 2, 3))
    actual = conv_from_definition(x, w)
    expected = torch.nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_hello_layout_distinguishes_reordering_from_reshape():
    result = hello_layout()
    assert result['output'] == [[10., 13., 16.], [19., 22., 25.]]
    assert result['max_error'] == 0
    assert result['wrong_reshape_output'] != result['output']
    assert result['nhwc_storage'][:4] == [0., 10., 1., 11.]
