"""Small checks of stage causality and token-split forward/gradient semantics."""

import unittest
import torch
from cavm.transformer import ModelArgs, Transformer
from cavm.token_swin import WindowAttentionTokenTune


class CoreChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)

    def test_future_stage_cannot_change_earlier_outputs(self):
        model = Transformer(
            ModelArgs(dim=24, n_layers=2, n_heads=4, max_seq_len=12, seq_step=4)
        ).eval()
        x = torch.randn(1, 12, 24)
        changed = x.clone()
        changed[:, 8:] += torch.randn_like(changed[:, 8:]) * 10
        with torch.no_grad():
            a, b = model(x, 0), model(changed, 0)
        torch.testing.assert_close(a[:, :8], b[:, :8], rtol=0, atol=0)
        self.assertFalse(torch.allclose(a[:, 8:], b[:, 8:]))

    def test_split_keeps_forward_values_and_stops_frozen_input_gradients(self):
        model = WindowAttentionTokenTune(
            dim=12, num_heads=3, window_size=(2, 2, 2), qkv_bias=True
        ).eval()
        x = torch.randn(2, 8, 12, requires_grad=True)
        active = torch.zeros(2, 8, dtype=torch.bool)
        active[:, ::2] = True
        y = model(x, None, active)
        with torch.no_grad():
            full = model(x, None, torch.ones_like(active))
            split = model(x, None, active)
        torch.testing.assert_close(y, full, rtol=1e-5, atol=1e-6)
        self.assertFalse(split.requires_grad)
        y.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad[active].abs().sum().item(), 0)
        self.assertEqual(x.grad[~active].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
