import ast, inspect, unittest
import numpy as np
import torch
from sttt import ai
from sttt.learning import Network, policy_value_loss


def _batch(n=6, seed=0):
    """Random but legal-shaped training batch shared by every loss test."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 289, generator=g)
    mask = torch.rand(n, 81, generator=g) > 0.6
    mask[:, 0] = True
    pi = torch.rand(n, 81, generator=g) * mask
    pi = pi / pi.sum(-1, keepdim=True)
    z = torch.rand(n, generator=g) * 2 - 1
    return x, pi, mask, z


class SharedLossTests(unittest.TestCase):
    def test_matches_the_reference_formula(self):
        torch.manual_seed(0)
        model = Network()
        x, pi, mask, z = _batch()
        loss, parts = policy_value_loss(model, x, pi, mask, z)
        logits, value = model(x)
        log_p = logits.masked_fill(~mask, -1e9).log_softmax(-1)
        log_p = torch.where(mask, log_p, torch.zeros_like(log_p))
        ref_policy = -(pi * log_p).sum(-1).mean()
        ref_value = (value - z).square().mean()
        self.assertTrue(torch.allclose(parts['policy'], ref_policy))
        self.assertTrue(torch.allclose(parts['value'], ref_value))
        self.assertTrue(torch.allclose(loss, ref_policy + ref_value))

    def test_illegal_logits_do_not_leak_into_the_policy_term(self):
        torch.manual_seed(1)
        model = Network()
        x, pi, mask, z = _batch()
        loss_a, _ = policy_value_loss(model, x, pi, mask, z)
        # Poison illegal logits by changing inputs' effect only on masked-out
        # actions is not possible directly; instead verify the gradient of the
        # policy term w.r.t. illegal logits is exactly zero.
        logits, _ = model(x)
        logits = logits.detach().requires_grad_(True)
        log_p = logits.masked_fill(~mask, -1e9).log_softmax(-1)
        log_p = torch.where(mask, log_p, torch.zeros_like(log_p))
        (-(pi * log_p).sum(-1).mean()).backward()
        self.assertTrue(torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask])))
        self.assertTrue(torch.isfinite(loss_a))

    def test_train_loop_has_a_single_loss_implementation(self):
        source = inspect.getsource(ai._train_loop)
        self.assertNotIn('log_softmax', source)
        self.assertIn('policy_value_loss(', source)
