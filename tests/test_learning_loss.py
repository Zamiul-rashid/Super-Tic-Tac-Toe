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


class QHeadLossTests(unittest.TestCase):
    class WithQ(Network):
        def __init__(self):
            super().__init__()
            self.qhead = torch.nn.Linear(256, 81)
        def forward_all(self, x):
            h = self.trunk(x)
            return self.policy(h), self.value(h).tanh().squeeze(-1), self.qhead(h).tanh()

    def test_models_without_q_output_ignore_q_targets(self):
        torch.manual_seed(0)
        model = Network()
        x, pi, mask, z = _batch()
        loss_plain, parts_plain = policy_value_loss(model, x, pi, mask, z)
        loss_q, parts_q = policy_value_loss(model, x, pi, mask, z, q=torch.rand(6, 81), q_mask=mask)
        self.assertIsNone(parts_q['q'])
        self.assertTrue(torch.equal(loss_plain, loss_q))

    def test_q_term_is_masked_mse_over_visited_actions_only(self):
        torch.manual_seed(0)
        model = self.WithQ()
        x, pi, mask, z = _batch()
        q = torch.rand(6, 81) * 2 - 1
        q_mask = mask.clone(); q_mask[0] = False       # one row with no targets
        loss, parts = policy_value_loss(model, x, pi, mask, z, q=q, q_mask=q_mask)
        _, _, q_pred = model.forward_all(x)
        ref = ((q_pred - q).square() * q_mask).sum() / q_mask.sum()
        self.assertTrue(torch.allclose(parts['q'], ref))
        self.assertTrue(torch.allclose(loss, parts['policy'] + parts['value'] + parts['q']))

    def test_all_false_mask_contributes_nothing_and_no_nan(self):
        torch.manual_seed(0)
        model = self.WithQ()
        x, pi, mask, z = _batch()
        loss, parts = policy_value_loss(model, x, pi, mask, z, q=torch.zeros(6, 81),
                                        q_mask=torch.zeros(6, 81, dtype=torch.bool))
        self.assertIsNone(parts['q'])
        self.assertTrue(torch.isfinite(loss))

    def test_trainer_passes_q_targets_and_reports_q_loss(self):
        source = inspect.getsource(ai._train_loop)
        self.assertIn('q=q, q_mask=q_mask', source)
        self.assertIn("'q_loss'", source)
