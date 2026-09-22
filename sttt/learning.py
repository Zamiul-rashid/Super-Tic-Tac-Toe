"""Small policy/value network; CPU search and minibatch GPU training."""
import numpy as np
import torch
from torch import nn

from .encoding import INPUTS, encode, legal_masks, encode_states  # noqa: F401


def policy_value_loss(model, x, pi, mask, z, *, use_fp16=False, q=None, q_mask=None):
    """AlphaZero objective plus, when the model has an action-value head and the
    batch carries visited-action targets, uttt.ai's dense Q regression: masked
    MSE over visited legal actions, averaged over the number of targets so rows
    without targets (legacy replay) contribute nothing.
    """
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_fp16):
        logits, value, q_pred = model.forward_all(x)
        mask_val = -1e4 if use_fp16 else -1e9
        logits = logits.masked_fill(~mask, mask_val)
        log_p = logits.log_softmax(-1)
        log_p = torch.where(mask, log_p, torch.zeros_like(log_p))
        policy_loss = -(pi * log_p).sum(-1).mean()
        value_loss = (value - z).square().mean()
        loss = policy_loss + value_loss
        q_loss = None
        if q_pred is not None and q is not None and q_mask is not None and bool(q_mask.any()):
            q_loss = ((q_pred.float() - q).square() * q_mask).sum() / q_mask.sum()
            loss = loss + q_loss
    return loss, {'policy': policy_loss, 'value': value_loss, 'q': q_loss}


class BasePolicyValue(nn.Module):
    def forward_all(self, x):
        """(logits, value, q) — q is None for architectures without an action-value head."""
        logits, value = self(x)
        return logits, value, None

    @torch.inference_mode()
    def evaluate_many(self, states):
        # M3: there used to be a second copy of this body below an unconditional
        # `return`, and that dead copy was the ONLY one honouring use_fp16. So
        # --fp16 silently did nothing for self-play inference -- every
        # evaluation ran FP32 while the run reported mixed precision. One path
        # now, and it actually consults the flag.
        if not states:
            return []
        if any(s.result is not None for s in states):
            raise ValueError('Neural evaluation expects nonterminal positions')

        features, mask = encode_states(states, backend=getattr(self, 'encode_backend', 'auto'))
        device = next(self.parameters()).device
        x = torch.from_numpy(features)
        mask_t = torch.from_numpy(mask).to(device)

        use_fp16 = getattr(self, 'use_fp16', False) and device.type == 'cuda'
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_fp16):
            logits, values = self(x.to(device))
            # -inf overflows to NaN under float16 autocast; -1e4 is the same
            # guard the training step uses and saturates softmax identically.
            fill = -1e4 if use_fp16 else -torch.inf
            logits = logits.masked_fill(~mask_t, fill)
            probabilities = logits.float().softmax(-1).cpu().numpy().astype(np.float64)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return list(zip(probabilities, values.float().cpu().numpy().tolist()))

    @torch.inference_mode()
    def evaluate(self, state):
        return self.evaluate_many([state])[0]

class Network(BasePolicyValue):
    """Legacy 160k-parameter 2-layer MLP."""
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(INPUTS, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.policy = nn.Linear(256, 81)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy(h), self.value(h).tanh().squeeze(-1)

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    def forward(self, x):
        return torch.relu(x + self.net(x))

class ResNet(BasePolicyValue):
    """~1.8M parameter Deep Residual Network with LayerNorm."""
    def __init__(self, inputs=INPUTS, hidden=512, blocks=3):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(blocks)])
        self.policy = nn.Linear(hidden, 81)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.in_proj(x)
        for b in self.blocks:
            h = b(h)
        return self.policy(h), self.value(h).tanh().squeeze(-1)

def arch_name(model):
    from .unet import UNet          # local import: unet imports BasePolicyValue from here
    if isinstance(model, UNet):
        return 'unet'
    return 'resnet' if isinstance(model, ResNet) else 'mlp'

def create_model(arch='resnet'):
    if arch == 'resnet':
        return ResNet()
    if arch == 'mlp':
        return Network()
    if arch == 'unet':
        from .unet import UNet      # local import: unet imports BasePolicyValue from here
        return UNet()
    raise ValueError(f"Unknown architecture: {arch}")

def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    arch = checkpoint.get('arch')
    state = checkpoint['model']
    if arch is None:
        if 'stem.0.weight' in state:
            arch = 'unet'
        elif 'in_proj.0.weight' in state or 'blocks.0.net.0.weight' in state:
            arch = 'resnet'
        else:
            arch = 'mlp'
    model = create_model(arch)
    model.load_state_dict(state)
    return model.eval(), checkpoint
