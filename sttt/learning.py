"""Small policy/value network; CPU search and minibatch GPU training."""
import numpy as np
import torch
from torch import nn

INPUTS = 81*3 + 9*4 + 10

def encode(state):
    cells = np.array(state.cells) * state.turn
    boards = np.array(state.boards)
    return np.concatenate([(cells == k).astype(np.float32) for k in (0,1,-1)] +
                          [(boards == k).astype(np.float32) for k in (0,state.turn,-state.turn,2)] +
                          [np.eye(10, dtype=np.float32)[state.forced+1]])

def legal_masks(states):
    """bool[N, 81] legal-action masks, owned and contiguous."""
    mask = np.zeros((len(states), 81), dtype=bool)
    for i, state in enumerate(states):
        mask[i, state.legal_actions()] = True
    return mask


def encode_states(states, backend='auto'):
    """Owned float32[N, 289] features plus bool[N, 81] masks.

    `backend` is explicit on purpose. The previous code probed
    `is_cpp_available()` and used native encoding whenever the extension
    happened to be importable -- so a run that deliberately requested the
    Python reference path was still encoded natively, which is exactly the
    comparison the reference path exists to make.

    The returned features are always an owned, C-contiguous array. The native
    encoder hands back a read-only view over immutable Python bytes, which is
    not a valid owner for `torch.from_numpy`; copying is the correct behaviour
    until the extension exposes an owner-backed buffer.
    """
    if backend not in ('auto', 'cpp', 'python'):
        raise ValueError(f"unknown encode backend {backend!r}")
    if not states:
        return np.empty((0, INPUTS), dtype=np.float32), np.empty((0, 81), dtype=bool)

    use_cpp = False
    if backend in ('auto', 'cpp'):
        try:
            from .cpp_env import encode_batch as _cpp_encode_batch, is_cpp_available
            use_cpp = bool(is_cpp_available())
        except ImportError:
            use_cpp = False
        if backend == 'cpp' and not use_cpp:
            raise RuntimeError(
                'encode backend "cpp" requested but the native encoder is '
                'unavailable; refusing to substitute the Python encoder.')

    if use_cpp:
        features = np.ascontiguousarray(_cpp_encode_batch(states), dtype=np.float32)
    else:
        features = np.ascontiguousarray(
            np.stack([encode(s) for s in states]), dtype=np.float32)
    if not features.flags.owndata or not features.flags.writeable:
        features = features.copy()
    return features, legal_masks(states)


class BasePolicyValue(nn.Module):
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

def create_model(arch='resnet'):
    if arch == 'resnet':
        return ResNet()
    if arch == 'mlp':
        return Network()
    raise ValueError(f"Unknown architecture: {arch}")

def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    arch = checkpoint.get('arch')
    if arch == 'resnet':
        model = ResNet()
    elif arch == 'mlp':
        model = Network()
    else:
        state = checkpoint['model']
        if 'in_proj.0.weight' in state or 'blocks.0.net.0.weight' in state:
            model = ResNet()
        else:
            model = Network()
    model.load_state_dict(checkpoint['model'])
    return model.eval(), checkpoint
