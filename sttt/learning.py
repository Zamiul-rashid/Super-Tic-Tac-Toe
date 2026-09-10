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

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(INPUTS,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU())
        self.policy = nn.Linear(256,81)
        self.value = nn.Linear(256,1)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy(h), self.value(h).tanh().squeeze(-1)

    @torch.inference_mode()
    def evaluate(self, state):
        p,v = self(torch.from_numpy(encode(state)).unsqueeze(0).to(next(self.parameters()).device))
        legal = state.legal_actions()
        probs = np.zeros(81, dtype=np.float64)
        probs[legal] = p[0,legal].softmax(0).cpu().numpy()
        probs /= probs.sum()
        return probs, v.item()

def load_model(path):
    model = Network()
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    return model.eval(), checkpoint
