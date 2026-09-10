# Smart Adaptive Agent for Ultimate Tic-Tac-Toe

## Executive recommendation

Build a hybrid agent, not a single end-to-end neural network and not a large if/else rule table:

1. A **correct, deterministic game environment** provides legal moves, transitions, terminal results, cloning, and serialization.
2. A learned **policy/value network** estimates which moves are promising and how good a position is.
3. **Monte Carlo Tree Search (MCTS)** looks ahead from the current position and improves the network's move distribution.
4. A separate **opponent model** learns the human's tendencies from the current game and previous games.
5. The agent keeps a persistent **opponent belief/context** online, but updates the neural-network weights offline or in small scheduled batches to avoid forgetting.

This is closest in spirit to AlphaZero plus opponent modeling. Ultimate Tic-Tac-Toe is a two-player, zero-sum, perfect-information game, so the environment can expose the full board to the agent; the uncertainty is not hidden information, but what the human is likely to do next. The game is still strategically nontrivial: a move selects the opponent's next local board, and published analysis gives the first player a winning strategy under one rule set with an optimal win in at most 43 moves, while the second player can resist for at least 29 moves.^1

The important design distinction is:

- **Rules are engineered.** The agent must never learn whether a move is legal.
- **Strategy is learned.** The agent learns board value, move preferences, and how this particular human responds.
- **Adaptation is state.** The agent should update a compact representation of the human during play rather than rewrite its entire brain after every move.

## What the papers suggest

### AlphaZero: learned policy/value plus search

AlphaZero starts from random play, receives only the rules, and improves through self-play. Its network predicts a move prior and a scalar position value; MCTS uses those predictions to choose stronger actions, and the resulting search distribution becomes a training target.^2 This is a strong fit for Ultimate Tic-Tac-Toe because the game has a compact, exact simulator and a branching structure that makes shallow handcrafted evaluation unreliable.

Do not copy AlphaZero's scale. A small residual CNN or even a carefully designed MLP is enough to start. The conceptual loop is what matters:

```text
position -> network(policy, value)
         -> MCTS guided by policy/value
         -> improved move distribution
         -> self-play data
         -> network update
```

### MuZero: a later learned-dynamics experiment

MuZero replaces the known simulator inside search with a learned latent dynamics model that predicts reward, policy, and value rather than reconstructing the full board.^3 That is scientifically interesting, but it is the wrong first milestone here. This is a small deterministic game whose rules can be implemented exactly. Use the real simulator first; consider MuZero only as a later experiment comparing model-based and rule-based planning.

### Opponent modeling: predict behavior, then use it

DRON showed a direct route for opponent modeling: encode observations of the opponent and feed the learned representation into the agent's action-value computation; a mixture-of-experts variant discovered different opponent patterns without requiring explicit labels.^4 DPIQN/DRPIQN similarly learns policy features and uses a recurrent model when the agent has only a short history.^5

For this game, the observable event is simple: the human's board/cell action, the position in which it occurred, and the preceding history. The model does not need to guess a personality label such as “aggressive.” It can learn a latent vector that answers the useful question: **given this board and this history, which legal move is this human likely to make?**

### Fast adaptation and changing opponents

RL² demonstrates the general idea of putting a fast learner inside the hidden state of a recurrent network: slow training learns the update procedure, while the recurrent state adapts from a few observations at test time.^6 LeMOL applies a related idea specifically to opponents that change over time, using episode summaries and an RNN to track the opponent's policy evolution rather than assuming a stationary opponent.^7

For a single human player, start with a simpler and more interpretable version: a Bayesian belief over a library of opponent policies. If many players are later collected, or if the agent must model learning humans, replace or augment that belief with a GRU/LSTM opponent encoder.

### Population training: avoid learning one brittle counter

Pure self-play can produce a strong-looking agent that is exploitable by a style it never saw. PSRO treats policies as a population and repeatedly adds approximate best responses, creating a meta-game over policies rather than training against only the latest self.^8 The full PSRO machinery is unnecessary initially. Keep a pool of old network checkpoints and diverse opponent bots, sample from them during training, and periodically add a best-response policy when an obvious weakness is found.

The 2024 HOP paper is especially close to the desired architecture: infer latent properties of other agents, update beliefs both across and within episodes, and use the inferred information to guide MCTS.^9 Its environments are more complex and mixed-motive than this game, so use it as an architectural reference, not as a reason to reproduce the full algorithm.

## Recommended system architecture

### 1. Environment API

Refactor the current console game into a pure environment before training anything. The existing project has the board classes and rendering, but it needs a machine-facing interface such as:

```python
class UltimateTicTacToe:
    def initial_state(self) -> State: ...
    def legal_actions(self, state: State) -> list[int]: ...
    def apply(self, state: State, action: int) -> State: ...
    def is_terminal(self, state: State) -> bool: ...
    def outcome(self, state: State) -> int: ...  # -1, 0, +1
    def clone(self, state: State) -> State: ...
    def encode(self, state: State, player: int) -> np.ndarray: ...
    def render(self, state: State) -> None: ...
```

Represent an action as one integer from 0 to 80: `action = 9 * local_board + cell`. Store explicitly:

- the 81 cell contents;
- the 9 local-board statuses: undecided, X, O, or drawn/full;
- the local board to which the next player is sent, or “any board”;
- the player to move;
- terminal status and outcome.

The legal-action mask must be generated by the environment and passed to both the network and MCTS. The network may score all 81 outputs, but illegal actions must receive probability zero before sampling or search.

Before adding learning, test every rule with property-based tests: applying a legal action changes exactly one cell; the next-board constraint is correct; sending a player to a completed board allows any valid board; local and global wins are detected; no action is legal after terminal state; X/O symmetry is preserved.

Your current console implementation also hard-codes `boardno = 4`, so it always starts in the center board even though the README describes a free first move. Resolve this rule explicitly. Add a draw/terminal condition, `clone`, and `legal_actions`; these are prerequisites for search and training.

### 2. State encoder

Start with a tensor of planes rather than a hand-written feature score. A practical encoding is:

- 1 plane for the current player's marks;
- 1 plane for the opponent's marks;
- 1 plane for cells in playable local boards;
- 2 planes indicating local boards owned by either player;
- 1 plane indicating drawn/full local boards;
- 1 plane marking the required next local board;
- 1 constant plane for side-to-move or use canonical “current player first” encoding.

Use data augmentation by applying the eight rotations/reflections of the 3×3 global board and the corresponding transformations inside each local board. This reduces the amount of self-play needed and is valid only if the environment transforms the next-board pointer consistently.

### 3. Base policy/value network

Use a small shared trunk with two heads:

```text
board tensor -> residual/MLP trunk -> h
                                  ├─ policy head: 81 logits
                                  └─ value head: scalar in [-1, 1]
```

Then add the opponent context:

```text
human action history -> GRU or Bayesian belief -> z
board representation h + z -> policy/value heads
```

For a first implementation, concatenate a 16–64 dimensional context vector `z` into the trunk after board encoding. Keep the base network strong enough to play without `z`; this prevents the agent from becoming helpless against a novel or random human.

### 4. Opponent model

Implement this in two stages.

#### Stage A: Bayesian policy library — recommended first

Create a small set of opponent policies:

- uniform random legal move;
- local tactical player that prioritizes immediate local wins/blocks;
- global tactical player that prioritizes winning or blocking a local-board line;
- center/corner biased player;
- shallow minimax or shallow MCTS players;
- noisy versions of the trained policy at several temperatures/search budgets;
- policies sampled from old training checkpoints.

For each policy (k), estimate how likely it is to select the human's observed action:

\[
\log b_k \leftarrow \log b_k + \log(\epsilon + \pi_k(a_t \mid s_t))
\]

Normalize the beliefs after every human move. Discount old evidence if the human changes style:

\[
\log b_k \leftarrow \rho\log b_k + \log(\epsilon + \pi_k(a_t \mid s_t)),
\quad 0 < \rho \le 1.
\]

The opponent prediction used by search is the mixture

\[
\pi_{opp}(a \mid s,h)=\sum_k b_k\pi_k(a\mid s).
\]

This already gives visible adaptation: if the human repeatedly sacrifices a local board to send the agent somewhere awkward, the belief shifts toward policies exhibiting that behavior, and MCTS evaluates the likely response rather than an idealized best response.

#### Stage B: learned recurrent opponent encoder

Once the base system works, train a GRU/LSTM to predict the opponent action from `(state, previous action, result, legal mask)` history. Train it on trajectories generated by the policy library, self-play checkpoints, and human games. The encoder output `z_t` is fed into the agent policy/value network and optionally into the opponent policy used at opponent nodes in MCTS.

Use a mixture of supervised behavior loss and game-strength loss:

\[
L = L_{AZ} + \beta L_{opp},
\]

where

\[
L_{AZ}=(z-v_\theta(s))^2 - \pi_{MCTS}^{T}\log p_\theta(\cdot\mid s,z)
\]

and

\[
L_{opp}=-\log q_\phi(a^{human}_t\mid s_t,h_t).
\]

Do not interpret high action-prediction accuracy as high playing strength. The opponent model is useful only if conditioning the policy/search on it improves outcomes against that opponent.

### 5. MCTS with an opponent model

At the agent's nodes, use PUCT with the network policy prior. At human nodes, there are two sensible choices:

1. **Expected-response search:** sample or branch according to `π_opp`, which is appropriate when the goal is to exploit the observed human.
2. **Robust-response search:** mix the opponent model with a strong-policy prior, for example `p = α π_opp + (1-α) π_strong`, where `α` increases with model confidence.

Use robust search early in a match and move toward exploitative search only when the model has accumulated evidence. This prevents the classic failure mode where one surprising human move causes the agent to overfit and blunder.

At the root, choose from visit counts, not raw neural logits. During self-play, sample from visit counts with temperature; during evaluation and human play, use a low temperature or argmax with a small amount of controlled noise.

## Training curriculum

### Milestone 0: deterministic baseline

Implement and test the environment. Add random, tactical, shallow minimax, and shallow MCTS bots. This is not the final intelligence; it is the test harness and source of diverse trajectories.

### Milestone 1: search without learning

Implement MCTS using random rollouts and a simple terminal evaluator. Confirm it beats random and tactical bots. This isolates search bugs before neural training.

### Milestone 2: AlphaZero-style self-play

Train the policy/value network from self-play. Maintain a replay buffer of recent and historical games. Use a population of frozen checkpoints instead of always playing only the latest network. Evaluate against held-out checkpoints, deterministic bots, and random-seed variations.

### Milestone 3: opponent modeling

Train or hand-specify the policy library, update Bayesian beliefs after each human action, and route the mixture into opponent-node MCTS. Add a UI display in debug mode showing predicted top moves and confidence, but keep it hidden during normal play.

### Milestone 4: human adaptation

Persist a player profile containing recent trajectories, belief state, and aggregate statistics. At the start of a match, initialize from the profile; during the match, update quickly; after the match, append to replay. Do not run a large gradient update synchronously after every move.

### Milestone 5: learned fast adaptation

Only after the previous milestones work, train a recurrent opponent encoder or an RL²-style meta-learner. Train across many simulated “players” whose styles and policies change between episodes. Hold out entire opponent generators during evaluation so the model cannot memorize labels.

## Online learning policy

A good deployment rule is:

```text
Every human move:
    update opponent belief/context
    run MCTS conditioned on current context
    play a legal action

After each completed game:
    store trajectory and prediction errors
    update player profile

Every N games or when idle:
    fine-tune from a replay mixture of old self-play + recent human games
    keep the previous checkpoint unless evaluation improves
```

The replay mixture matters. Continual-learning work shows that replaying past experience helps new learning without catastrophic forgetting; the same principle applies here.^10 A practical starting ratio is 80% old self-play/population data and 20% recent human data, adjusted only after measuring whether adaptation is too slow or the base strength is degrading.

## Evaluation: prove that it learns you

Report separate metrics instead of one win rate:

| Question | Metric | Required comparison |
|---|---|---|
| Is it legal? | illegal-action rate | must be 0% |
| Is it strong? | win/draw/loss and Elo-like rating | random, tactical, MCTS, held-out checkpoints |
| Does it predict the human? | log loss, top-1/top-3 accuracy, calibration | opponent model vs uniform/strong-policy prior |
| Does prediction help? | score against the same human | with vs without opponent context |
| Does it learn over games? | first 5 vs last 5 game performance | same human, same starting-side balance |
| Can it handle change? | recovery after a style switch | fixed belief, discounted belief, recurrent model |
| Is it robust? | worst-case score over opponent population | latest checkpoint vs population-trained agent |

Run ablations: no MCTS, MCTS without opponent model, opponent model without MCTS conditioning, no replay, no checkpoint population, and no symmetry augmentation. The claim “smart” is supported only if the full system beats these ablations and improves against a held-out human or held-out style generator.

## What not to build first

- Do not train a giant transformer from a handful of games; there is not enough data.
- Do not update all network weights after every move; one human game is far too little data and will cause forgetting.
- Do not remove the exact rules in the name of learning; illegal-move learning creates a broken agent.
- Do not train only against the latest self-play copy; it encourages co-adaptation and brittle play.
- Do not use raw win/loss alone to train an opponent predictor; the same loss can result from a good move, a blunder, or the agent's own mistake.
- Do not use a language model to choose board moves. A language model can explain moves or generate commentary, but the board agent should be a learned policy/value/search system with an exact game state.

## Compute budget

The dominant cost is self-play with MCTS. The opponent-belief update is tiny, and training a GRU opponent predictor is usually small compared with generating the trajectories it consumes. These are planning estimates rather than guarantees; implementation language, batching, number of simulations, and stopping criteria can change runtime by an order of magnitude.

| Target | Self-play/search budget | Hardware | Approximate wall time |
|---|---:|---|---:|
| Environment, rule tests, random/tactical bots | 10,000–1,000,000 games without neural search | 4–8 CPU cores, 8 GB RAM | minutes to a few hours |
| First learned agent | 5,000–20,000 games; 32–128 MCTS simulations/move; 0.1–3M parameters | CPU or one 8 GB GPU | hours to 2 days |
| Good hobby-strength agent | 50,000–250,000 games; 128–400 simulations/move; 1–10M parameters | one 8–16 GB GPU, 8–16 CPU cores, 16–32 GB RAM | 1–7 days |
| Stronger research experiment with population training | 250,000–2,000,000 games; 400–800 simulations/move; multiple checkpoints and ablations | one 12–24 GB GPU plus many CPU self-play workers, or several GPUs | 1–4 weeks |

The ranges assume batched neural inference. Unbatched MCTS can underuse a GPU; a small Tic-Tac-Toe AlphaZero implementation explicitly notes that sequential single-position inference can make a GPU slower than a CPU, while GPUs become useful when MCTS evaluations are batched or the game/network is larger.^11

As an empirical Ultimate Tic-Tac-Toe reference, the open-source `uttt.ai` project reports a 5-million-parameter policy/value network, two 8-million-position datasets, development on two RTX 2080 Ti GPUs, and a training goal that fit a personal computer over several weeks.^12 That is a useful upper-bound reference for a serious hobby implementation, not a minimum requirement. Its reported inference measurements also show why search count matters: 1,000 neural MCTS simulations took about 4.4 seconds and 10,000 took about 17.8 seconds in its single-threaded C++ setup on an i7-10700K/RTX 2080 Ti system.^12

For the complete adaptive system, budget approximately:

- **Development:** CPU-only is sufficient.
- **Initial AlphaZero-like training:** one modest GPU is comfortable; CPU-only is possible but likely frustrating.
- **Opponent model:** add less than 10% to the compute budget if trained from existing trajectories.
- **Online human play:** no training cluster is required. Store the human profile and update Bayesian beliefs on every move; run a few hundred to a few thousand MCTS simulations locally.
- **Population training and ablations:** expect the compute to multiply with the number of opponent populations, seeds, and evaluation matches—not because the network is large, but because every experiment generates many games.

The sensible starting purchase is therefore no special hardware: implement and test on the current machine, then use one 8–12 GB GPU if self-play generation becomes the bottleneck. Do not rent TPU-scale infrastructure. Spend additional compute on batched self-play workers, better evaluation, and opponent diversity before increasing network size.

## Suggested project layout

```text
sttt/
  env.py              # immutable State, legal actions, transitions, terminal logic
  encode.py           # tensor encoding and symmetry transforms
  bots.py             # random, tactical, minimax, MCTS opponents
  model.py            # policy/value network and opponent encoder
  search.py           # PUCT/MCTS, opponent-node distributions
  opponent.py         # policy library, Bayesian belief, GRU encoder later
  selfplay.py         # parallel trajectory generation
  train.py            # replay, losses, checkpointing
  evaluate.py         # round-robin, ablations, adaptation metrics
  cli.py              # human interface
```

The fastest credible route is: environment → MCTS → AlphaZero-style network → Bayesian opponent model → recurrent/meta opponent model. Each stage is useful on its own, testable, and provides evidence that the next layer is improving the actual player experience.

## References

1. Guillaume Bertholon, Rémi Géraud-Stewart, Axel Kugelmann, Théo Lenoir, and David Naccache, “At Most 43 Moves, At Least 29: Optimal Strategies and Bounds for Ultimate Tic-Tac-Toe,” arXiv:2006.02353, 2020. https://arxiv.org/abs/2006.02353
2. David Silver et al., “Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm,” arXiv:1712.01815, 2017. https://arxiv.org/abs/1712.01815
3. Julian Schrittwieser et al., “Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model,” arXiv:1911.08265, 2019. https://arxiv.org/abs/1911.08265
4. He He, Jordan Boyd-Graber, Kevin Kwok, and Hal Daumé III, “Opponent Modeling in Deep Reinforcement Learning,” ICML/PMLR 48, 2016. https://proceedings.mlr.press/v48/he16.html
5. Zhang-Wei Hong, Shih-Yang Su, Tzu-Yun Shann, Yi-Hsiang Chang, and Chun-Yi Lee, “A Deep Policy Inference Q-Network for Multi-Agent Systems,” AAMAS, 2018. https://arxiv.org/abs/1712.07893
6. Yan Duan et al., “RL²: Fast Reinforcement Learning via Slow Reinforcement Learning,” arXiv:1611.02779, 2016. https://arxiv.org/abs/1611.02779
7. Ian Davies, Zheng Tian, and Jun Wang, “Learning to Model Opponent Learning,” arXiv:2006.03923, 2020. https://arxiv.org/abs/2006.03923
8. Marc Lanctot et al., “A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning,” NeurIPS, 2017. https://proceedings.neurips.cc/paper_files/paper/2017/file/3323fe11e9595c09af38fe67567a9394-Paper.pdf
9. Yizhe Huang et al., “Efficient Adaptation in Mixed-Motive Environments via Hierarchical Opponent Modeling and Planning,” ICML/PMLR 235, 2024. https://proceedings.mlr.press/v235/huang24p.html
10. David Rolnick et al., “Experience Replay for Continual Learning,” NeurIPS, 2018. https://arxiv.org/abs/1811.11682
11. Weill Labs, “AlphaZero (tic-tac-toe),” open-source implementation and engineering notes. https://github.com/weill-labs/alphazero
12. Arnowaczynski, “uttt.ai: AlphaZero-like AI solution for Ultimate Tic-Tac-Toe,” open-source implementation and training report. https://github.com/arnowaczynski/utttai

Additional implementation reference: Google DeepMind's OpenSpiel provides research environments and algorithms for reinforcement learning and search/planning in games, including an AlphaZero example for Tic-Tac-Toe. https://github.com/google-deepmind/open_spiel and https://openspiel.readthedocs.io/en/stable/alpha_zero.html
