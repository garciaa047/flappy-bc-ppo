# flappy-bc-ppo

A Flappy Bird environment built from scratch in pygame and wrapped as a [Gymnasium](https://gymnasium.farama.org/) environment, used to explore whether **Behavioural Cloning (BC) from human demonstrations can meaningfully accelerate PPO training** — and what happens when it does.

By [@garciaa047](https://github.com/garciaa047)

---

## Why This Project

Reinforcement learning agents learning Flappy Bird from scratch face a fundamental problem: the reward signal is extremely sparse early in training. The agent dies almost immediately, receives a −1 reward, and has very little signal to learn from. This makes the initial learning phase slow and sample-inefficient.

This project explores a practical solution: **first teach the agent by imitating a human player**, then let PPO refine that behaviour through self-play. The hypothesis is that a BC-initialised agent should learn faster and more stably than one starting from random weights.

The results confirmed this — but also revealed an interesting failure mode worth understanding.

---

## The Pipeline

```
1. Play the game          →  collect_human_data.py
2. Train on human demos   →  train_bc.py
3. Fine-tune with PPO     →  finetune_ppo.py
4. Evaluate               →  evaluate.py
```

Each stage is a standalone script. You can run only PPO from scratch (`train_ppo.py`) and compare it directly against the BC+PPO pipeline.

---

## Results

### Sample Efficiency: BC+PPO vs PPO Only

The headline result: at **150k timesteps**, BC+PPO achieves an average reward of **13**, while PPO alone sits at **0.09**. PPO alone does not reach a comparable level of performance until approximately **500k timesteps** — over 3× more samples.

![PPO and PPO + BC average reward plot](./plots/plot_1_reward_comparison.png)

### The Collapse

At around **190k–220k timesteps**, the BC+PPO agent's performance drops sharply — from ~13 average reward back down to ~4. This is not a bug. It is a well-known failure mode caused by the **value function cold start problem**:

- The BC stage only trains the *actor* (policy network). The *critic* (value function) is randomly initialised.
- In the early PPO phase, the critic is noisy and produces small, inconsistent gradient updates — the BC policy is largely preserved.
- Once the critic has trained long enough to produce confident estimates, it begins driving large policy updates. But those estimates are biased, having been fit to rollouts from an already-good policy. The result is a confident-but-wrong critic that pushes the actor into worse regions of parameter space.
- The degraded actor produces worse rollouts, which makes the critic's estimates worse — a compounding feedback loop.

<!-- Insert collapse graph here showing reward drop around 190k steps -->

Understanding *why* this happens is arguably more valuable than the result itself. Potential fixes include pre-warming the critic before PPO begins, reducing the learning rate further, or adding a KL penalty to slow policy drift.

---

## Project Structure

```
flappy-bc-ppo/
├── flappy/
│   ├── game.py               # Pygame Flappy Bird implementation
│   └── env.py                # Gymnasium wrapper (observation, action, reward)
├── train_ppo.py              # Train a PPO agent from scratch
├── train_bc.py               # Train a BC policy on human demonstrations
├── finetune_ppo.py           # Fine-tune a BC policy with PPO
├── collect_human_data.py     # Play the game and log demonstrations
├── evaluate.py               # Evaluate any saved model visually
├── models/                   # Saved model checkpoints
├── human_data/               # Collected human demonstration episodes (.npz)
└── tensorboard_logs/         # Training logs
    ├── PPO_log/
    └── finetuned_PPO_log/
```

### Environment Design

| Property | Value |
|---|---|
| Observation space | `Box(5,)` — bird y, velocity, next pipe x, pipe top y, pipe bottom y |
| Action space | `Discrete(2)` — 0: do nothing, 1: flap |
| Reward | +1 per pipe passed, −1 on death |
| Timestep | Fixed 16ms (deterministic, reproducible) |

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Play the game

```bash
python -m flappy.game
```

### Collect human demonstrations

```bash
python collect_human_data.py
python collect_human_data.py --max_episodes 30
```

Only episodes scoring ≥ 15 pipes are saved. Each episode is stored as a `.npz` file in `human_data/`.

### Train BC on your demonstrations

```bash
python train_bc.py
python train_bc.py --n_epochs 100 --batch_size 64
```

Saves `models/flappyPPO_bc`.

### Fine-tune with PPO

```bash
python finetune_ppo.py
python finetune_ppo.py --total_timesteps 500_000
```

Saves `models/flappyPPO_finetuned_latest` and `models/flappyPPO_finetuned_best_model`.

### Train PPO from scratch (baseline)

```bash
python train_ppo.py
```

### Evaluate a model

```bash
python evaluate.py --model models/flappyPPO_finetuned_best_model --fps 60
python evaluate.py --model models/flappyPPO_best_model --fps 60
python evaluate.py --headless --episodes 50   # fast headless evaluation
```

### View training curves

```bash
tensorboard --logdir=./tensorboard_logs/
```

---

## What I Learned

- **Imitation learning as a warm start is effective but not free.** BC significantly accelerates early PPO training, but introduces a new failure mode (value function cold start) that requires careful handling.

- **The critic matters as much as the actor.** Most intuition about "the policy" focuses on the actor. This project made clear that a randomly initialised critic can actively destroy a well-trained actor once it becomes confident enough to produce large gradient updates.

- **Building the full pipeline end-to-end is where the real learning happens.** Integrating pygame → Gymnasium → SB3 → imitation required understanding how each layer expects data, what assumptions each library makes, and where those assumptions conflict.

- **Diagnosing failure is more instructive than achieving success.** The performance collapse around 200k steps was unexpected, but understanding *why* it happens — and what it would take to fix — is a deeper insight than simply watching the reward go up.

---

## Future Work

- [ ] Pre-warm the critic on BC rollouts before starting PPO updates
- [ ] Add a KL divergence penalty to slow policy drift after BC initialisation
- [ ] Experiment with DAgger as an alternative to offline BC
- [ ] Add pixel-based observations and a CNN policy (`CnnPolicy`)
- [ ] Replace placeholder rectangle sprites with proper Flappy Bird assets
