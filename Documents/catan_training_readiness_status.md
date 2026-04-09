# Catan Training Readiness Status

## Purpose

This document summarizes:
- what has already been built for the Catan RL stack,
- what is stable enough to treat as a current foundation,
- and what is still left before we can start real training.

## What Has Been Done

### 1. Base game engine
Implemented in `catan.py`.

Current engine capabilities:
- randomized base-game board generation
- full graph-based topology
- setup snake draft
- roads, settlements, cities
- robber movement and stealing
- resource production
- development cards
- Largest Army
- Longest Road
- maritime trading
- simplified domestic trading in the core engine
- CLI gameplay
- SVG rendering
- event logs

Important current project choice:
- domestic trading is disabled in the RL wrapper for v1 training

### 2. Rules validation and regression tests
Implemented in `test_catan.py`.

Covered areas include:
- setup legality
- production
- robber/discard flow
- bank shortage behavior
- city production
- harbor behavior
- Longest Road
- Largest Army
- win timing
- dev-card timing
- domestic trade flow in the simulator

This gives us a fairly stable simulator baseline.

### 3. RL environment wrapper
Implemented in `catan_rl_env.py`.

Current wrapper capabilities:
- `reset()`
- `observe()`
- `step(action_index)`
- rank-focused default reward config
- per-step reward components in `info`
- optional max-turn truncation
- legal action filtering for training
- domestic trades disabled by config for v1 training

Important current action design:
- action indexing mode is `phase_local`
- action index is chosen from the currently legal action list
- this is good enough for initial custom-loop training
- this is not yet a single fixed global action catalog

### 4. Frozen observation tensor schema
Implemented in `catan_rl_env.py` and documented in `Documents/catan_observation_tensor_schema.md`.

Current tensor blocks:
- `global_features`
- `hex_features`
- `intersection_features`
- `edge_features`
- `player_features`
- `private_features`

Current frozen shapes:
- `global_features`: `[21]`
- `hex_features`: `[19, 20]`
- `intersection_features`: `[54, 16]`
- `edge_features`: `[72, 9]`
- `player_features`: `[4, 11]`
- `private_features`: `[15]`

The observation boundary is now explicit:
- public information is exposed publicly
- the acting player gets exact private hand/dev-card information
- opponents do not expose exact hidden hands

### 5. Legal action tensorization
Implemented in `catan_rl_env.py`.

The env now exports:
- `action_tensor_spec`
- `action_features`
- `legal_action_mask`

This means the model can score the currently legal actions using learned action features instead of needing a giant fixed action table immediately.

### 6. Shared encoder
Implemented in `catan_model.py`.

Current model pieces:
- `SharedCatanEncoder`
- per-entity pooled encoders
- shared latent trunk

The shared encoder consumes the frozen tensor schema and produces a latent state embedding.

### 7. Phase-specific policy heads and value head
Implemented in `catan_model.py`.

Current policy/value pieces:
- `PhaseActionHead`
- `CatanPolicyValueNet`
- phase-specific action scoring over legal action features
- scalar value head
- deterministic and stochastic action sampling

This means the model can now:
- encode a game state
- score legal actions for the current phase
- sample a valid action
- estimate a value for the state

### 8. Rollout collection scaffold
Implemented in `catan_rollout.py`.

Current rollout support:
- one full episode rollout
- deterministic or sampled action selection
- step records containing:
  - action index
  - reward
  - value
  - log-prob
  - entropy
  - legal action count
  - reward components

This is enough to start building PPO-style trajectory collection.

### 9. RL-specific tests
Implemented in:
- `test_catan_rl_env.py`
- `test_catan_model.py`

Coverage includes:
- tensor schema presence and shapes
- action feature shapes
- encoder forward pass
- policy/value forward pass
- legal action sampling
- deterministic rollout smoke test

## Current Status

We are no longer at the “design only” stage.

We now have:
- a working Catan simulator,
- a frozen observation schema,
- a legal-action representation,
- a PyTorch shared encoder,
- phase-specific policy heads,
- a value head,
- and a minimal rollout collector.

That means the project is now structurally ready for a first training loop.

## What Is Left Before We Can Start Training

### 1. PPO training loop
This is the main missing piece.

Still needed:
- trajectory batching
- return computation
- advantage computation
- PPO clipped objective
- value loss
- entropy bonus
- optimizer step
- minibatch training loop

This is the biggest remaining blocker.

### 2. Advantage / return computation
We need a clear implementation of:
- discounted returns
- ideally GAE-lambda

Without this, we can collect rollouts but not train a stable actor-critic policy.

### 3. Multi-episode batch collection
Right now we can collect an episode.

Still needed:
- collect many episodes / steps
- flatten them into training batches
- preserve per-step:
  - logits or log-probs
  - values
  - rewards
  - dones
  - action masks if needed

### 4. Checkpointing
Still needed:
- save model weights
- save optimizer state
- save config snapshot
- save training metrics

This is important so we can resume training and compare runs.

### 5. Training config layer
We should add a first config path for:
- learning rate
- gamma
- gae lambda
- clip epsilon
- entropy coefficient
- value coefficient
- rollout horizon
- batch size
- minibatch size
- epochs per update
- seed

This does not have to be fancy yet, but it should exist.

### 6. Metrics logging
Before training starts, we should log at least:
- episode reward
- average placement
- winner distribution
- average final VP
- policy loss
- value loss
- entropy
- explained variance if possible
- average game length

This can be simple console logging at first.

### 7. Baseline training script
We need an entry point such as:
- `train_catan_ppo.py`

That script should:
- build envs
- build the model
- collect rollouts
- train
- evaluate periodically
- save checkpoints

### 8. Evaluation loop
Strictly speaking, we can start training before this is polished, but it is still important.

At minimum we want:
- deterministic evaluation episodes
- average placement
- win rate / first-place rate
- average turns

### 9. Optional but likely useful: vectorized environment collection
Not strictly required for the first training run, but likely useful soon.

For v1 we can start with:
- sequential environment rollouts

Then later add:
- multi-env rollout workers

## Things That Are Not Blocking First Training

These can wait until after the first PPO baseline runs.

### 1. Fixed global action catalog
Right now we use `phase_local` action indexing.
That is okay for the first custom training loop.

### 2. Domestic trade training
We intentionally disabled it for v1 training.
That is fine.

### 3. GNN encoder
Not needed yet.
The pooled entity encoder is a good first baseline.

### 4. Self-play league / opponent pools
Not needed for the first proof-of-life training run.
We can start with a shared policy across all seats.

### 5. Reward experimentation beyond the current rank-focused default
We already have the reward-component pathway.
We can tune reward configs after the first training loop works.

## Practical Definition Of “Ready To Start Training”

We can start real training once these are done:
1. PPO update code
2. trajectory batching
3. GAE / return computation
4. training script
5. checkpointing and basic logging

That is the remaining core work.

## Recommended Next Step

Build the first minimal PPO trainer.

Suggested order:
1. add trajectory and GAE utilities
2. add PPO loss/update step
3. add a simple `train_catan_ppo.py`
4. run a very small deterministic smoke training job

## Verification Snapshot

Current automated status at the time of writing:
- `python3 -m unittest -q test_catan.py test_catan_rl_env.py test_catan_model.py`
- result: all tests passing

## Bottom Line

What is already done:
- simulator
- RL wrapper
- frozen tensor schema
- legal action tensorization
- shared encoder
- phase-specific policy heads
- value head
- rollout collector
- tests

What is left before actual training:
- PPO trainer
- return/advantage computation
- batching
- checkpointing
- config/logging
- training entry script
