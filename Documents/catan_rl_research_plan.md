# Catan RL Research Plan

## Goal

This document lays out a practical research and implementation plan for training reinforcement learning agents to play the current base-game Catan environment in `catan.py`.

The objective is not just to train a policy, but to build a training stack that is:
- compatible with the current simulator,
- easy to extend as rules/UI improve,
- stable under self-play,
- able to support imperfect-information training,
- and flexible enough to experiment with different reward functions and evaluation methodologies.

## Recommendation Summary

The best first approach is:
- use a shared encoder for the acting player's observation,
- use phase-specific policy heads,
- use one value head,
- train with masked PPO in self-play,
- and add opponent-pool / league training after the first baseline works.

Recommended model structure:

```text
observation -> shared encoder -> latent embedding -> phase-specific policy head
                                       \
                                        -> value head
```

Recommended training setup:
- one shared policy reused across all four player seats,
- legal-action masking at every step,
- self-play first,
- then opponent-pool training against historical checkpoints and scripted bots.

Recommended reward setup:
- log many reward components,
- combine them through a configurable scalarization layer,
- keep `win` dominant,
- and compare sparse, placement-based, and shaped reward presets.

## Why This Approach Fits Catan

Catan is not like chess.

Key differences:
- 4 players instead of 2,
- stochastic dice and shuffled development cards,
- imperfect information,
- general-sum rather than clean zero-sum dynamics,
- domestic trade interaction,
- and many distinct action subspaces depending on phase.

Because of that, an AlphaZero-style setup is not the best place to start.

A better first baseline is:
- policy-gradient RL,
- masked action spaces,
- self-play,
- and a stable structured observation interface.

## Research Context

### Strong baseline family

The most practical first family is:
- PPO or Masked PPO,
- in multi-agent self-play,
- with fixed discrete action indexing and legal-action masking.

Why:
- it is easy to integrate with the current environment,
- it handles large legal/illegal action spaces well when masking is added,
- it is much easier to debug than search-heavy methods,
- and it supports iterative experimentation on observations and rewards.

### Why not start with AlphaZero / MCTS

AlphaZero-style training works best in environments that are:
- mostly deterministic,
- 2-player,
- perfect information,
- and zero-sum.

Catan does not fit that cleanly.

Problems:
- hidden cards,
- stochastic rolls,
- 4-player non-zero-sum interaction,
- large branching factor,
- and negotiation/trading.

That does not mean search is useless, only that it is not the best first training system.

### Why not start with Deep CFR / ReBeL

Deep CFR and ReBeL are important references for imperfect-information games, but they are not the easiest first implementation for this problem.

Reasons:
- their strongest use cases are closer to 2-player zero-sum games,
- engineering complexity is much higher,
- and a working masked-PPO baseline is much more likely to produce a useful agent quickly.

Inference from the literature:
- these methods are worth revisiting later if the PPO baseline plateaus or if hidden-information reasoning becomes the dominant bottleneck.

## Sources

Primary references used for this plan:
- OpenSpiel intro: <https://openspiel.readthedocs.io/en/latest/intro.html>
- OpenSpiel information-state tensor docs: <https://openspiel.readthedocs.io/en/stable/api_reference/state_information_state_tensor.html>
- OpenAI Spinning Up PPO docs: <https://spinningup.openai.com/en/latest/algorithms/ppo.html>
- Invalid action masking paper: <https://arxiv.org/abs/2006.14171>
- SB3 Maskable PPO docs: <https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html>
- PettingZoo action masking tutorial: <https://pettingzoo.farama.org/tutorials/custom_environment/3-action-masking/>
- RLlib multi-agent env docs: <https://docs.ray.io/en/latest/rllib/multi-agent-envs.html>
- RLlib RLModule docs: <https://docs.ray.io/en/latest/rllib/rl-modules.html>
- Deep CFR: <https://proceedings.mlr.press/v97/brown19b.html>
- ReBeL: <https://proceedings.neurips.cc/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html>
- Potential-based reward shaping: <https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf>

## Proposed Learning Stack

## Environment API

The engine should expose a training-facing API that is stable and minimal.

Recommended interface:

```python
obs = env.observe(player_id)
action_mask = env.legal_action_mask(player_id)
next_obs, reward, done, info = env.step(action)
```

If we later integrate with libraries directly, the best fit is probably one of:
- PettingZoo AEC or parallel multi-agent API,
- RLlib `MultiAgentEnv`,
- or a thin custom wrapper with the same semantics.

Important principle:
- the agent should receive the acting player's observation,
- not the full omniscient simulator state,
- unless we intentionally build a cheating oracle baseline for comparison.

## Policy / Value Model

Recommended structure:
- shared encoder,
- one policy head per phase,
- one value head.

Architecture sketch:

```text
observation
  -> shared encoder
  -> latent state embedding
       -> policy_head_turn_start
       -> policy_head_main
       -> policy_head_robber_discard
       -> policy_head_robber_move
       -> policy_head_trade_response
       -> policy_head_road_building
       -> value head
```

Why phase-specific heads are a good fit:
- the meaning of actions changes a lot by phase,
- phases have very different action counts and semantics,
- and phase heads reduce wasted capacity compared with one enormous undifferentiated output layer.

## Action Space Strategy

Use a fixed global action index space plus a legal-action mask.

This is important.

Catan has many parameterized actions, but the set of legal actions at any step is sparse. A mask prevents the policy from wasting probability on impossible moves.

### Example fixed action families

Recommended action-index families:
- `roll_dice`
- `end_turn`
- `build_settlement@intersection_id`
- `build_city@intersection_id`
- `build_road@edge_id`
- `build_road_free@edge_id`
- `move_robber@hex_id@victim_id_or_none`
- `discard_combo@index`
- `buy_dev_card`
- `play_knight@hex_id@victim_id_or_none`
- `play_road_building`
- `play_year_of_plenty@r1@r2`
- `play_monopoly@resource`
- `maritime_trade@give@Resource@get@Resource@rate`
- `propose_trade@give@Resource@get@Resource`
- `accept_trade`
- `decline_trade`

A large fixed action space is acceptable if:
- all actions are indexed deterministically,
- and the mask zeros out illegal ones.

## Observation Design

The policy should consume only what the acting player is allowed to know.

### Observation blocks

Recommended observation blocks:

1. Global state features
- active player id
- current phase
- turn number
- whether the dice have already been rolled this turn
- whether a development card has been played this turn
- last roll
- robber hex id
- longest road owner
- largest army owner
- pending trade state

2. Hex features, shape `19 x F_hex`
- terrain type one-hot
- token number encoding
- robber flag
- optional production-weight feature
- optional harbor-adjacency summary

3. Intersection features, shape `54 x F_intersection`
- owner one-hot or empty
- structure type: empty / settlement / city
- coastal flag
- optional adjacent hex summaries

4. Edge features, shape `72 x F_edge`
- road owner one-hot or empty
- optional endpoint metadata
- coastal flag if useful

5. Player summary features, shape `4 x F_player`
For all players:
- visible VP
- resource count total
- development card count total
- roads remaining
- settlements remaining
- cities remaining
- played Knights
For the acting player only:
- exact resource hand
- exact dev-card hand

6. Legal action mask, shape `N_actions`
- binary mask over fixed action indices

### Important information-design rule

Do not give the model opponents' exact hands in the standard training observation.

Catan is imperfect-information. Opponent exact resources and exact development cards should remain hidden unless we intentionally create an oracle baseline.

## Shared Encoder Design

The first encoder should be structured and simple.

Recommended baseline encoder:
- encode hexes with a small MLP,
- encode intersections with a small MLP,
- encode edges with a small MLP,
- encode player summaries with a small MLP,
- pool each entity set,
- concatenate pooled features with global features,
- pass through a final shared MLP trunk.

### Baseline encoder sketch

```text
global features -----------------------\
hex set -> per-hex MLP -> pool --------\
intersection set -> per-node MLP -> pool -> concat -> shared trunk -> latent
edge set -> per-edge MLP -> pool ------/
player set -> per-player MLP -> flatten/
```

Why start here instead of a graph neural network:
- much easier to implement,
- easier to debug,
- likely good enough for a first strong baseline,
- and gives us a clean benchmark before adding architectural complexity.

### When to consider a graph encoder later

Move to a GNN if:
- the pooled-MLP baseline plateaus,
- board-topology reasoning appears to be the bottleneck,
- or we want stronger generalization across road/intersection structures.

A future graph design could use message passing across:
- hex <-> intersections,
- intersections <-> edges,
- or a unified graph over intersections and edges plus hex context.

## Training Methodology

## Phase 1: Single shared policy, self-play baseline

Start with:
- one policy shared across all 4 seats,
- self-play,
- randomized board seeds,
- masked PPO,
- no architecture tricks beyond the shared encoder and phase heads.

Why:
- simplest path to a functioning baseline,
- easiest way to validate the observation and action schema,
- and easiest way to compare reward setups.

### Expected benefits
- automatic curriculum through self-play,
- reduced maintenance compared with hand-coded opponents only,
- and stable seat symmetry if all players share weights.

## Phase 2: Opponent-pool training

After the first self-play baseline works, introduce an opponent pool.

Pool members should include:
- the current main policy,
- historical checkpoints,
- scripted heuristic bots,
- and possibly random or weak baselines.

Why:
- reduces overfitting to one co-evolving opponent population,
- creates a broader curriculum,
- and helps robustness.

## Phase 3: League-style orchestration

After opponent pools work, move toward a simple league.

Suggested league roles:
- current learning policy,
- frozen milestone policies,
- exploiters or diversity policies later if needed,
- scripted baselines for sanity checks.

This does not need to be as complex as AlphaStar-style league training initially. Even a rotating evaluation and training opponent pool is enough to get the benefits.

## Training Loop

Recommended rollout/update loop:

1. Run many parallel environments.
2. Collect acting-player observations, masks, actions, rewards, and terminal outcomes.
3. Update the current policy with masked PPO.
4. Snapshot checkpoints every `N` updates.
5. Add or refresh historical checkpoints in the opponent pool.
6. Periodically evaluate on a fixed benchmark suite.
7. Promote only if evaluation improves on target metrics.

## Evaluation Methodology

Do not evaluate only on training reward.

Track benchmark performance on:
- 1st-place rate,
- average placement,
- average final VP,
- average game length,
- win rate versus random bots,
- win rate versus heuristic bots,
- win rate versus recent checkpoints,
- and game-behavior metrics like trade use, robber targeting, and build mix.

Recommended benchmark suite:
- fixed random seeds,
- fixed board sets,
- several opponent mixtures,
- and a stable evaluation configuration so checkpoint comparisons stay meaningful.

## Reward System Design

This is worth designing carefully now.

The cleanest pattern is:
- log many reward components,
- keep them separate,
- and compute the scalar training reward from a config file.

### Why this helps

This lets us:
- try sparse rewards without changing simulator code,
- compare reward schemes cleanly,
- track agent incentives directly,
- and avoid losing useful diagnostics when we change reward weights.

## Reward components to log

Suggested reward components:
- `win`
- `placement_score`
- `final_vp`
- `vp_gain`
- `longest_road_gain`
- `largest_army_gain`
- `roads_built`
- `settlements_built`
- `cities_built`
- `dev_cards_bought`
- `dev_cards_played`
- `cards_gained`
- `cards_stolen`
- `trade_offer_accepted`
- `trade_offer_declined`
- `resources_discarded`
- `robber_steals`
- `turn_penalty`
- `game_length_penalty`
- `moves_taken`

These should be logged even if their weight is zero.

## Reward scalarization

Suggested reward config structure:

```yaml
reward:
  win: 1.0
  placement_score: 0.25
  final_vp: 0.05
  vp_gain: 0.02
  longest_road_gain: 0.05
  largest_army_gain: 0.05
  roads_built: 0.003
  settlements_built: 0.01
  cities_built: 0.02
  dev_cards_bought: 0.003
  dev_cards_played: 0.004
  cards_gained: 0.001
  cards_stolen: 0.002
  turn_penalty: -0.0005
  game_length_penalty: -0.0002
  moves_taken: -0.0001
```

Then compute scalar reward as:

```python
reward_t = sum(weight[name] * component[name] for name in weights)
```

### Important design guidance

Keep `win` dominant.

Shaping rewards should help learning, not replace the real objective.

If shaping terms become too large, the agent may learn behaviors that look active or clever but do not improve actual win rate.

## Recommended reward presets

We should support multiple reward presets from day one.

### Preset 1: Sparse

```yaml
reward:
  win: 1.0
```

Use this to measure alignment with the real objective.

### Preset 2: Placement-based

```yaml
reward:
  placement_score: 1.0
```

Example placement mapping:
- 1st = `1.0`
- 2nd = `0.5`
- 3rd = `0.25`
- 4th = `0.0`

This can be helpful in 4-player games because it gives denser end-of-episode feedback than pure win/loss.

### Preset 3: Shaped

```yaml
reward:
  win: 1.0
  placement_score: 0.25
  final_vp: 0.05
  vp_gain: 0.02
  settlements_built: 0.01
  cities_built: 0.02
  dev_cards_played: 0.003
  turn_penalty: -0.0005
```

This should be conservative. The goal is to help exploration, not redefine success.

## Reward-shaping methodology

The strongest theory-backed shaping family is potential-based shaping.

In practice, some useful Catan metrics like card gain or dev-card usage are not guaranteed to be policy-invariant, so they should be treated as pragmatic experimental knobs rather than theoretically neutral shaping.

That is okay, as long as we:
- log everything clearly,
- compare against sparse reward baselines,
- and evaluate by actual game outcomes, not just training reward.

## Orchestration Plan

## Experiment management

The training system should treat these as first-class configurable units:
- model architecture version,
- observation schema version,
- action schema version,
- reward preset,
- self-play / pool-play mode,
- checkpoint interval,
- evaluation suite.

### Suggested config families
- `env_config.yaml`
- `model_config.yaml`
- `reward_config.yaml`
- `train_config.yaml`
- `eval_config.yaml`

## Checkpointing

Checkpoint regularly and track metadata:
- training step,
- wall-clock time,
- reward preset,
- policy architecture,
- evaluation summary,
- opponent pool composition.

## Evaluation reports

Each major checkpoint should produce a compact report with:
- 1st-place rate,
- average placement,
- average final VP,
- average turns,
- win rate against random,
- win rate against heuristic,
- win rate against previous main,
- reward-component averages.

## Staged Research Plan

## Stage A: RL-ready environment interface

Build the environment wrappers and schemas first.

Deliverables:
- stable observation schema,
- stable action indexing,
- legal action mask,
- per-step `info` dictionary with reward components,
- deterministic seeding behavior.

## Stage B: Baseline masked PPO

Deliverables:
- shared encoder,
- phase-specific heads,
- value head,
- self-play training loop,
- sparse / placement / shaped reward presets,
- benchmark evaluation suite.

Success criteria:
- beats random bots consistently,
- learns legal and coherent action sequences,
- and shows monotonic improvement on benchmark metrics.

## Stage C: Opponent-pool training

Deliverables:
- historical checkpoint pool,
- heuristic bots in evaluation and training,
- sampling strategy for opponents,
- checkpoint promotion logic.

Success criteria:
- better robustness against older policies and baselines,
- less collapse to narrow self-play metas.

## Stage D: Better encoders and memory

Only do this after Stage B and C are stable.

Options:
- graph encoder,
- attention over entities,
- recurrent memory or short history window,
- belief-state features derived from public action history.

## Recommended Immediate Next Steps

1. Freeze a versioned observation schema for the acting player.
2. Freeze a versioned fixed action-index mapping.
3. Add a legal-action mask export method if it does not already exist in the exact format training code will need.
4. Add reward-component logging in `info` rather than hardcoding one scalar reward.
5. Build a minimal training wrapper around masked PPO.
6. Add random and heuristic scripted baselines for evaluation.
7. Run reward-preset ablations before touching more advanced architectures.

## Strong Practical Recommendation

If the goal is to get a real learning system working with the highest chance of success, start with:
- structured pooled-MLP encoder,
- masked PPO,
- one shared policy across all players,
- configurable reward presets,
- and opponent-pool training only after the baseline works.

That gives us:
- the simplest reliable first implementation,
- the clearest debugging path,
- and a solid benchmark before we spend effort on graph models or more advanced imperfect-information methods.
