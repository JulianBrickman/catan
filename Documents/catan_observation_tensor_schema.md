# Catan Observation Tensor Schema

## Purpose

This document freezes the first encoder-facing observation contract for the RL environment in `catan_rl_env.py`.

Current schema versions:
- observation schema: `v1`
- tensor schema: `v1`
- action indexing mode: `phase_local`

This schema is designed for the shared encoder stage of the project.

## Design Goals

The schema should be:
- stable enough to build a model against,
- faithful to imperfect information,
- explicit about feature ordering,
- easy to tensorize in PyTorch,
- and simple enough to debug before we optimize architecture.

## High-Level Layout

`env.observe(player_id)` returns both structured fields and tensorized fields.

Relevant top-level keys:
- `global`
- `hexes`
- `intersections`
- `edges`
- `players`
- `private`
- `legal_actions`
- `legal_action_mask`
- `tensor_spec`
- `tensor_obs`

The shared encoder should use `tensor_obs`.

## Tensor Blocks

### `global_features`

Shape:
- `[21]`

Feature order:
1. `observer_player_id_norm`
2. `active_player_id_norm`
3. `turn_number_norm`
4. `last_roll_norm`
5. `robber_hex_id_norm`
6. `rolled_this_turn`
7. `dev_card_played_this_turn`
8. `has_winner`
9. `winner_id_norm`
10. `longest_road_owner_norm`
11. `largest_army_owner_norm`
12. `pending_trade_responder_norm`
13. `phase_setup_settlement`
14. `phase_setup_road`
15. `phase_turn_start`
16. `phase_robber_discard`
17. `phase_robber_move`
18. `phase_trade_response`
19. `phase_main`
20. `phase_road_building`
21. `phase_game_over`

Notes:
- normalized player ids use `-1.0` when absent
- phase is encoded both as scalar index in the structured `global` block and as one-hot here

### `hex_features`

Shape:
- `[19, 20]`

Feature order:
1. `terrain_desert`
2. `terrain_hills`
3. `terrain_forest`
4. `terrain_mountains`
5. `terrain_fields`
6. `terrain_pasture`
7. `resource_none`
8. `resource_brick`
9. `resource_lumber`
10. `resource_ore`
11. `resource_grain`
12. `resource_wool`
13. `token_norm`
14. `token_pips_norm`
15. `has_robber`
16. `q_norm`
17. `r_norm`
18. `x_norm`
19. `y_norm`
20. `is_coastal_hex`

### `intersection_features`

Shape:
- `[54, 16]`

Feature order:
1. `owner_none`
2. `owner_p0`
3. `owner_p1`
4. `owner_p2`
5. `owner_p3`
6. `piece_empty`
7. `piece_settlement`
8. `piece_city`
9. `coastal`
10. `adjacent_hex_count_norm`
11. `adjacent_edge_count_norm`
12. `adjacent_intersection_count_norm`
13. `is_observer_owned`
14. `is_enemy_owned`
15. `x_norm`
16. `y_norm`

### `edge_features`

Shape:
- `[72, 9]`

Feature order:
1. `owner_none`
2. `owner_p0`
3. `owner_p1`
4. `owner_p2`
5. `owner_p3`
6. `coastal`
7. `adjacent_hex_count_norm`
8. `is_observer_owned`
9. `is_enemy_owned`

### `player_features`

Shape:
- `[4, 11]`

Feature order:
1. `is_observer`
2. `turn_order_relative_norm`
3. `visible_vp_norm`
4. `total_resource_count_norm`
5. `total_dev_card_count_norm`
6. `roads_remaining_norm`
7. `settlements_remaining_norm`
8. `cities_remaining_norm`
9. `played_knights_norm`
10. `has_longest_road`
11. `has_largest_army`

Important imperfect-information rule:
- opponent rows do not include exact resource-card counts by type
- opponent rows do not include exact development-card identities
- only public summary information is exposed here

### `private_features`

Shape:
- `[15]`

Feature order:
1. `resource_brick_count_norm`
2. `resource_lumber_count_norm`
3. `resource_ore_count_norm`
4. `resource_grain_count_norm`
5. `resource_wool_count_norm`
6. `dev_knight_count_norm`
7. `dev_road_building_count_norm`
8. `dev_year_of_plenty_count_norm`
9. `dev_monopoly_count_norm`
10. `dev_victory_point_count_norm`
11. `new_dev_knight_count_norm`
12. `new_dev_road_building_count_norm`
13. `new_dev_year_of_plenty_count_norm`
14. `new_dev_monopoly_count_norm`
15. `new_dev_victory_point_count_norm`

## Legal Actions

Current action mode:
- `phase_local`

Meaning:
- action indices are local to the current legal action list
- `step(action_index)` selects from `legal_actions[action_index]`
- this is not yet the final fixed global action-head scheme

Why this is acceptable for now:
- it lets us freeze the encoder input contract first
- it keeps the wrapper usable for early rollout collection
- it avoids forcing a premature global index design for discard combinatorics and other phase-specific actions

## Intended Shared Encoder Usage

Recommended v1 encoder inputs:
- `global_features`
- `hex_features`
- `intersection_features`
- `edge_features`
- `player_features`
- `private_features`

Recommended architecture:
- small per-entity MLP for `hex_features`
- small per-entity MLP for `intersection_features`
- small per-entity MLP for `edge_features`
- small per-player MLP for `player_features`
- one MLP for `global_features`
- one MLP for `private_features`
- concatenate pooled outputs into one shared trunk

## Notes On Stability

What is frozen now:
- block names
- block shapes
- feature ordering within each block
- private/public split

What is intentionally not frozen yet:
- final policy action-head indexing
- reward-weight presets
- model hidden sizes
- sequence/history augmentation
- graph neural network migration

## Source Of Truth

The runtime source of truth is:
- `CatanRLEnv.tensor_spec()` in `catan_rl_env.py`

If this document and code ever diverge, prefer the code and then update this file.
