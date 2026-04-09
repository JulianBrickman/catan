# Catan Eval Metrics Guide

## Purpose

This document explains how to interpret the evaluation metrics produced by `train_catan_ppo.py`.

These metrics are meant to answer a simple question:
- is the policy actually getting better at finishing and winning games,
- or is it still wandering, stalling, or exploiting weak reward signals?

## Most Important Eval Metrics

### `eval_first_place`

This is the most important top-level metric right now.

Meaning:
- the fraction of evaluation episodes where the agent finishes in 1st place

How to read it:
- higher is better
- this is the clearest direct performance signal
- if this rises over time, the policy is usually becoming stronger

Notes:
- with only a few eval episodes, this metric is noisy
- it becomes much more meaningful when averaged over more games

## `eval_truncations`

This is extremely important.

Meaning:
- how many evaluation episodes hit the `max_turns` cap instead of ending naturally

How to read it:
- lower is better
- high truncation means the policy is not finishing games cleanly
- if truncations are frequent, the model may be stalling or taking weak low-progress actions

Interpretation:
- high `eval_truncations` + low `eval_first_place` is a warning sign
- high `eval_truncations` + rising `eval_avg_final_vp` can still mean progress, but the policy is not yet converting games well

## `eval_avg_final_vp`

This is one of the most useful supporting metrics.

Meaning:
- the average final victory points across all players at the end of evaluation episodes

How to read it:
- higher is generally better
- this tells us whether games are progressing toward meaningful board states
- it is especially useful when games truncate

Interpretation:
- low `eval_avg_final_vp` + high truncations means the games are barely developing
- moderate or high `eval_avg_final_vp` + some truncations means the games are progressing, but not closing reliably yet

## `eval_avg_turns`

Meaning:
- the average number of turns in evaluation episodes

How to read it:
- this metric is contextual, not purely good or bad
- lower is not always better
- very high values matter mostly when combined with low VP progress or high truncation

Interpretation:
- high `eval_avg_turns` + high `eval_truncations` + low `eval_avg_final_vp` usually means weak play or stalling
- moderate `eval_avg_turns` + good `eval_first_place` is usually healthier

## `winner_counts`

Meaning:
- how many evaluation episodes were won by each player id, or by nobody

How to read it:
- useful for spotting instability or bias
- if `none` appears often, too many eval games are truncating
- if only one seat dominates repeatedly, that may indicate seed noise or unintended seat bias

## Best Way To Read The Metrics Together

### Healthy learning pattern

A good trend looks like this:
- `eval_first_place` goes up
- `eval_truncations` goes down
- `eval_avg_final_vp` goes up
- `winner_counts["none"]` goes down

This combination suggests the agent is learning to both progress the game and actually finish it.

### Warning pattern

A concerning pattern looks like this:
- `eval_first_place` stays noisy or flat
- `eval_truncations` stays high
- `eval_avg_final_vp` stays low
- `eval_avg_turns` stays near the turn cap

This usually means:
- the policy is still weak,
- games are stalling,
- or the reward shaping is not pushing enough toward point-gaining and game-closing behavior.

### Mixed but improving pattern

This pattern can still be okay early on:
- `eval_truncations` still happen
- but `eval_avg_final_vp` rises
- and `eval_first_place` improves slowly

This usually means:
- the policy is learning to progress the board,
- but has not yet learned to consistently convert that progress into finished wins.

## Practical Priority Order

If you want a simple priority order for reading the metrics, use this:

1. `eval_first_place`
2. `eval_truncations`
3. `eval_avg_final_vp`
4. `winner_counts`
5. `eval_avg_turns`

Why:
- 1st place is the clearest success signal
- truncation tells you whether games are finishing
- final VP tells you whether progress is real even when games do not finish
- winner distribution helps catch odd behavior
- average turns is useful, but only in context

## Examples

### Good example

- `eval_first_place = 0.50`
- `eval_truncations = 0`
- `eval_avg_final_vp = 8.2`

Interpretation:
- the policy is finishing games and getting into strong point totals
- this is a promising result

### Bad example

- `eval_first_place = 0.00`
- `eval_truncations = 4`
- `eval_avg_final_vp = 2.5`
- `eval_avg_turns = 1000`

Interpretation:
- the model is not finishing games
- it is likely wandering or making very weak progress

### Mixed early-training example

- `eval_first_place = 0.25`
- `eval_truncations = 2`
- `eval_avg_final_vp = 5.8`

Interpretation:
- the policy is not consistently strong yet,
- but it is making meaningful progress in the board state
- worth continuing training and watching whether truncations fall

## Recommendation For Current Project Stage

Right now, the best way to evaluate new runs is:
- check `eval_first_place`
- check `eval_truncations`
- check `eval_avg_final_vp`
- open the saved `final_state.svg`
- read `player_breakdown.md`

Why:
- numbers tell you whether learning is trending in the right direction
- the SVG and player breakdown tell you whether the resulting game states actually look sensible

## Bottom Line

The single best summary rule is:
- better agents should win more often,
- truncate less often,
- and end games at higher final VP totals.

If only one of those improves, the model may still be learning.
If none of them improve, the training setup probably needs adjustment.
