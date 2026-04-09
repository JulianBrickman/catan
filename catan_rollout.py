from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from catan import ActionError, GameState
from catan_model import CatanPolicyValueNet, action_features_to_torch, tensor_obs_to_torch
from catan_rl_env import CatanEnvConfig, CatanRLEnv


@dataclass
class RolloutStep:
    player_id: int
    phase: str
    action_index: int
    reward: float
    done: bool
    value: float
    log_prob: float
    entropy: float
    legal_action_count: int
    action: Dict[str, object]
    reward_components: Dict[str, float]
    tensor_obs: Dict[str, object]
    action_features: List[List[float]]
    legal_action_mask: List[int]


@dataclass
class RolloutEpisode:
    steps: List[RolloutStep]
    winner: Optional[int]
    placements: Optional[Dict[int, int]]
    turns: int
    truncated: bool = False
    final_state: Optional[GameState] = None
    final_observation: Optional[Dict[str, object]] = None
    final_event_log: Optional[List[str]] = None


class CatanRolloutCollector:
    def __init__(
        self,
        model: CatanPolicyValueNet,
        env_config: Optional[CatanEnvConfig] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.env_config = env_config or CatanEnvConfig()
        self.device = device

    def run_episode(self, seed: Optional[int] = None, deterministic: bool = False) -> RolloutEpisode:
        env = CatanRLEnv(self.env_config)
        obs = env.reset(seed=seed)
        steps: List[RolloutStep] = []

        while True:
            if env.state.pending_phase == "game_over" or env.state.winner is not None or not obs["legal_actions"]:
                return RolloutEpisode(
                    steps=steps,
                    winner=env.state.winner,
                    placements=env.compute_placements() if env.state.players else None,
                    turns=env.state.turn_number,
                    truncated=bool(env.config.max_turns is not None and env.state.turn_number >= env.config.max_turns),
                    final_state=env.state,
                    final_observation=obs,
                    final_event_log=list(env.state.event_log),
                )
            tensor_obs = tensor_obs_to_torch(obs["tensor_obs"], device=self.device)
            action_features = action_features_to_torch(obs["action_features"], device=self.device)
            action_mask = torch.tensor([obs["legal_action_mask"]], dtype=torch.float32, device=self.device)
            phase = obs["global"]["phase"]
            sample = self.model.sample_action(
                tensor_obs,
                action_features,
                phase=phase,
                action_mask=action_mask,
                deterministic=deterministic,
            )
            action_index = int(sample["action_index"].item())
            try:
                next_obs, reward, done, info = env.step(action_index)
            except ActionError as exc:
                if str(exc) != "The game is already over.":
                    raise
                return RolloutEpisode(
                    steps=steps,
                    winner=env.state.winner,
                    placements=env.compute_placements() if env.state.players else None,
                    turns=env.state.turn_number,
                    truncated=bool(env.config.max_turns is not None and env.state.turn_number >= env.config.max_turns),
                    final_state=env.state,
                    final_observation=obs,
                    final_event_log=list(env.state.event_log),
                )
            steps.append(
                RolloutStep(
                    player_id=info["acting_player"],
                    phase=phase,
                    action_index=action_index,
                    reward=float(reward),
                    done=bool(done),
                    value=float(sample["value"].item()),
                    log_prob=float(sample["log_prob"].item()),
                    entropy=float(sample["entropy"].item()),
                    legal_action_count=len(obs["legal_actions"]),
                    action=info["action"],
                    reward_components=dict(info["reward_components"]),
                    tensor_obs=obs["tensor_obs"],
                    action_features=obs["action_features"],
                    legal_action_mask=list(obs["legal_action_mask"]),
                )
            )
            obs = next_obs
            if done:
                return RolloutEpisode(
                    steps=steps,
                    winner=info.get("winner"),
                    placements=info.get("placements"),
                    turns=env.state.turn_number,
                    truncated=bool(info.get("truncated", False)),
                    final_state=env.state,
                    final_observation=obs,
                    final_event_log=list(env.state.event_log),
                )
