from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from catan_model import CatanPolicyValueNet, action_features_to_torch, tensor_obs_to_torch
from catan_rl_env import CatanEnvConfig
from catan_rollout import CatanRolloutCollector, RolloutEpisode, RolloutStep


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_episodes: int = 4
    update_epochs: int = 4
    minibatch_size: int = 32
    normalize_advantages: bool = True
    checkpoint_dir: str = "checkpoints"
    device: str = "cpu"


@dataclass
class PPOBatch:
    tensor_obs: Dict[str, Tensor]
    action_features: Tensor
    action_mask: Tensor
    action_index: Tensor
    old_log_prob: Tensor
    advantages: Tensor
    returns: Tensor
    value_targets: Tensor
    phases: List[str]


@dataclass
class TrainingMetrics:
    episodes_collected: int
    steps_collected: int
    average_episode_reward: float
    average_episode_turns: float
    winner_counts: Dict[str, int]
    truncation_count: int
    average_first_place_rate: float
    average_final_placement_score: float
    average_final_vp: float
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    explained_variance: float


@dataclass
class EvaluationMetrics:
    episodes_evaluated: int
    steps_evaluated: int
    average_episode_reward: float
    average_episode_turns: float
    winner_counts: Dict[str, int]
    truncation_count: int
    average_first_place_rate: float
    average_final_placement_score: float
    average_final_vp: float


class StepDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, object]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.records[index]


class CatanPPOTrainer:
    def __init__(
        self,
        model: CatanPolicyValueNet,
        env_config: Optional[CatanEnvConfig] = None,
        ppo_config: Optional[PPOConfig] = None,
    ) -> None:
        self.model = model
        self.env_config = env_config or CatanEnvConfig()
        self.ppo_config = ppo_config or PPOConfig()
        self.device = torch.device(self.ppo_config.device)
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.ppo_config.learning_rate)
        self.collector = CatanRolloutCollector(model, self.env_config, device=self.device)
        self.update_step = 0

    def collect_episodes(
        self,
        num_episodes: Optional[int] = None,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ) -> List[RolloutEpisode]:
        episode_count = self.ppo_config.rollout_episodes if num_episodes is None else num_episodes
        episodes: List[RolloutEpisode] = []
        base_seed = seed if seed is not None else self.env_config.seed
        for episode_index in range(episode_count):
            episode_seed = None if base_seed is None else base_seed + episode_index
            episodes.append(self.collector.run_episode(seed=episode_seed, deterministic=deterministic))
        return episodes

    def evaluate(self, num_episodes: int = 4, seed: Optional[int] = None, deterministic: bool = True) -> EvaluationMetrics:
        episodes = self.collect_episodes(num_episodes=num_episodes, seed=seed, deterministic=deterministic)
        return summarize_evaluation_metrics(episodes)

    def prepare_batch(self, episodes: Sequence[RolloutEpisode]) -> PPOBatch:
        records: List[Dict[str, object]] = []
        for episode in episodes:
            rewards = [step.reward for step in episode.steps]
            values = [step.value for step in episode.steps]
            dones = [step.done for step in episode.steps]
            advantages, returns = compute_gae(
                rewards,
                values,
                dones,
                gamma=self.ppo_config.gamma,
                gae_lambda=self.ppo_config.gae_lambda,
            )
            for step, advantage, ret in zip(episode.steps, advantages, returns):
                records.append(
                    {
                        "tensor_obs": step.tensor_obs,
                        "action_features": step.action_features,
                        "action_mask": step.legal_action_mask,
                        "action_index": step.action_index,
                        "old_log_prob": step.log_prob,
                        "advantage": advantage,
                        "return": ret,
                        "value_target": ret,
                        "phase": step.phase,
                    }
                )

        if not records:
            raise ValueError("Cannot prepare PPO batch from zero rollout steps.")

        if self.ppo_config.normalize_advantages:
            advantages_tensor = torch.tensor([record["advantage"] for record in records], dtype=torch.float32)
            normalized = normalize_tensor(advantages_tensor)
            for record, value in zip(records, normalized.tolist()):
                record["advantage"] = value

        collated = collate_step_records(records, device=self.device)
        return PPOBatch(**collated)

    def train_update(self, episodes: Sequence[RolloutEpisode]) -> TrainingMetrics:
        batch = self.prepare_batch(episodes)
        records = split_batch_to_records(batch)
        dataset = StepDataset(records)
        dataloader = DataLoader(
            dataset,
            batch_size=min(self.ppo_config.minibatch_size, len(dataset)),
            shuffle=True,
            collate_fn=lambda samples: collate_step_records(samples, device=self.device),
        )

        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        total_losses: List[float] = []

        for _ in range(self.ppo_config.update_epochs):
            for minibatch in dataloader:
                losses = self._ppo_loss(PPOBatch(**minibatch))
                self.optimizer.zero_grad()
                losses["total_loss"].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo_config.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(losses["policy_loss"].item()))
                value_losses.append(float(losses["value_loss"].item()))
                entropies.append(float(losses["entropy"].item()))
                total_losses.append(float(losses["total_loss"].item()))

        self.update_step += 1
        with torch.no_grad():
            predicted_values = evaluate_phase_local_batch(
                self.model,
                batch.tensor_obs,
                batch.action_features,
                batch.action_mask,
                batch.phases,
            )["value"]
        return summarize_training_metrics(
            episodes,
            average(policy_losses),
            average(value_losses),
            average(entropies),
            average(total_losses),
            explained_variance_from_batch(batch.value_targets, predicted_values),
        )

    def save_checkpoint(self, path: str | Path, extra: Optional[Dict[str, object]] = None) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ppo_config": asdict(self.ppo_config),
            "env_config": asdict(self.env_config),
            "update_step": self.update_step,
            "extra": extra or {},
        }
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, path: str | Path, load_optimizer: bool = True) -> Dict[str, object]:
        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.update_step = int(payload.get("update_step", 0))
        return payload

    def _ppo_loss(self, batch: PPOBatch) -> Dict[str, Tensor]:
        outputs = evaluate_phase_local_batch(
            self.model,
            batch.tensor_obs,
            batch.action_features,
            batch.action_mask,
            batch.phases,
        )
        dist = torch.distributions.Categorical(logits=outputs["policy_logits"])
        new_log_prob = dist.log_prob(batch.action_index)
        entropy = dist.entropy().mean()
        ratio = torch.exp(new_log_prob - batch.old_log_prob)
        unclipped = ratio * batch.advantages
        clipped = torch.clamp(ratio, 1.0 - self.ppo_config.clip_epsilon, 1.0 + self.ppo_config.clip_epsilon) * batch.advantages
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = F.mse_loss(outputs["value"], batch.value_targets)
        total_loss = policy_loss + self.ppo_config.value_coef * value_loss - self.ppo_config.entropy_coef * entropy
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "total_loss": total_loss,
        }


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[List[float], List[float]]:
    advantages = [0.0] * len(rewards)
    returns = [0.0] * len(rewards)
    next_advantage = 0.0
    next_value = 0.0
    for index in reversed(range(len(rewards))):
        done = float(dones[index])
        delta = rewards[index] + gamma * next_value * (1.0 - done) - values[index]
        next_advantage = delta + gamma * gae_lambda * (1.0 - done) * next_advantage
        advantages[index] = next_advantage
        returns[index] = advantages[index] + values[index]
        next_value = values[index]
    return advantages, returns


def normalize_tensor(values: Tensor, eps: float = 1e-8) -> Tensor:
    return (values - values.mean()) / (values.std(unbiased=False) + eps)


def collate_step_records(records: Sequence[Dict[str, object]], device: torch.device | str) -> Dict[str, object]:
    tensor_keys = [
        "global_features",
        "hex_features",
        "intersection_features",
        "edge_features",
        "player_features",
        "private_features",
    ]
    tensor_obs: Dict[str, Tensor] = {}
    for key in tensor_keys:
        tensors = [torch.tensor(record["tensor_obs"][key], dtype=torch.float32, device=device) for record in records]
        tensor_obs[key] = torch.stack(tensors, dim=0)

    max_actions = max(len(record["action_features"]) for record in records)
    action_feature_dim = len(records[0]["action_features"][0])
    action_features = torch.zeros((len(records), max_actions, action_feature_dim), dtype=torch.float32, device=device)
    action_mask = torch.zeros((len(records), max_actions), dtype=torch.float32, device=device)
    for row_index, record in enumerate(records):
        action_rows = torch.tensor(record["action_features"], dtype=torch.float32, device=device)
        mask_row = torch.tensor(record["action_mask"], dtype=torch.float32, device=device)
        count = action_rows.shape[0]
        action_features[row_index, :count] = action_rows
        action_mask[row_index, :count] = mask_row

    return {
        "tensor_obs": tensor_obs,
        "action_features": action_features,
        "action_mask": action_mask,
        "action_index": torch.tensor([record["action_index"] for record in records], dtype=torch.long, device=device),
        "old_log_prob": torch.tensor([record["old_log_prob"] for record in records], dtype=torch.float32, device=device),
        "advantages": torch.tensor([record["advantage"] for record in records], dtype=torch.float32, device=device),
        "returns": torch.tensor([record["return"] for record in records], dtype=torch.float32, device=device),
        "value_targets": torch.tensor([record["value_target"] for record in records], dtype=torch.float32, device=device),
        "phases": [str(record["phase"]) for record in records],
    }


def split_batch_to_records(batch: PPOBatch) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    batch_size = batch.action_index.shape[0]
    for index in range(batch_size):
        valid_action_count = int(batch.action_mask[index].sum().item())
        records.append(
            {
                "tensor_obs": {key: value[index].tolist() for key, value in batch.tensor_obs.items()},
                "action_features": batch.action_features[index, :valid_action_count].tolist(),
                "action_mask": batch.action_mask[index, :valid_action_count].tolist(),
                "action_index": int(batch.action_index[index].item()),
                "old_log_prob": float(batch.old_log_prob[index].item()),
                "advantage": float(batch.advantages[index].item()),
                "return": float(batch.returns[index].item()),
                "value_target": float(batch.value_targets[index].item()),
                "phase": batch.phases[index],
            }
        )
    return records


def evaluate_phase_local_batch(
    model: CatanPolicyValueNet,
    tensor_obs: Dict[str, Tensor],
    action_features: Tensor,
    action_mask: Tensor,
    phases: Sequence[str],
) -> Dict[str, Tensor]:
    latent_outputs = model.encoder(tensor_obs)
    latent = latent_outputs["latent"]
    policy_logits = torch.empty(action_mask.shape, dtype=torch.float32, device=latent.device)
    for row_index, phase in enumerate(phases):
        row_logits = model.phase_heads[phase](latent[row_index : row_index + 1], action_features[row_index : row_index + 1])
        policy_logits[row_index] = row_logits.squeeze(0)
    policy_logits = policy_logits.masked_fill(action_mask <= 0, -1e9)
    value = model.value_head(latent).squeeze(-1)
    return {
        **latent_outputs,
        "policy_logits": policy_logits,
        "value": value,
    }


def summarize_training_metrics(
    episodes: Sequence[RolloutEpisode],
    policy_loss: float,
    value_loss: float,
    entropy: float,
    total_loss: float,
    explained_variance: float,
) -> TrainingMetrics:
    steps = [step for episode in episodes for step in episode.steps]
    rewards_by_episode = [sum(step.reward for step in episode.steps) for episode in episodes]
    winner_counts: Dict[str, int] = {}
    first_place_count = 0
    placement_scores: List[float] = []
    final_vps: List[float] = []
    truncation_count = 0
    placement_map = {1: 1.0, 2: 0.5, 3: 0.25, 4: 0.0}
    for episode in episodes:
        truncation_count += 1 if episode.truncated else 0
        winner_key = "none" if episode.winner is None else str(episode.winner)
        winner_counts[winner_key] = winner_counts.get(winner_key, 0) + 1
        if episode.placements is not None:
            for player_id, place in episode.placements.items():
                if place == 1:
                    first_place_count += 1 if episode.winner == player_id else 0
                placement_scores.append(placement_map.get(place, 0.0))
        if episode.final_state is not None:
            for player in episode.final_state.players:
                final_vps.append(float(episode.final_state.total_victory_points(player.player_id)))
    return TrainingMetrics(
        episodes_collected=len(episodes),
        steps_collected=len(steps),
        average_episode_reward=average(rewards_by_episode),
        average_episode_turns=average([episode.turns for episode in episodes]),
        winner_counts=winner_counts,
        truncation_count=truncation_count,
        average_first_place_rate=(sum(1 for episode in episodes if episode.winner is not None) / max(1, len(episodes))),
        average_final_placement_score=average(placement_scores),
        average_final_vp=average(final_vps),
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        total_loss=total_loss,
        explained_variance=explained_variance,
    )


def summarize_evaluation_metrics(episodes: Sequence[RolloutEpisode]) -> EvaluationMetrics:
    steps = [step for episode in episodes for step in episode.steps]
    rewards_by_episode = [sum(step.reward for step in episode.steps) for episode in episodes]
    winner_counts: Dict[str, int] = {}
    placement_scores: List[float] = []
    final_vps: List[float] = []
    truncation_count = 0
    placement_map = {1: 1.0, 2: 0.5, 3: 0.25, 4: 0.0}
    first_place_wins = 0
    for episode in episodes:
        truncation_count += 1 if episode.truncated else 0
        winner_key = "none" if episode.winner is None else str(episode.winner)
        winner_counts[winner_key] = winner_counts.get(winner_key, 0) + 1
        if episode.placements is not None:
            for player_id, place in episode.placements.items():
                placement_scores.append(placement_map.get(place, 0.0))
                if place == 1 and episode.winner == player_id:
                    first_place_wins += 1
        if episode.final_state is not None:
            for player in episode.final_state.players:
                final_vps.append(float(episode.final_state.total_victory_points(player.player_id)))
    return EvaluationMetrics(
        episodes_evaluated=len(episodes),
        steps_evaluated=len(steps),
        average_episode_reward=average(rewards_by_episode),
        average_episode_turns=average([episode.turns for episode in episodes]),
        winner_counts=winner_counts,
        truncation_count=truncation_count,
        average_first_place_rate=first_place_wins / max(1, len(episodes)),
        average_final_placement_score=average(placement_scores),
        average_final_vp=average(final_vps),
    )


def average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def explained_variance_from_batch(targets: Tensor, predictions: Tensor) -> float:
    target_var = torch.var(targets, unbiased=False)
    if float(target_var.item()) <= 1e-8:
        return 0.0
    residual_var = torch.var(targets - predictions, unbiased=False)
    return float((1.0 - residual_var / target_var).item())


def save_metrics(path: str | Path, metrics: TrainingMetrics) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    return output_path
