from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from catan_rl_env import PHASE_ORDER


@dataclass(frozen=True)
class SharedEncoderConfig:
    global_hidden_dim: int = 64
    private_hidden_dim: int = 64
    entity_hidden_dim: int = 64
    entity_output_dim: int = 64
    trunk_hidden_dim: int = 256
    latent_dim: int = 256
    dropout: float = 0.0


@dataclass(frozen=True)
class PolicyHeadConfig:
    action_hidden_dim: int = 128
    value_hidden_dim: int = 128
    dropout: float = 0.0


def tensor_obs_to_torch(
    tensor_obs: Mapping[str, Sequence],
    *,
    device: torch.device | str | None = None,
) -> Dict[str, Tensor]:
    result: Dict[str, Tensor] = {}
    for key, value in tensor_obs.items():
        tensor = torch.tensor(value, dtype=torch.float32, device=device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        result[key] = tensor
    return result


def action_features_to_torch(
    action_features: Sequence[Sequence[float]],
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    tensor = torch.tensor(action_features, dtype=torch.float32, device=device)
    if tensor.ndim != 2:
        raise ValueError(f"Expected action feature tensor of shape [actions, features], got {tuple(tensor.shape)}")
    return tensor.unsqueeze(0)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers: List[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EntityPoolEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.entity_mlp = MLP(input_dim, [hidden_dim], hidden_dim, dropout=dropout)
        self.proj = MLP(hidden_dim * 2, [hidden_dim], output_dim, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected entity tensor of shape [batch, items, features], got {tuple(x.shape)}")
        encoded = self.entity_mlp(x)
        mean_pool = encoded.mean(dim=1)
        max_pool, _ = encoded.max(dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return self.proj(pooled)


class SharedCatanEncoder(nn.Module):
    def __init__(self, tensor_spec: Mapping[str, object], config: SharedEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or SharedEncoderConfig()
        self.tensor_spec = tensor_spec

        global_dim = int(tensor_spec["global_features"]["size"])
        private_dim = int(tensor_spec["private_features"]["size"])
        hex_dim = int(tensor_spec["hex_features"]["shape"][1])
        intersection_dim = int(tensor_spec["intersection_features"]["shape"][1])
        edge_dim = int(tensor_spec["edge_features"]["shape"][1])
        player_dim = int(tensor_spec["player_features"]["shape"][1])

        self.global_encoder = MLP(
            global_dim,
            [self.config.global_hidden_dim],
            self.config.global_hidden_dim,
            dropout=self.config.dropout,
        )
        self.private_encoder = MLP(
            private_dim,
            [self.config.private_hidden_dim],
            self.config.private_hidden_dim,
            dropout=self.config.dropout,
        )
        self.hex_encoder = EntityPoolEncoder(
            hex_dim,
            self.config.entity_hidden_dim,
            self.config.entity_output_dim,
            dropout=self.config.dropout,
        )
        self.intersection_encoder = EntityPoolEncoder(
            intersection_dim,
            self.config.entity_hidden_dim,
            self.config.entity_output_dim,
            dropout=self.config.dropout,
        )
        self.edge_encoder = EntityPoolEncoder(
            edge_dim,
            self.config.entity_hidden_dim,
            self.config.entity_output_dim,
            dropout=self.config.dropout,
        )
        self.player_encoder = EntityPoolEncoder(
            player_dim,
            self.config.entity_hidden_dim,
            self.config.entity_output_dim,
            dropout=self.config.dropout,
        )

        trunk_input_dim = (
            self.config.global_hidden_dim
            + self.config.private_hidden_dim
            + self.config.entity_output_dim * 4
        )
        self.trunk = MLP(
            trunk_input_dim,
            [self.config.trunk_hidden_dim],
            self.config.latent_dim,
            dropout=self.config.dropout,
        )

    def forward(self, tensor_obs: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        required_keys = {
            "global_features",
            "hex_features",
            "intersection_features",
            "edge_features",
            "player_features",
            "private_features",
        }
        missing = required_keys.difference(tensor_obs.keys())
        if missing:
            raise KeyError(f"Missing tensor observation keys: {sorted(missing)}")

        global_embedding = self.global_encoder(self._ensure_2d(tensor_obs["global_features"], "global_features"))
        private_embedding = self.private_encoder(self._ensure_2d(tensor_obs["private_features"], "private_features"))
        hex_embedding = self.hex_encoder(tensor_obs["hex_features"])
        intersection_embedding = self.intersection_encoder(tensor_obs["intersection_features"])
        edge_embedding = self.edge_encoder(tensor_obs["edge_features"])
        player_embedding = self.player_encoder(tensor_obs["player_features"])

        trunk_input = torch.cat(
            [
                global_embedding,
                private_embedding,
                hex_embedding,
                intersection_embedding,
                edge_embedding,
                player_embedding,
            ],
            dim=-1,
        )
        latent = self.trunk(trunk_input)
        return {
            "latent": latent,
            "global_embedding": global_embedding,
            "private_embedding": private_embedding,
            "hex_embedding": hex_embedding,
            "intersection_embedding": intersection_embedding,
            "edge_embedding": edge_embedding,
            "player_embedding": player_embedding,
        }

    def _ensure_2d(self, x: Tensor, name: str) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected {name} to have shape [batch, features], got {tuple(x.shape)}")
        return x


class PhaseActionHead(nn.Module):
    def __init__(self, latent_dim: int, action_feature_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.scorer = MLP(
            latent_dim + action_feature_dim,
            [hidden_dim],
            1,
            dropout=dropout,
        )

    def forward(self, latent: Tensor, action_features: Tensor) -> Tensor:
        if latent.ndim != 2:
            raise ValueError(f"Expected latent to have shape [batch, latent_dim], got {tuple(latent.shape)}")
        if action_features.ndim != 3:
            raise ValueError(
                f"Expected action_features to have shape [batch, actions, features], got {tuple(action_features.shape)}"
            )
        batch_size, num_actions, _ = action_features.shape
        expanded_latent = latent.unsqueeze(1).expand(batch_size, num_actions, latent.shape[-1])
        scores = self.scorer(torch.cat([expanded_latent, action_features], dim=-1)).squeeze(-1)
        return scores


class CatanPolicyValueNet(nn.Module):
    def __init__(
        self,
        tensor_spec: Mapping[str, object],
        action_feature_dim: int,
        encoder_config: SharedEncoderConfig | None = None,
        head_config: PolicyHeadConfig | None = None,
    ) -> None:
        super().__init__()
        self.encoder = SharedCatanEncoder(tensor_spec, encoder_config)
        self.encoder_config = self.encoder.config
        self.head_config = head_config or PolicyHeadConfig()
        self.action_feature_dim = action_feature_dim
        self.phase_heads = nn.ModuleDict(
            {
                phase: PhaseActionHead(
                    self.encoder_config.latent_dim,
                    action_feature_dim,
                    self.head_config.action_hidden_dim,
                    dropout=self.head_config.dropout,
                )
                for phase in PHASE_ORDER
            }
        )
        self.value_head = MLP(
            self.encoder_config.latent_dim,
            [self.head_config.value_hidden_dim],
            1,
            dropout=self.head_config.dropout,
        )

    def forward(
        self,
        tensor_obs: Mapping[str, Tensor],
        action_features: Tensor,
        phase: str,
        action_mask: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        if phase not in self.phase_heads:
            raise KeyError(f"Unknown phase head: {phase}")
        encoder_outputs = self.encoder(tensor_obs)
        latent = encoder_outputs["latent"]
        logits = self.phase_heads[phase](latent, action_features)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask <= 0, -1e9)
        value = self.value_head(latent).squeeze(-1)
        return {
            **encoder_outputs,
            "policy_logits": logits,
            "value": value,
        }

    def sample_action(
        self,
        tensor_obs: Mapping[str, Tensor],
        action_features: Tensor,
        phase: str,
        action_mask: Tensor | None = None,
        deterministic: bool = False,
    ) -> Dict[str, Tensor]:
        outputs = self.forward(tensor_obs, action_features, phase, action_mask=action_mask)
        logits = outputs["policy_logits"]
        if logits.shape[-1] == 0:
            raise ValueError("Cannot sample from an empty legal action set.")
        if deterministic:
            action_index = torch.argmax(logits, dim=-1)
            log_prob = torch.zeros_like(action_index, dtype=torch.float32)
            entropy = torch.zeros_like(action_index, dtype=torch.float32)
        else:
            dist = Categorical(logits=logits)
            action_index = dist.sample()
            log_prob = dist.log_prob(action_index)
            entropy = dist.entropy()
        return {
            **outputs,
            "action_index": action_index,
            "log_prob": log_prob,
            "entropy": entropy,
        }
