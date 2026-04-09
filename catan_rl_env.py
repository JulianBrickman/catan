from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from catan import (
    DEVELOPMENT_CARD_COUNTS,
    RESOURCE_LIST,
    Terrain,
    Action,
    ActionError,
    GameState,
    PLAYER_COLOR_NAMES,
    build_game,
)


OBSERVATION_SCHEMA_VERSION = "v1"
ACTION_INDEX_MODE = "phase_local"
TENSOR_SCHEMA_VERSION = "v1"
ACTION_TYPE_ORDER = [
    "roll_dice",
    "end_turn",
    "build_settlement",
    "build_city",
    "build_road",
    "build_road_free",
    "move_robber",
    "discard_resources",
    "buy_development_card",
    "play_knight",
    "play_road_building",
    "play_year_of_plenty",
    "play_monopoly",
    "trade_maritime",
    "propose_trade",
    "accept_trade",
    "decline_trade",
    "finish_road_building",
]

PHASE_ORDER = [
    "setup_settlement",
    "setup_road",
    "turn_start",
    "robber_discard",
    "robber_move",
    "trade_response",
    "main",
    "road_building",
    "game_over",
]
PHASE_TO_INDEX = {phase: index for index, phase in enumerate(PHASE_ORDER)}
TERRAIN_ORDER = [terrain.value for terrain in Terrain]
STRUCTURE_ORDER = ["empty", "settlement", "city"]
RESOURCE_OR_NONE_ORDER = ["none", *RESOURCE_LIST]
HARBOR_RESOURCE_ORDER = ["generic", *RESOURCE_LIST]
RESOURCE_NAME_TO_INDEX = {resource: index for index, resource in enumerate(RESOURCE_LIST)}
DEFAULT_PLACEMENT_REWARDS = {
    1: 1.0,
    2: 0.5,
    3: 0.25,
    4: 0.0,
}


@dataclass
class RewardConfig:
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "placement_score": 1.0,
            "win": 0.0,
            "final_vp": 0.0,
            "vp_gain": 0.0,
            "moves_taken": 0.0,
            "turn_penalty": 0.0,
            "roads_built": 0.0,
            "settlements_built": 0.0,
            "cities_built": 0.0,
            "dev_cards_bought": 0.0,
            "dev_cards_played": 0.0,
            "cards_gained": 0.0,
            "cards_stolen": 0.0,
            "robber_steals": 0.0,
            "resources_discarded": 0.0,
        }
    )
    placement_rewards: Dict[int, float] = field(default_factory=lambda: dict(DEFAULT_PLACEMENT_REWARDS))

    def scalarize(self, components: Dict[str, float]) -> float:
        return sum(self.weights.get(name, 0.0) * value for name, value in components.items())


@dataclass
class CatanEnvConfig:
    num_players: int = 4
    seed: Optional[int] = None
    allow_domestic_trade: bool = False
    max_turns: Optional[int] = None
    reward_config: RewardConfig = field(default_factory=RewardConfig)


class CatanRLEnv:
    """Training-facing wrapper around the base Catan engine.

    This v1 wrapper intentionally uses phase-local action indexing rather than a single
    global fixed action catalog. That keeps the environment stable for training experiments
    while we postpone the hardest catalog-design problem: robber-discard combinatorics.
    """

    def __init__(self, config: Optional[CatanEnvConfig] = None) -> None:
        self.config = config or CatanEnvConfig()
        self.state: GameState = build_game(
            num_players=self.config.num_players,
            seed=self.config.seed,
        )
        self._last_legal_actions: List[Action] = []

    def reset(self, seed: Optional[int] = None) -> Dict[str, object]:
        effective_seed = self.config.seed if seed is None else seed
        self.state = build_game(num_players=self.config.num_players, seed=effective_seed)
        self._last_legal_actions = self.legal_actions()
        return self.observe()

    def observe(self, player_id: Optional[int] = None) -> Dict[str, object]:
        observer_id = self.state.active_player if player_id is None else player_id
        legal_actions = self.legal_actions_for_player(observer_id)
        return {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "tensor_schema_version": TENSOR_SCHEMA_VERSION,
            "action_index_mode": ACTION_INDEX_MODE,
            "observer_player_id": observer_id,
            "active_player_id": self.state.active_player,
            "global": self._global_features(),
            "hexes": self._hex_features(),
            "intersections": self._intersection_features(),
            "edges": self._edge_features(),
            "players": self._player_features(observer_id),
            "private": self._private_features(observer_id),
            "legal_actions": [self.serialize_action(action) for action in legal_actions],
            "legal_action_mask": [1] * len(legal_actions),
            "action_tensor_spec": self.action_tensor_spec(),
            "action_features": self.tensorize_legal_actions(legal_actions),
            "tensor_spec": self.tensor_spec(),
            "tensor_obs": self.tensorize_observation(observer_id),
        }

    def tensor_spec(self) -> Dict[str, object]:
        return {
            "tensor_schema_version": TENSOR_SCHEMA_VERSION,
            "global_features": {
                "size": len(self._global_feature_names()),
                "names": self._global_feature_names(),
            },
            "hex_features": {
                "shape": [len(self.state.board.hexes), len(self._hex_feature_names())],
                "names": self._hex_feature_names(),
            },
            "intersection_features": {
                "shape": [len(self.state.board.intersections), len(self._intersection_feature_names())],
                "names": self._intersection_feature_names(),
            },
            "edge_features": {
                "shape": [len(self.state.board.edges), len(self._edge_feature_names())],
                "names": self._edge_feature_names(),
            },
            "player_features": {
                "shape": [len(self.state.players), len(self._player_feature_names())],
                "names": self._player_feature_names(),
            },
            "private_features": {
                "size": len(self._private_feature_names()),
                "names": self._private_feature_names(),
            },
        }

    def tensorize_observation(self, player_id: Optional[int] = None) -> Dict[str, object]:
        observer_id = self.state.active_player if player_id is None else player_id
        return {
            "global_features": self._tensorize_global_features(observer_id),
            "hex_features": self._tensorize_hex_features(),
            "intersection_features": self._tensorize_intersection_features(observer_id),
            "edge_features": self._tensorize_edge_features(observer_id),
            "player_features": self._tensorize_player_features(observer_id),
            "private_features": self._tensorize_private_features(observer_id),
        }

    def action_tensor_spec(self) -> Dict[str, object]:
        return {
            "action_feature_size": len(self._action_feature_names()),
            "action_feature_names": self._action_feature_names(),
        }

    def tensorize_legal_actions(self, legal_actions: Optional[Sequence[Action]] = None) -> List[List[float]]:
        actions = self.legal_actions() if legal_actions is None else list(legal_actions)
        return [self._tensorize_action(action) for action in actions]

    def legal_actions(self) -> List[Action]:
        return self._filter_actions(self.state.legal_actions())

    def legal_actions_for_player(self, player_id: int) -> List[Action]:
        if player_id != self.state.active_player:
            return []
        return self.legal_actions()

    def step(self, action_index: int) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        legal_actions = self.legal_actions()
        if action_index < 0 or action_index >= len(legal_actions):
            raise ActionError(f"Action index {action_index} is out of range for {len(legal_actions)} legal actions.")

        acting_player = self.state.active_player
        before = self._snapshot()
        chosen = legal_actions[action_index]
        self.state.apply_action(chosen)
        after = self._snapshot()

        reward_components = self._reward_components(before, after, acting_player, chosen)
        truncated = self.config.max_turns is not None and self.state.turn_number >= self.config.max_turns
        done = self.state.pending_phase == "game_over" or truncated
        scalar_reward = self.config.reward_config.scalarize(reward_components)
        info: Dict[str, object] = {
            "acting_player": acting_player,
            "next_active_player": self.state.active_player,
            "action": self.serialize_action(chosen),
            "reward_components": reward_components,
            "winner": self.state.winner,
            "phase": self.state.pending_phase,
            "truncated": truncated,
        }

        if done:
            placements = self.compute_placements()
            terminal_components = self.terminal_reward_components(placements)
            info["placements"] = placements
            info["terminal_reward_components"] = terminal_components
            info["terminal_rewards_by_player"] = {
                player_id: self.config.reward_config.scalarize(components)
                for player_id, components in terminal_components.items()
            }
            scalar_reward += info["terminal_rewards_by_player"][acting_player]

        self._last_legal_actions = self.legal_actions()
        return self.observe(), scalar_reward, done, info

    def serialize_action(self, action: Action) -> Dict[str, object]:
        return {
            "action_type": action.action_type,
            "params": dict(action.params),
        }

    def compute_placements(self) -> Dict[int, int]:
        ranking = sorted(
            range(len(self.state.players)),
            key=lambda player_id: (
                self.state.total_victory_points(player_id),
                self.state.visible_victory_points(player_id),
                -player_id,
            ),
            reverse=True,
        )
        placements: Dict[int, int] = {}
        current_place = 1
        for index, player_id in enumerate(ranking):
            if index > 0:
                previous = ranking[index - 1]
                if not self._placement_tied(player_id, previous):
                    current_place = index + 1
            placements[player_id] = current_place
        return placements

    def terminal_reward_components(self, placements: Dict[int, int]) -> Dict[int, Dict[str, float]]:
        terminal: Dict[int, Dict[str, float]] = {}
        for player_id in range(len(self.state.players)):
            place = placements[player_id]
            terminal[player_id] = {
                "placement_score": self.config.reward_config.placement_rewards.get(place, 0.0),
                "win": 1.0 if self.state.winner == player_id else 0.0,
                "final_vp": float(self.state.total_victory_points(player_id)),
            }
        return terminal

    def action_space_size(self) -> int:
        return len(self.legal_actions())

    def _filter_actions(self, actions: Sequence[Action]) -> List[Action]:
        if self.config.allow_domestic_trade:
            return list(actions)
        filtered: List[Action] = []
        for action in actions:
            if action.action_type in {"propose_trade", "accept_trade", "decline_trade"}:
                continue
            filtered.append(action)
        return filtered

    def _global_features(self) -> Dict[str, object]:
        return {
            "turn_number": self.state.turn_number,
            "active_player_id": self.state.active_player,
            "phase": self.state.pending_phase,
            "phase_index": PHASE_TO_INDEX[self.state.pending_phase],
            "phase_one_hot": [1 if phase == self.state.pending_phase else 0 for phase in PHASE_ORDER],
            "rolled_this_turn": self.state.rolled_this_turn,
            "dev_card_played_this_turn": self.state.dev_card_played_this_turn,
            "last_roll": self.state.last_roll if self.state.last_roll is not None else 0,
            "robber_hex_id": self.state.robber_hex_id,
            "longest_road_owner": -1 if self.state.longest_road_owner is None else self.state.longest_road_owner,
            "largest_army_owner": -1 if self.state.largest_army_owner is None else self.state.largest_army_owner,
            "winner": -1 if self.state.winner is None else self.state.winner,
            "pending_trade_responder": -1
            if self.state.pending_trade_responder is None
            else self.state.pending_trade_responder,
        }

    def _global_feature_names(self) -> List[str]:
        return [
            "observer_player_id_norm",
            "active_player_id_norm",
            "turn_number_norm",
            "last_roll_norm",
            "robber_hex_id_norm",
            "rolled_this_turn",
            "dev_card_played_this_turn",
            "has_winner",
            "winner_id_norm",
            "longest_road_owner_norm",
            "largest_army_owner_norm",
            "pending_trade_responder_norm",
            *[f"phase_{phase}" for phase in PHASE_ORDER],
        ]

    def _hex_feature_names(self) -> List[str]:
        return [
            *[f"terrain_{terrain}" for terrain in TERRAIN_ORDER],
            *[f"resource_{resource_name}" for resource_name in RESOURCE_OR_NONE_ORDER],
            "token_norm",
            "token_pips_norm",
            "has_robber",
            "q_norm",
            "r_norm",
            "x_norm",
            "y_norm",
            "is_coastal_hex",
        ]

    def _intersection_feature_names(self) -> List[str]:
        return [
            "owner_none",
            *[f"owner_p{player_id}" for player_id in range(len(self.state.players))],
            *[f"piece_{piece}" for piece in STRUCTURE_ORDER],
            "coastal",
            "adjacent_hex_count_norm",
            "adjacent_edge_count_norm",
            "adjacent_intersection_count_norm",
            "is_observer_owned",
            "is_enemy_owned",
            "x_norm",
            "y_norm",
        ]

    def _edge_feature_names(self) -> List[str]:
        return [
            "owner_none",
            *[f"owner_p{player_id}" for player_id in range(len(self.state.players))],
            "coastal",
            "adjacent_hex_count_norm",
            "is_observer_owned",
            "is_enemy_owned",
        ]

    def _player_feature_names(self) -> List[str]:
        return [
            "is_observer",
            "turn_order_relative_norm",
            "visible_vp_norm",
            "total_resource_count_norm",
            "total_dev_card_count_norm",
            "roads_remaining_norm",
            "settlements_remaining_norm",
            "cities_remaining_norm",
            "played_knights_norm",
            "has_longest_road",
            "has_largest_army",
        ]

    def _private_feature_names(self) -> List[str]:
        return [
            *[f"resource_{resource_name}_count_norm" for resource_name in RESOURCE_LIST],
            *[f"dev_{card_name}_count_norm" for card_name in DEVELOPMENT_CARD_COUNTS],
            *[f"new_dev_{card_name}_count_norm" for card_name in DEVELOPMENT_CARD_COUNTS],
        ]

    def _action_feature_names(self) -> List[str]:
        return [
            *[f"action_{action_type}" for action_type in ACTION_TYPE_ORDER],
            "intersection_id_norm",
            "edge_id_norm",
            "hex_id_norm",
            "victim_id_norm",
            "player_id_norm",
            "rate_norm",
            *[f"resource_flag_{resource_name}" for resource_name in RESOURCE_LIST],
            *[f"give_resource_{resource_name}" for resource_name in RESOURCE_LIST],
            *[f"receive_resource_{resource_name}" for resource_name in RESOURCE_LIST],
            *[f"year_of_plenty_pick_{resource_name}" for resource_name in RESOURCE_LIST],
            *[f"discard_{resource_name}_norm" for resource_name in RESOURCE_LIST],
        ]

    def _hex_features(self) -> List[Dict[str, object]]:
        features: List[Dict[str, object]] = []
        for tile in self.state.board.hexes:
            features.append(
                {
                    "id": tile.id,
                    "terrain": tile.terrain.value,
                    "resource": None if tile.resource is None else tile.resource.value,
                    "token": 0 if tile.token is None else tile.token,
                    "has_robber": tile.id == self.state.robber_hex_id,
                    "neighbor_hex_ids": list(tile.neighbor_hex_ids),
                }
            )
        return features

    def _intersection_features(self) -> List[Dict[str, object]]:
        features: List[Dict[str, object]] = []
        for node in self.state.board.intersections:
            owner = self.state.city_owners.get(node.id, self.state.settlement_owners.get(node.id))
            piece = None
            if node.id in self.state.city_owners:
                piece = "city"
            elif node.id in self.state.settlement_owners:
                piece = "settlement"
            features.append(
                {
                    "id": node.id,
                    "owner": -1 if owner is None else owner,
                    "piece": piece,
                    "coastal": node.coastal,
                    "adjacent_hex_ids": list(node.adjacent_hex_ids),
                    "adjacent_edge_ids": list(node.adjacent_edge_ids),
                    "adjacent_intersection_ids": list(node.adjacent_intersection_ids),
                }
            )
        return features

    def _edge_features(self) -> List[Dict[str, object]]:
        features: List[Dict[str, object]] = []
        for edge in self.state.board.edges:
            owner = self.state.road_owners.get(edge.id)
            features.append(
                {
                    "id": edge.id,
                    "owner": -1 if owner is None else owner,
                    "intersection_ids": list(edge.intersection_ids),
                    "adjacent_hex_ids": list(edge.adjacent_hex_ids),
                    "coastal": edge.coastal,
                }
            )
        return features

    def _player_features(self, observer_id: int) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for player in self.state.players:
            rows.append(
                {
                    "player_id": player.player_id,
                    "color": PLAYER_COLOR_NAMES[player.player_id],
                    "is_observer": player.player_id == observer_id,
                    "visible_vp": self.state.visible_victory_points(player.player_id),
                    "total_resource_count": player.total_resources(),
                    "total_development_card_count": player.total_development_cards(),
                    "roads_remaining": player.roads_remaining,
                    "settlements_remaining": player.settlements_remaining,
                    "cities_remaining": player.cities_remaining,
                    "played_knights": player.played_knights,
                }
            )
        return rows

    def _private_features(self, observer_id: int) -> Dict[str, object]:
        player = self.state.players[observer_id]
        return {
            "resources": dict(player.resources),
            "development_cards": dict(player.development_cards),
            "new_development_cards": dict(player.new_development_cards),
        }

    def _tensorize_global_features(self, observer_id: int) -> List[float]:
        return [
            self._normalize_player_id(observer_id),
            self._normalize_player_id(self.state.active_player),
            self._normalize_turn_number(self.state.turn_number),
            self._normalize_roll(self.state.last_roll),
            self._normalize_hex_id(self.state.robber_hex_id),
            float(self.state.rolled_this_turn),
            float(self.state.dev_card_played_this_turn),
            float(self.state.winner is not None),
            self._normalize_player_id(self.state.winner),
            self._normalize_player_id(self.state.longest_road_owner),
            self._normalize_player_id(self.state.largest_army_owner),
            self._normalize_player_id(self.state.pending_trade_responder),
            *[1.0 if phase == self.state.pending_phase else 0.0 for phase in PHASE_ORDER],
        ]

    def _tensorize_hex_features(self) -> List[List[float]]:
        rows: List[List[float]] = []
        for tile in self.state.board.hexes:
            token = 0 if tile.token is None else tile.token
            rows.append(
                [
                    *self._one_hot(tile.terrain.value, TERRAIN_ORDER),
                    *self._one_hot(
                        "none" if tile.resource is None else tile.resource.value,
                        RESOURCE_OR_NONE_ORDER,
                    ),
                    token / 12.0,
                    self._token_pips(token),
                    float(tile.id == self.state.robber_hex_id),
                    self._normalize_coordinate(tile.q, 2.0),
                    self._normalize_coordinate(tile.r, 2.0),
                    self._normalize_coordinate(tile.x, 4.0),
                    self._normalize_coordinate(tile.y, 3.0),
                    float(len(tile.neighbor_hex_ids) < 6),
                ]
            )
        return rows

    def _tensorize_intersection_features(self, observer_id: int) -> List[List[float]]:
        rows: List[List[float]] = []
        for node in self.state.board.intersections:
            owner = self.state.city_owners.get(node.id, self.state.settlement_owners.get(node.id))
            piece = "empty"
            if node.id in self.state.city_owners:
                piece = "city"
            elif node.id in self.state.settlement_owners:
                piece = "settlement"
            rows.append(
                [
                    *self._owner_one_hot(owner),
                    *self._one_hot(piece, STRUCTURE_ORDER),
                    float(node.coastal),
                    len(node.adjacent_hex_ids) / 3.0,
                    len(node.adjacent_edge_ids) / 3.0,
                    len(node.adjacent_intersection_ids) / 3.0,
                    float(owner == observer_id),
                    float(owner is not None and owner != observer_id),
                    self._normalize_coordinate(node.x, 4.5),
                    self._normalize_coordinate(node.y, 3.5),
                ]
            )
        return rows

    def _tensorize_edge_features(self, observer_id: int) -> List[List[float]]:
        rows: List[List[float]] = []
        for edge in self.state.board.edges:
            owner = self.state.road_owners.get(edge.id)
            rows.append(
                [
                    *self._owner_one_hot(owner),
                    float(edge.coastal),
                    len(edge.adjacent_hex_ids) / 2.0,
                    float(owner == observer_id),
                    float(owner is not None and owner != observer_id),
                ]
            )
        return rows

    def _tensorize_player_features(self, observer_id: int) -> List[List[float]]:
        rows: List[List[float]] = []
        num_players = len(self.state.players)
        for player in self.state.players:
            relative_turn = (player.player_id - observer_id) % num_players
            rows.append(
                [
                    float(player.player_id == observer_id),
                    relative_turn / max(1, num_players - 1),
                    self.state.visible_victory_points(player.player_id) / 10.0,
                    player.total_resources() / 19.0,
                    player.total_development_cards() / 25.0,
                    player.roads_remaining / 15.0,
                    player.settlements_remaining / 5.0,
                    player.cities_remaining / 4.0,
                    player.played_knights / 14.0,
                    float(self.state.longest_road_owner == player.player_id),
                    float(self.state.largest_army_owner == player.player_id),
                ]
            )
        return rows

    def _tensorize_private_features(self, observer_id: int) -> List[float]:
        player = self.state.players[observer_id]
        return [
            *[player.resources[resource_name] / 19.0 for resource_name in RESOURCE_LIST],
            *[
                player.development_cards[card_name] / DEVELOPMENT_CARD_COUNTS[card_name]
                for card_name in DEVELOPMENT_CARD_COUNTS
            ],
            *[
                player.new_development_cards[card_name] / DEVELOPMENT_CARD_COUNTS[card_name]
                for card_name in DEVELOPMENT_CARD_COUNTS
            ],
        ]

    def _snapshot(self) -> Dict[str, object]:
        return {
            "winner": self.state.winner,
            "pending_phase": self.state.pending_phase,
            "total_vp": {
                player.player_id: self.state.total_victory_points(player.player_id)
                for player in self.state.players
            },
            "played_knights": {
                player.player_id: player.played_knights for player in self.state.players
            },
            "resources": {
                player.player_id: dict(player.resources) for player in self.state.players
            },
            "structures": {
                player.player_id: {
                    "roads_built": 15 - player.roads_remaining,
                    "settlements_built": 5 - player.settlements_remaining,
                    "cities_built": 4 - player.cities_remaining,
                    "dev_cards_total": player.total_development_cards(),
                }
                for player in self.state.players
            },
            "largest_army_owner": self.state.largest_army_owner,
            "longest_road_owner": self.state.longest_road_owner,
        }

    def _reward_components(
        self,
        before: Dict[str, object],
        after: Dict[str, object],
        acting_player: int,
        action: Action,
    ) -> Dict[str, float]:
        before_total_vp = before["total_vp"][acting_player]
        after_total_vp = after["total_vp"][acting_player]
        before_resources = before["resources"][acting_player]
        after_resources = after["resources"][acting_player]
        before_structures = before["structures"][acting_player]
        after_structures = after["structures"][acting_player]

        cards_gained = 0
        resources_discarded = 0
        for resource_name in RESOURCE_LIST:
            delta = after_resources[resource_name] - before_resources[resource_name]
            if delta > 0:
                cards_gained += delta
            elif delta < 0 and action.action_type == "discard_resources":
                resources_discarded += -delta

        cards_stolen = 1 if action.action_type in {"move_robber", "play_knight"} and cards_gained > 0 else 0
        robber_steals = cards_stolen
        dev_cards_before = before_structures["dev_cards_total"]
        dev_cards_after = after_structures["dev_cards_total"]
        return {
            "placement_score": 0.0,
            "win": 1.0 if self.state.winner == acting_player else 0.0,
            "final_vp": float(after_total_vp) if self.state.winner == acting_player else 0.0,
            "vp_gain": float(after_total_vp - before_total_vp),
            "moves_taken": 1.0,
            "turn_penalty": 1.0,
            "roads_built": float(after_structures["roads_built"] - before_structures["roads_built"]),
            "settlements_built": float(after_structures["settlements_built"] - before_structures["settlements_built"]),
            "cities_built": float(after_structures["cities_built"] - before_structures["cities_built"]),
            "dev_cards_bought": float(max(0, dev_cards_after - dev_cards_before)),
            "dev_cards_played": 1.0 if action.action_type.startswith("play_") else 0.0,
            "cards_gained": float(cards_gained),
            "cards_stolen": float(cards_stolen),
            "robber_steals": float(robber_steals),
            "resources_discarded": float(resources_discarded),
        }

    def _placement_tied(self, player_a: int, player_b: int) -> bool:
        return (
            self.state.total_victory_points(player_a) == self.state.total_victory_points(player_b)
            and self.state.visible_victory_points(player_a) == self.state.visible_victory_points(player_b)
        )

    def _tensorize_action(self, action: Action) -> List[float]:
        params = action.params
        resource_flag = [0.0] * len(RESOURCE_LIST)
        give_resource = [0.0] * len(RESOURCE_LIST)
        receive_resource = [0.0] * len(RESOURCE_LIST)
        year_of_plenty = [0.0] * len(RESOURCE_LIST)
        discard_counts = [0.0] * len(RESOURCE_LIST)

        monopoly_resource = params.get("resource")
        if isinstance(monopoly_resource, str) and monopoly_resource in RESOURCE_NAME_TO_INDEX:
            resource_flag[RESOURCE_NAME_TO_INDEX[monopoly_resource]] = 1.0

        give_name = params.get("give_resource")
        if isinstance(give_name, str) and give_name in RESOURCE_NAME_TO_INDEX:
            give_resource[RESOURCE_NAME_TO_INDEX[give_name]] = 1.0

        receive_name = params.get("receive_resource")
        if isinstance(receive_name, str) and receive_name in RESOURCE_NAME_TO_INDEX:
            receive_resource[RESOURCE_NAME_TO_INDEX[receive_name]] = 1.0

        resources = params.get("resources")
        if isinstance(resources, list):
            for resource_name in resources:
                if resource_name in RESOURCE_NAME_TO_INDEX:
                    year_of_plenty[RESOURCE_NAME_TO_INDEX[resource_name]] += 0.5
        elif isinstance(resources, dict):
            for resource_name, amount in resources.items():
                if resource_name in RESOURCE_NAME_TO_INDEX:
                    discard_counts[RESOURCE_NAME_TO_INDEX[resource_name]] = min(float(amount) / 4.0, 1.0)

        return [
            *self._one_hot(action.action_type, ACTION_TYPE_ORDER),
            self._normalize_intersection_id(params.get("intersection_id")),
            self._normalize_edge_id(params.get("edge_id")),
            self._normalize_hex_id(params.get("hex_id")),
            self._normalize_player_id(params.get("victim_id")),
            self._normalize_player_id(params.get("player_id")),
            float(params.get("rate", 0)) / 4.0,
            *resource_flag,
            *give_resource,
            *receive_resource,
            *year_of_plenty,
            *discard_counts,
        ]

    def _one_hot(self, value: str, categories: Sequence[str]) -> List[float]:
        return [1.0 if category == value else 0.0 for category in categories]

    def _owner_one_hot(self, owner: Optional[int]) -> List[float]:
        return [float(owner is None), *[1.0 if owner == player_id else 0.0 for player_id in range(len(self.state.players))]]

    def _normalize_player_id(self, player_id: Optional[int]) -> float:
        if player_id is None:
            return -1.0
        return player_id / max(1, len(self.state.players) - 1)

    def _normalize_hex_id(self, hex_id: Optional[int]) -> float:
        if hex_id is None:
            return -1.0
        return hex_id / max(1, len(self.state.board.hexes) - 1)

    def _normalize_intersection_id(self, intersection_id: Optional[int]) -> float:
        if intersection_id is None:
            return -1.0
        return intersection_id / max(1, len(self.state.board.intersections) - 1)

    def _normalize_edge_id(self, edge_id: Optional[int]) -> float:
        if edge_id is None:
            return -1.0
        return edge_id / max(1, len(self.state.board.edges) - 1)

    def _normalize_turn_number(self, turn_number: int) -> float:
        return min(turn_number / 200.0, 1.0)

    def _normalize_roll(self, roll: Optional[int]) -> float:
        if roll is None:
            return 0.0
        return roll / 12.0

    def _normalize_coordinate(self, value: float, scale: float) -> float:
        return (value / scale + 1.0) / 2.0

    def _token_pips(self, token: int) -> float:
        pip_map = {
            0: 0.0,
            2: 1.0,
            3: 2.0,
            4: 3.0,
            5: 4.0,
            6: 5.0,
            8: 5.0,
            9: 4.0,
            10: 3.0,
            11: 2.0,
            12: 1.0,
        }
        return pip_map[token] / 5.0
