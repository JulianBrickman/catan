from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


BOARD_RADIUS = 2
HEX_VERTEX_ANGLES = (-30, 30, 90, 150, 210, 270)
PLAYER_COLORS = ("#d94841", "#2f6fed", "#d99a00", "#2d8a4b")
PLAYER_COLOR_NAMES = ("Red", "Blue", "Gold", "Green")
TERRAIN_COLORS = {
    "desert": "#d9c27d",
    "hills": "#cc704b",
    "forest": "#4f8a4f",
    "mountains": "#8a8f98",
    "fields": "#d9bf4b",
    "pasture": "#84c26d",
}
RESOURCE_COLORS = {
    "brick": "#b84c34",
    "lumber": "#356f35",
    "ore": "#626873",
    "grain": "#c09b1e",
    "wool": "#5a9f51",
    "generic": "#2f4858",
}
RESOURCE_LIST = ("brick", "lumber", "ore", "grain", "wool")
DEVELOPMENT_CARD_COUNTS = {
    "knight": 14,
    "road_building": 2,
    "year_of_plenty": 2,
    "monopoly": 2,
    "victory_point": 5,
}
BUILD_COSTS = {
    "road": {"brick": 1, "lumber": 1},
    "settlement": {"brick": 1, "lumber": 1, "wool": 1, "grain": 1},
    "city": {"ore": 3, "grain": 2},
    "development_card": {"ore": 1, "grain": 1, "wool": 1},
}
SETUP_PHASES = ("setup_settlement", "setup_road")
ACTION_PHASES = (
    "turn_start",
    "robber_discard",
    "robber_move",
    "main",
    "road_building",
    "game_over",
)
HARBOR_EDGE_BY_INTERSECTION_LIMIT = 2


class Terrain(str, Enum):
    DESERT = "desert"
    HILLS = "hills"
    FOREST = "forest"
    MOUNTAINS = "mountains"
    FIELDS = "fields"
    PASTURE = "pasture"


class Resource(str, Enum):
    BRICK = "brick"
    LUMBER = "lumber"
    ORE = "ore"
    GRAIN = "grain"
    WOOL = "wool"


TERRAIN_TO_RESOURCE: Dict[Terrain, Optional[Resource]] = {
    Terrain.DESERT: None,
    Terrain.HILLS: Resource.BRICK,
    Terrain.FOREST: Resource.LUMBER,
    Terrain.MOUNTAINS: Resource.ORE,
    Terrain.FIELDS: Resource.GRAIN,
    Terrain.PASTURE: Resource.WOOL,
}


class ActionError(ValueError):
    pass


def player_label(player_id: int) -> str:
    return f"P{player_id} ({PLAYER_COLOR_NAMES[player_id]})"


@dataclass(frozen=True)
class Action:
    action_type: str
    params: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class HexTile:
    id: int
    q: int
    r: int
    x: float
    y: float
    terrain: Terrain
    resource: Optional[Resource]
    token: Optional[int]
    corner_ids: Tuple[int, ...]
    edge_ids: Tuple[int, ...]
    neighbor_hex_ids: Tuple[int, ...]


@dataclass(frozen=True)
class Intersection:
    id: int
    x: float
    y: float
    adjacent_hex_ids: Tuple[int, ...]
    adjacent_edge_ids: Tuple[int, ...]
    adjacent_intersection_ids: Tuple[int, ...]
    coastal: bool


@dataclass(frozen=True)
class Edge:
    id: int
    intersection_ids: Tuple[int, int]
    adjacent_hex_ids: Tuple[int, ...]
    coastal: bool


@dataclass(frozen=True)
class Harbor:
    id: int
    edge_id: int
    rate: int
    resource: Optional[Resource]


@dataclass(frozen=True)
class BoardTopology:
    hexes: Tuple[HexTile, ...]
    intersections: Tuple[Intersection, ...]
    edges: Tuple[Edge, ...]
    harbors: Tuple[Harbor, ...]


@dataclass
class PlayerState:
    player_id: int
    resources: Dict[str, int] = field(
        default_factory=lambda: {resource.value: 0 for resource in Resource}
    )
    development_cards: Dict[str, int] = field(
        default_factory=lambda: {key: 0 for key in DEVELOPMENT_CARD_COUNTS}
    )
    new_development_cards: Dict[str, int] = field(
        default_factory=lambda: {key: 0 for key in DEVELOPMENT_CARD_COUNTS}
    )
    roads_remaining: int = 15
    settlements_remaining: int = 5
    cities_remaining: int = 4
    played_knights: int = 0
    visible_victory_points: int = 0

    def total_resources(self) -> int:
        return sum(self.resources.values())

    def total_development_cards(self) -> int:
        return sum(self.development_cards.values()) + sum(self.new_development_cards.values())


@dataclass
class GameState:
    board: BoardTopology
    players: List[PlayerState]
    road_owners: Dict[int, int] = field(default_factory=dict)
    settlement_owners: Dict[int, int] = field(default_factory=dict)
    city_owners: Dict[int, int] = field(default_factory=dict)
    robber_hex_id: int = 0
    bank_resources: Dict[str, int] = field(
        default_factory=lambda: {resource.value: 19 for resource in Resource}
    )
    development_deck: List[str] = field(default_factory=list)
    active_player: int = 0
    starting_player: int = 0
    setup_round: int = 1
    setup_order_index: int = 0
    dev_card_played_this_turn: bool = False
    pending_phase: str = "setup_settlement"
    longest_road_owner: Optional[int] = None
    largest_army_owner: Optional[int] = None
    turn_number: int = 0
    last_roll: Optional[int] = None
    winner: Optional[int] = None
    pending_discard_players: List[int] = field(default_factory=list)
    pending_road_building: int = 0
    current_turn_road_builds: int = 0
    last_settlement_placed: Optional[int] = None
    event_log: List[str] = field(default_factory=list)

    def to_observation(self, player_id: int) -> Dict[str, object]:
        player = self.players[player_id]
        public_players = []
        for other in self.players:
            public_players.append(
                {
                    "player_id": other.player_id,
                    "resource_count": other.total_resources(),
                    "development_card_count": other.total_development_cards(),
                    "roads_remaining": other.roads_remaining,
                    "settlements_remaining": other.settlements_remaining,
                    "cities_remaining": other.cities_remaining,
                    "played_knights": other.played_knights,
                    "visible_victory_points": other.visible_victory_points,
                }
            )

        return {
            "active_player": self.active_player,
            "pending_phase": self.pending_phase,
            "robber_hex_id": self.robber_hex_id,
            "longest_road_owner": self.longest_road_owner,
            "largest_army_owner": self.largest_army_owner,
            "turn_number": self.turn_number,
            "last_roll": self.last_roll,
            "winner": self.winner,
            "hexes": [
                {
                    "id": tile.id,
                    "terrain": tile.terrain.value,
                    "resource": tile.resource.value if tile.resource else None,
                    "token": tile.token,
                    "has_robber": tile.id == self.robber_hex_id,
                }
                for tile in self.board.hexes
            ],
            "intersections": [
                {
                    "id": node.id,
                    "adjacent_hex_ids": list(node.adjacent_hex_ids),
                    "owner": self.city_owners.get(node.id, self.settlement_owners.get(node.id)),
                    "piece": "city"
                    if node.id in self.city_owners
                    else "settlement"
                    if node.id in self.settlement_owners
                    else None,
                    "coastal": node.coastal,
                }
                for node in self.board.intersections
            ],
            "edges": [
                {
                    "id": edge.id,
                    "intersection_ids": list(edge.intersection_ids),
                    "owner": self.road_owners.get(edge.id),
                    "coastal": edge.coastal,
                }
                for edge in self.board.edges
            ],
            "harbors": [
                {
                    "edge_id": harbor.edge_id,
                    "rate": harbor.rate,
                    "resource": harbor.resource.value if harbor.resource else None,
                }
                for harbor in self.board.harbors
            ],
            "player_private": {
                "player_id": player_id,
                "resources": dict(player.resources),
                "development_cards": dict(player.development_cards),
                "new_development_cards": dict(player.new_development_cards),
            },
            "players": public_players,
            "legal_actions": [
                {"action_type": action.action_type, "params": action.params}
                for action in self.legal_actions()
            ],
        }

    def write_svg(self, output_path: str | Path) -> Path:
        renderer = SvgRenderer(self.board, self)
        return renderer.write(output_path)

    def legal_actions(self) -> List[Action]:
        if self.pending_phase == "game_over":
            return []
        if self.pending_phase == "setup_settlement":
            return [Action("build_settlement", {"intersection_id": node_id}) for node_id in self.legal_initial_settlement_ids()]
        if self.pending_phase == "setup_road":
            return [Action("build_road", {"edge_id": edge_id}) for edge_id in self.legal_setup_road_ids()]
        if self.pending_phase == "turn_start":
            return [Action("roll_dice", {})]
        if self.pending_phase == "robber_discard":
            player_id = self.pending_discard_players[0]
            return [
                Action("discard_resources", {"player_id": player_id, "resources": discard})
                for discard in enumerate_discards(self.players[player_id].resources, self.required_discard_count(player_id))
            ]
        if self.pending_phase == "robber_move":
            actions = []
            for hex_id in self.legal_robber_hex_ids():
                victims = self.legal_robber_victims(hex_id)
                if victims:
                    for victim in victims:
                        actions.append(Action("move_robber", {"hex_id": hex_id, "victim_id": victim}))
                else:
                    actions.append(Action("move_robber", {"hex_id": hex_id, "victim_id": None}))
            return actions
        if self.pending_phase == "road_building":
            road_ids = self.legal_road_ids(self.active_player, setup=False)
            actions = [Action("build_road_free", {"edge_id": edge_id}) for edge_id in road_ids]
            actions.append(Action("finish_road_building", {}))
            return actions
        if self.pending_phase == "main":
            return self.legal_main_phase_actions()
        raise ActionError(f"Unknown phase: {self.pending_phase}")

    def legal_main_phase_actions(self) -> List[Action]:
        actions = [Action("end_turn", {})]
        if self.can_afford(self.active_player, BUILD_COSTS["road"]):
            actions.extend(
                Action("build_road", {"edge_id": edge_id})
                for edge_id in self.legal_road_ids(self.active_player, setup=False)
            )
        actions.extend(Action("build_settlement", {"intersection_id": node_id}) for node_id in self.legal_settlement_ids(self.active_player))
        actions.extend(Action("build_city", {"intersection_id": node_id}) for node_id in self.legal_city_ids(self.active_player))
        actions.extend(Action("trade_domestic", trade) for trade in self.legal_domestic_trades(self.active_player))
        actions.extend(Action("trade_maritime", trade) for trade in self.legal_maritime_trades(self.active_player))
        if self.development_deck and self.can_afford(self.active_player, BUILD_COSTS["development_card"]):
            actions.append(Action("buy_development_card", {}))
        if not self.dev_card_played_this_turn:
            actions.extend(self.legal_development_card_actions())
        return actions

    def legal_development_card_actions(self) -> List[Action]:
        actions: List[Action] = []
        player = self.players[self.active_player]
        if player.development_cards["knight"] > 0:
            for hex_id in self.legal_robber_hex_ids():
                victims = self.legal_robber_victims(hex_id)
                if victims:
                    for victim in victims:
                        actions.append(Action("play_knight", {"hex_id": hex_id, "victim_id": victim}))
                else:
                    actions.append(Action("play_knight", {"hex_id": hex_id, "victim_id": None}))
        if player.development_cards["road_building"] > 0:
            actions.append(Action("play_road_building", {}))
        if player.development_cards["year_of_plenty"] > 0:
            for pair in year_of_plenty_choices(self.bank_resources):
                actions.append(Action("play_year_of_plenty", {"resources": pair}))
        if player.development_cards["monopoly"] > 0:
            for resource_name in RESOURCE_LIST:
                actions.append(Action("play_monopoly", {"resource": resource_name}))
        return actions

    def apply_action(self, action: Action) -> None:
        if self.winner is not None:
            raise ActionError("The game is already over.")

        action_type = action.action_type
        params = action.params

        if self.pending_phase == "setup_settlement":
            if action_type != "build_settlement":
                raise ActionError("Only settlement placement is allowed during setup.")
            self._apply_setup_settlement(int(params["intersection_id"]))
            return

        if self.pending_phase == "setup_road":
            if action_type != "build_road":
                raise ActionError("Only road placement is allowed during setup road phase.")
            self._apply_setup_road(int(params["edge_id"]))
            return

        if self.pending_phase == "turn_start":
            if action_type != "roll_dice":
                raise ActionError("You must roll dice at the start of the turn.")
            self.roll_dice()
            return

        if self.pending_phase == "robber_discard":
            if action_type != "discard_resources":
                raise ActionError("Players must discard before the robber moves.")
            self._apply_discard(int(params["player_id"]), params["resources"])
            return

        if self.pending_phase == "robber_move":
            if action_type not in {"move_robber", "play_knight"}:
                raise ActionError("A robber move is required.")
            self._apply_robber_move(int(params["hex_id"]), params.get("victim_id"), action_type == "play_knight")
            return

        if self.pending_phase == "road_building":
            if action_type == "finish_road_building":
                self.pending_road_building = 0
                self.pending_phase = "main"
                return
            if action_type != "build_road_free":
                raise ActionError("Only free roads are allowed during Road Building.")
            self._apply_build_road(int(params["edge_id"]), free=True)
            self.pending_road_building -= 1
            if self.pending_road_building <= 0 or not self.legal_road_ids(self.active_player, setup=False):
                self.pending_road_building = 0
                self.pending_phase = "main"
            return

        if self.pending_phase != "main":
            raise ActionError(f"Unsupported phase: {self.pending_phase}")

        if action_type == "end_turn":
            self.end_turn()
        elif action_type == "build_road":
            self._apply_build_road(int(params["edge_id"]), free=False)
        elif action_type == "build_settlement":
            self._apply_build_settlement(int(params["intersection_id"]))
        elif action_type == "build_city":
            self._apply_build_city(int(params["intersection_id"]))
        elif action_type == "trade_domestic":
            self._apply_domestic_trade(
                int(params["partner_id"]),
                str(params["give_resource"]),
                str(params["receive_resource"]),
            )
        elif action_type == "trade_maritime":
            self._apply_maritime_trade(str(params["give_resource"]), str(params["receive_resource"]), int(params["rate"]))
        elif action_type == "buy_development_card":
            self._apply_buy_development_card()
        elif action_type == "play_knight":
            self._consume_development_card("knight")
            self.players[self.active_player].played_knights += 1
            self.dev_card_played_this_turn = True
            self.update_largest_army()
            self._apply_robber_move(int(params["hex_id"]), params.get("victim_id"), played_as_knight=True)
        elif action_type == "play_road_building":
            self._consume_development_card("road_building")
            self.dev_card_played_this_turn = True
            self.pending_road_building = 2
            self.pending_phase = "road_building"
        elif action_type == "play_year_of_plenty":
            self._consume_development_card("year_of_plenty")
            self.dev_card_played_this_turn = True
            resources = params["resources"]
            if not isinstance(resources, list) or len(resources) != 2:
                raise ActionError("Year of Plenty requires two resources.")
            for resource_name in resources:
                self.take_bank_resource(self.active_player, str(resource_name), 1)
            self.check_for_winner()
        elif action_type == "play_monopoly":
            self._consume_development_card("monopoly")
            self.dev_card_played_this_turn = True
            self._apply_monopoly(str(params["resource"]))
            self.check_for_winner()
        else:
            raise ActionError(f"Unknown action type: {action_type}")

    def _apply_setup_settlement(self, intersection_id: int) -> None:
        if intersection_id not in self.legal_initial_settlement_ids():
            raise ActionError(f"Intersection {intersection_id} is not a legal setup settlement.")
        self.place_settlement(self.active_player, intersection_id, is_setup=True)
        self.last_settlement_placed = intersection_id
        self.pending_phase = "setup_road"

    def _apply_setup_road(self, edge_id: int) -> None:
        if edge_id not in self.legal_setup_road_ids():
            raise ActionError(f"Edge {edge_id} is not a legal setup road.")
        self.place_road(self.active_player, edge_id)
        if self.setup_round == 2:
            self.grant_second_settlement_resources(self.active_player, self.last_settlement_placed)
        self.advance_setup_turn()

    def _apply_discard(self, player_id: int, resources: object) -> None:
        if not self.pending_discard_players or self.pending_discard_players[0] != player_id:
            raise ActionError("This player is not currently required to discard.")
        if not isinstance(resources, dict):
            raise ActionError("Discard resources must be a resource-count mapping.")
        discard = normalize_resource_dict(resources)
        required = self.required_discard_count(player_id)
        if sum(discard.values()) != required:
            raise ActionError(f"Player {player_id} must discard exactly {required} cards.")
        for resource_name, amount in discard.items():
            if self.players[player_id].resources[resource_name] < amount:
                raise ActionError(f"Player {player_id} cannot discard {amount} {resource_name}.")
        for resource_name, amount in discard.items():
            self.players[player_id].resources[resource_name] -= amount
            self.bank_resources[resource_name] += amount
        self.pending_discard_players.pop(0)
        if not self.pending_discard_players:
            self.pending_phase = "robber_move"

    def _apply_robber_move(self, hex_id: int, victim_id: object, played_as_knight: bool) -> None:
        if hex_id not in self.legal_robber_hex_ids():
            raise ActionError(f"Hex {hex_id} is not a legal robber destination.")
        victim = None if victim_id is None else int(victim_id)
        victims = self.legal_robber_victims(hex_id)
        if victim is not None and victim not in victims:
            raise ActionError(f"Player {victim} is not a legal robber victim at hex {hex_id}.")
        if victim is None and victims:
            raise ActionError("A victim must be selected when robbable players exist.")
        self.robber_hex_id = hex_id
        if victim is not None:
            self.steal_random_resource(self.active_player, victim)
        if played_as_knight:
            self.log(f"{player_label(self.active_player)} played a knight on hex H{hex_id}.")
        else:
            self.log(f"{player_label(self.active_player)} moved the robber to hex H{hex_id}.")
        self.pending_phase = "main"
        self.check_for_winner()

    def _apply_build_road(self, edge_id: int, free: bool) -> None:
        if edge_id not in self.legal_road_ids(self.active_player, setup=False):
            raise ActionError(f"Edge {edge_id} is not a legal road placement.")
        if not free:
            self.pay_cost(self.active_player, BUILD_COSTS["road"])
        self.place_road(self.active_player, edge_id)
        self.current_turn_road_builds += 1
        self.update_longest_road()
        self.check_for_winner()

    def _apply_build_settlement(self, intersection_id: int) -> None:
        if intersection_id not in self.legal_settlement_ids(self.active_player):
            raise ActionError(f"Intersection {intersection_id} is not a legal settlement location.")
        self.pay_cost(self.active_player, BUILD_COSTS["settlement"])
        self.place_settlement(self.active_player, intersection_id, is_setup=False)
        self.update_longest_road()
        self.check_for_winner()

    def _apply_build_city(self, intersection_id: int) -> None:
        if intersection_id not in self.legal_city_ids(self.active_player):
            raise ActionError(f"Intersection {intersection_id} is not a legal city upgrade.")
        self.pay_cost(self.active_player, BUILD_COSTS["city"])
        self.upgrade_city(self.active_player, intersection_id)
        self.check_for_winner()

    def _apply_maritime_trade(self, give_resource: str, receive_resource: str, rate: int) -> None:
        options = self.legal_maritime_trades(self.active_player)
        trade = {
            "give_resource": give_resource,
            "receive_resource": receive_resource,
            "rate": rate,
        }
        if trade not in options:
            raise ActionError("That maritime trade is not currently legal.")
        self.players[self.active_player].resources[give_resource] -= rate
        self.bank_resources[give_resource] += rate
        self.take_bank_resource(self.active_player, receive_resource, 1)
        self.log(
            f"{player_label(self.active_player)} traded {rate} {give_resource} for 1 {receive_resource} with the bank."
        )

    def _apply_domestic_trade(
        self,
        partner_id: int,
        give_resource: str,
        receive_resource: str,
    ) -> None:
        trade = {
            "partner_id": partner_id,
            "give_resource": give_resource,
            "receive_resource": receive_resource,
        }
        if trade not in self.legal_domestic_trades(self.active_player):
            raise ActionError("That domestic trade is not currently legal.")
        actor = self.players[self.active_player]
        partner = self.players[partner_id]
        actor.resources[give_resource] -= 1
        partner.resources[give_resource] += 1
        partner.resources[receive_resource] -= 1
        actor.resources[receive_resource] += 1
        self.log(
            f"{player_label(self.active_player)} traded 1 {give_resource} for 1 {receive_resource} "
            f"with {player_label(partner_id)}."
        )

    def _apply_buy_development_card(self) -> None:
        if not self.development_deck:
            raise ActionError("The development deck is empty.")
        self.pay_cost(self.active_player, BUILD_COSTS["development_card"])
        card = self.development_deck.pop()
        self.players[self.active_player].new_development_cards[card] += 1
        self.log(f"{player_label(self.active_player)} bought a development card.")
        if card == "victory_point":
            self.check_for_winner()

    def _apply_monopoly(self, resource_name: str) -> None:
        total_taken = 0
        for player in self.players:
            if player.player_id == self.active_player:
                continue
            amount = player.resources[resource_name]
            if amount > 0:
                player.resources[resource_name] = 0
                self.players[self.active_player].resources[resource_name] += amount
                total_taken += amount
        self.log(
            f"{player_label(self.active_player)} monopolized {resource_name} for {total_taken} cards."
        )

    def roll_dice(self, forced_roll: Optional[int] = None) -> int:
        if self.pending_phase != "turn_start":
            raise ActionError("Dice can only be rolled at the start of a turn.")
        roll = forced_roll if forced_roll is not None else random.randint(1, 6) + random.randint(1, 6)
        if roll < 2 or roll > 12:
            raise ActionError("Dice roll must be between 2 and 12.")
        self.last_roll = roll
        self.log(f"{player_label(self.active_player)} rolled {roll}.")
        if roll == 7:
            self.pending_discard_players = [
                player.player_id for player in self.players if player.total_resources() > 7
            ]
            self.pending_phase = "robber_discard" if self.pending_discard_players else "robber_move"
        else:
            self.distribute_resources_for_roll(roll)
            self.pending_phase = "main"
        self.check_for_winner()
        return roll

    def end_turn(self) -> None:
        if self.pending_phase != "main":
            raise ActionError("A turn can only end during the main phase.")
        self.reveal_new_development_cards(self.active_player)
        self.active_player = (self.active_player + 1) % len(self.players)
        self.dev_card_played_this_turn = False
        self.current_turn_road_builds = 0
        self.last_roll = None
        self.turn_number += 1
        self.pending_phase = "turn_start"
        self.log(f"Turn advanced to {player_label(self.active_player)}.")
        self.check_for_winner(start_of_turn=True)

    def reveal_new_development_cards(self, player_id: int) -> None:
        player = self.players[player_id]
        for card_name, amount in list(player.new_development_cards.items()):
            if amount > 0:
                player.development_cards[card_name] += amount
                player.new_development_cards[card_name] = 0

    def legal_initial_settlement_ids(self) -> List[int]:
        legal = []
        occupied = set(self.settlement_owners) | set(self.city_owners)
        for node in self.board.intersections:
            if node.id in occupied:
                continue
            if any(neighbor in occupied for neighbor in node.adjacent_intersection_ids):
                continue
            legal.append(node.id)
        return legal

    def legal_setup_road_ids(self) -> List[int]:
        if self.last_settlement_placed is None:
            return []
        roads = []
        for edge_id in self.board.intersections[self.last_settlement_placed].adjacent_edge_ids:
            if edge_id not in self.road_owners:
                roads.append(edge_id)
        return roads

    def legal_road_ids(self, player_id: int, setup: bool) -> List[int]:
        player = self.players[player_id]
        if player.roads_remaining <= 0:
            return []
        legal = []
        for edge in self.board.edges:
            if edge.id in self.road_owners:
                continue
            if self.can_place_road(player_id, edge.id, setup=setup):
                legal.append(edge.id)
        return legal

    def legal_settlement_ids(self, player_id: int) -> List[int]:
        player = self.players[player_id]
        if player.settlements_remaining <= 0:
            return []
        if not self.can_afford(player_id, BUILD_COSTS["settlement"]):
            return []
        legal = []
        for node in self.board.intersections:
            if self.can_place_settlement(player_id, node.id, setup=False):
                legal.append(node.id)
        return legal

    def legal_city_ids(self, player_id: int) -> List[int]:
        player = self.players[player_id]
        if player.cities_remaining <= 0:
            return []
        if not self.can_afford(player_id, BUILD_COSTS["city"]):
            return []
        return [
            node_id
            for node_id, owner in self.settlement_owners.items()
            if owner == player_id
        ]

    def legal_domestic_trades(self, player_id: int) -> List[Dict[str, object]]:
        player = self.players[player_id]
        trades = []
        for partner in self.players:
            if partner.player_id == player_id:
                continue
            for give_resource in RESOURCE_LIST:
                if player.resources[give_resource] <= 0:
                    continue
                for receive_resource in RESOURCE_LIST:
                    if receive_resource == give_resource:
                        continue
                    if partner.resources[receive_resource] <= 0:
                        continue
                    trades.append(
                        {
                            "partner_id": partner.player_id,
                            "give_resource": give_resource,
                            "receive_resource": receive_resource,
                        }
                    )
        return trades

    def legal_maritime_trades(self, player_id: int) -> List[Dict[str, object]]:
        rates = self.player_harbor_rates(player_id)
        player = self.players[player_id]
        trades = []
        for give_resource, rate in rates.items():
            if player.resources[give_resource] < rate:
                continue
            for receive_resource in RESOURCE_LIST:
                if receive_resource == give_resource:
                    continue
                if self.bank_resources[receive_resource] <= 0:
                    continue
                trades.append(
                    {
                        "give_resource": give_resource,
                        "receive_resource": receive_resource,
                        "rate": rate,
                    }
                )
        return trades

    def legal_robber_hex_ids(self) -> List[int]:
        return [hex_tile.id for hex_tile in self.board.hexes if hex_tile.id != self.robber_hex_id]

    def legal_robber_victims(self, hex_id: int) -> List[int]:
        victims = set()
        for node_id in self.board.hexes[hex_id].corner_ids:
            owner = self.city_owners.get(node_id, self.settlement_owners.get(node_id))
            if owner is None or owner == self.active_player:
                continue
            if self.players[owner].total_resources() > 0:
                victims.add(owner)
        return sorted(victims)

    def can_place_road(self, player_id: int, edge_id: int, setup: bool) -> bool:
        if edge_id in self.road_owners:
            return False
        edge = self.board.edges[edge_id]
        if setup:
            if self.last_settlement_placed is None:
                return False
            return self.last_settlement_placed in edge.intersection_ids
        for node_id in edge.intersection_ids:
            if self.owns_structure(player_id, node_id):
                return True
        for node_id in edge.intersection_ids:
            if self.is_blocked_by_opponent(player_id, node_id):
                continue
            for adjacent_edge_id in self.board.intersections[node_id].adjacent_edge_ids:
                if adjacent_edge_id != edge_id and self.road_owners.get(adjacent_edge_id) == player_id:
                    return True
        return False

    def can_place_settlement(self, player_id: int, intersection_id: int, setup: bool) -> bool:
        if intersection_id in self.settlement_owners or intersection_id in self.city_owners:
            return False
        node = self.board.intersections[intersection_id]
        if any(
            neighbor in self.settlement_owners or neighbor in self.city_owners
            for neighbor in node.adjacent_intersection_ids
        ):
            return False
        if setup:
            return True
        if not any(self.road_owners.get(edge_id) == player_id for edge_id in node.adjacent_edge_ids):
            return False
        return True

    def player_harbor_rates(self, player_id: int) -> Dict[str, int]:
        rates = {resource_name: 4 for resource_name in RESOURCE_LIST}
        for harbor in self.board.harbors:
            edge = self.board.edges[harbor.edge_id]
            if any(self.owns_structure(player_id, node_id) for node_id in edge.intersection_ids):
                if harbor.resource is None:
                    for resource_name in RESOURCE_LIST:
                        rates[resource_name] = min(rates[resource_name], harbor.rate)
                else:
                    rates[harbor.resource.value] = min(rates[harbor.resource.value], harbor.rate)
        return rates

    def can_afford(self, player_id: int, cost: Dict[str, int]) -> bool:
        player = self.players[player_id]
        return all(player.resources[resource_name] >= amount for resource_name, amount in cost.items())

    def pay_cost(self, player_id: int, cost: Dict[str, int]) -> None:
        if not self.can_afford(player_id, cost):
            raise ActionError(f"Player {player_id} cannot afford cost {cost}.")
        player = self.players[player_id]
        for resource_name, amount in cost.items():
            player.resources[resource_name] -= amount
            self.bank_resources[resource_name] += amount

    def place_road(self, player_id: int, edge_id: int) -> None:
        player = self.players[player_id]
        if player.roads_remaining <= 0:
            raise ActionError(f"Player {player_id} has no roads remaining.")
        self.road_owners[edge_id] = player_id
        player.roads_remaining -= 1
        self.log(f"{player_label(player_id)} built a road on edge E{edge_id}.")

    def place_settlement(self, player_id: int, intersection_id: int, is_setup: bool) -> None:
        player = self.players[player_id]
        if player.settlements_remaining <= 0:
            raise ActionError(f"Player {player_id} has no settlements remaining.")
        self.settlement_owners[intersection_id] = player_id
        player.settlements_remaining -= 1
        player.visible_victory_points += 1
        phase_label = "setup " if is_setup else ""
        self.log(
            f"{player_label(player_id)} built a {phase_label}settlement on intersection I{intersection_id}."
        )

    def upgrade_city(self, player_id: int, intersection_id: int) -> None:
        player = self.players[player_id]
        if self.settlement_owners.get(intersection_id) != player_id:
            raise ActionError(f"Player {player_id} does not own a settlement at {intersection_id}.")
        if player.cities_remaining <= 0:
            raise ActionError(f"Player {player_id} has no cities remaining.")
        del self.settlement_owners[intersection_id]
        self.city_owners[intersection_id] = player_id
        player.cities_remaining -= 1
        player.settlements_remaining += 1
        player.visible_victory_points += 1
        self.log(f"{player_label(player_id)} upgraded intersection I{intersection_id} into a city.")

    def owns_structure(self, player_id: int, intersection_id: int) -> bool:
        return self.settlement_owners.get(intersection_id) == player_id or self.city_owners.get(intersection_id) == player_id

    def is_blocked_by_opponent(self, player_id: int, intersection_id: int) -> bool:
        owner = self.city_owners.get(intersection_id, self.settlement_owners.get(intersection_id))
        return owner is not None and owner != player_id

    def grant_second_settlement_resources(self, player_id: int, intersection_id: Optional[int]) -> None:
        if intersection_id is None:
            return
        for hex_id in self.board.intersections[intersection_id].adjacent_hex_ids:
            tile = self.board.hexes[hex_id]
            if tile.resource is None:
                continue
            if self.bank_resources[tile.resource.value] <= 0:
                continue
            self.take_bank_resource(player_id, tile.resource.value, 1)

    def take_bank_resource(self, player_id: int, resource_name: str, amount: int) -> None:
        if self.bank_resources[resource_name] < amount:
            raise ActionError(f"Bank does not have {amount} {resource_name}.")
        self.bank_resources[resource_name] -= amount
        self.players[player_id].resources[resource_name] += amount

    def distribute_resources_for_roll(self, roll: int) -> None:
        resource_claims: Dict[str, List[Tuple[int, int]]] = {resource_name: [] for resource_name in RESOURCE_LIST}
        for tile in self.board.hexes:
            if tile.token != roll or tile.id == self.robber_hex_id or tile.resource is None:
                continue
            resource_name = tile.resource.value
            for node_id in tile.corner_ids:
                if node_id in self.city_owners:
                    resource_claims[resource_name].append((self.city_owners[node_id], 2))
                elif node_id in self.settlement_owners:
                    resource_claims[resource_name].append((self.settlement_owners[node_id], 1))
        distributed_any = False
        for resource_name, claims in resource_claims.items():
            if not claims:
                continue
            total_claim = sum(amount for _, amount in claims)
            bank_available = self.bank_resources[resource_name]
            if bank_available >= total_claim:
                distributed_any = True
                for owner, amount in claims:
                    self.take_bank_resource(owner, resource_name, amount)
                    self.log(
                        f"{player_label(owner)} received {amount} {resource_name} from roll {roll}."
                    )
                continue
            unique_players = {owner for owner, _ in claims}
            if len(unique_players) == 1 and bank_available > 0:
                owner = next(iter(unique_players))
                distributed_any = True
                self.take_bank_resource(owner, resource_name, bank_available)
                self.log(
                    f"{player_label(owner)} received {bank_available} {resource_name} from roll {roll} "
                    f"(bank shortage)."
                )
            else:
                self.log(f"Bank shortage prevented {resource_name} from being distributed.")
        if not distributed_any:
            self.log(f"Roll {roll} produced no resources.")

    def steal_random_resource(self, thief_id: int, victim_id: int) -> Optional[str]:
        victim = self.players[victim_id]
        available = [resource_name for resource_name, amount in victim.resources.items() for _ in range(amount)]
        if not available:
            return None
        resource_name = random.choice(available)
        victim.resources[resource_name] -= 1
        self.players[thief_id].resources[resource_name] += 1
        self.log(
            f"{player_label(thief_id)} stole 1 {resource_name} from {player_label(victim_id)}."
        )
        return resource_name

    def required_discard_count(self, player_id: int) -> int:
        return self.players[player_id].total_resources() // 2

    def advance_setup_turn(self) -> None:
        num_players = len(self.players)
        if self.setup_round == 1:
            if self.setup_order_index < num_players - 1:
                self.setup_order_index += 1
                self.active_player = self.setup_order_index
            else:
                self.setup_round = 2
        else:
            if self.setup_order_index > 0:
                self.setup_order_index -= 1
                self.active_player = self.setup_order_index
            else:
                self.pending_phase = "turn_start"
                self.active_player = self.starting_player
                self.setup_order_index = 0
                self.last_settlement_placed = None
                self.dev_card_played_this_turn = False
                self.current_turn_road_builds = 0
                self.turn_number = 0
                self.log("Setup complete.")
                return
        self.active_player = self.setup_order_index
        self.pending_phase = "setup_settlement"
        self.last_settlement_placed = None

    def update_longest_road(self) -> None:
        lengths = {player.player_id: self.compute_longest_road_length(player.player_id) for player in self.players}
        qualifying = {player_id: length for player_id, length in lengths.items() if length >= 5}
        current_owner = self.longest_road_owner
        if current_owner is None:
            if qualifying:
                max_len = max(qualifying.values())
                winners = [player_id for player_id, length in qualifying.items() if length == max_len]
                if len(winners) == 1:
                    self.longest_road_owner = winners[0]
            return
        current_length = lengths[current_owner]
        if current_length < 5:
            challengers = {player_id: length for player_id, length in qualifying.items() if player_id != current_owner}
            if challengers:
                max_len = max(challengers.values())
                winners = [player_id for player_id, length in challengers.items() if length == max_len]
                self.longest_road_owner = winners[0] if len(winners) == 1 else None
            else:
                self.longest_road_owner = None
            return
        for player_id, length in qualifying.items():
            if player_id != current_owner and length > current_length:
                self.longest_road_owner = player_id
                return

    def compute_longest_road_length(self, player_id: int) -> int:
        owned_edges = [edge_id for edge_id, owner in self.road_owners.items() if owner == player_id]
        if not owned_edges:
            return 0
        best = 0
        start_nodes = set()
        for edge_id in owned_edges:
            start_nodes.update(self.board.edges[edge_id].intersection_ids)
        for node_id in start_nodes:
            best = max(best, self._dfs_longest_road_from_node(player_id, node_id, set()))
        return best

    def _dfs_longest_road_from_node(
        self, player_id: int, node_id: int, used_edges: set[int]
    ) -> int:
        best = 0
        for edge_id in self.board.intersections[node_id].adjacent_edge_ids:
            if edge_id in used_edges:
                continue
            if self.road_owners.get(edge_id) != player_id:
                continue
            edge = self.board.edges[edge_id]
            next_node = (
                edge.intersection_ids[0]
                if edge.intersection_ids[1] == node_id
                else edge.intersection_ids[1]
            )
            used_now = set(used_edges)
            used_now.add(edge_id)
            extension = 0
            if not self.is_blocked_by_opponent(player_id, next_node):
                extension = self._dfs_longest_road_from_node(player_id, next_node, used_now)
            best = max(best, 1 + extension)
        return best

    def update_largest_army(self) -> None:
        current_owner = self.largest_army_owner
        if current_owner is None:
            candidates = [player.player_id for player in self.players if player.played_knights >= 3]
            if not candidates:
                return
            best = max(self.players[player_id].played_knights for player_id in candidates)
            winners = [player_id for player_id in candidates if self.players[player_id].played_knights == best]
            if len(winners) == 1:
                self.largest_army_owner = winners[0]
            return
        current_count = self.players[current_owner].played_knights
        for player in self.players:
            if player.player_id != current_owner and player.played_knights > current_count:
                self.largest_army_owner = player.player_id
                return

    def visible_victory_points(self, player_id: int) -> int:
        points = self.players[player_id].visible_victory_points
        if self.longest_road_owner == player_id:
            points += 2
        if self.largest_army_owner == player_id:
            points += 2
        return points

    def total_victory_points(self, player_id: int) -> int:
        player = self.players[player_id]
        return self.visible_victory_points(player_id) + player.development_cards["victory_point"] + player.new_development_cards["victory_point"]

    def check_for_winner(self, start_of_turn: bool = False) -> None:
        if self.pending_phase in SETUP_PHASES:
            return
        for player in self.players:
            if self.total_victory_points(player.player_id) >= 10:
                if start_of_turn and player.player_id == self.active_player:
                    self.winner = player.player_id
                elif player.player_id == self.active_player:
                    self.winner = player.player_id
        if self.winner is not None:
            self.pending_phase = "game_over"
            self.log(
                f"{player_label(self.winner)} wins with {self.total_victory_points(self.winner)} VP."
            )

    def _consume_development_card(self, card_name: str) -> None:
        player = self.players[self.active_player]
        if player.development_cards[card_name] <= 0:
            raise ActionError(f"Player {self.active_player} cannot play {card_name}.")
        player.development_cards[card_name] -= 1

    def log(self, message: str) -> None:
        self.event_log.append(message)


def build_game(num_players: int = 4, seed: Optional[int] = None) -> GameState:
    if num_players not in (3, 4):
        raise ValueError("Base Catan supports 3 or 4 players.")

    rng = random.Random(seed)
    board = build_board_topology(rng)
    robber_hex_id = next(tile.id for tile in board.hexes if tile.terrain == Terrain.DESERT)
    development_deck = build_development_deck(rng)

    return GameState(
        board=board,
        players=[PlayerState(player_id=i) for i in range(num_players)],
        robber_hex_id=robber_hex_id,
        development_deck=development_deck,
    )


def build_development_deck(rng: random.Random) -> List[str]:
    deck = []
    for card_name, count in DEVELOPMENT_CARD_COUNTS.items():
        deck.extend([card_name] * count)
    rng.shuffle(deck)
    return deck


def build_board_topology(rng: random.Random) -> BoardTopology:
    axial_positions = [
        (q, r)
        for q in range(-BOARD_RADIUS, BOARD_RADIUS + 1)
        for r in range(-BOARD_RADIUS, BOARD_RADIUS + 1)
        if max(abs(q), abs(r), abs(-q - r)) <= BOARD_RADIUS
    ]
    axial_positions.sort(key=lambda pos: (pos[1], pos[0]))

    terrain_deck = (
        [Terrain.FOREST] * 4
        + [Terrain.PASTURE] * 4
        + [Terrain.FIELDS] * 4
        + [Terrain.HILLS] * 3
        + [Terrain.MOUNTAINS] * 3
        + [Terrain.DESERT]
    )
    rng.shuffle(terrain_deck)

    tokens = assign_number_tokens(axial_positions, terrain_deck, rng)

    corner_lookup: Dict[Tuple[float, float], int] = {}
    corner_positions: List[Tuple[float, float]] = []
    edge_lookup: Dict[Tuple[int, int], int] = {}
    edge_to_hexes: Dict[int, List[int]] = {}
    corner_to_hexes: Dict[int, List[int]] = {}
    hex_corner_ids: List[Tuple[int, ...]] = []
    hex_edge_ids: List[Tuple[int, ...]] = []

    for hex_id, (q, r) in enumerate(axial_positions):
        center_x, center_y = axial_to_pixel(q, r)
        local_corner_ids: List[int] = []
        local_edge_ids: List[int] = []

        for angle_deg in HEX_VERTEX_ANGLES:
            angle_rad = math.radians(angle_deg)
            px = round(center_x + math.cos(angle_rad), 6)
            py = round(center_y + math.sin(angle_rad), 6)
            key = (px, py)
            if key not in corner_lookup:
                corner_lookup[key] = len(corner_positions)
                corner_positions.append(key)
            corner_id = corner_lookup[key]
            local_corner_ids.append(corner_id)
            corner_to_hexes.setdefault(corner_id, []).append(hex_id)

        for i in range(6):
            a = local_corner_ids[i]
            b = local_corner_ids[(i + 1) % 6]
            edge_key = tuple(sorted((a, b)))
            if edge_key not in edge_lookup:
                edge_lookup[edge_key] = len(edge_lookup)
            edge_id = edge_lookup[edge_key]
            local_edge_ids.append(edge_id)
            edge_to_hexes.setdefault(edge_id, []).append(hex_id)

        hex_corner_ids.append(tuple(local_corner_ids))
        hex_edge_ids.append(tuple(local_edge_ids))

    hex_neighbors = {
        hex_id: tuple(
            other_id
            for other_id, (other_q, other_r) in enumerate(axial_positions)
            if (other_q, other_r) in {
                (q + 1, r),
                (q - 1, r),
                (q, r + 1),
                (q, r - 1),
                (q + 1, r - 1),
                (q - 1, r + 1),
            }
        )
        for hex_id, (q, r) in enumerate(axial_positions)
    }

    edges: List[Edge] = []
    corner_to_edges: Dict[int, List[int]] = {}
    for (a, b), edge_id in sorted(edge_lookup.items(), key=lambda item: item[1]):
        edge = Edge(
            id=edge_id,
            intersection_ids=(a, b),
            adjacent_hex_ids=tuple(sorted(edge_to_hexes[edge_id])),
            coastal=len(edge_to_hexes[edge_id]) == 1,
        )
        edges.append(edge)
        corner_to_edges.setdefault(a, []).append(edge_id)
        corner_to_edges.setdefault(b, []).append(edge_id)

    intersections: List[Intersection] = []
    for corner_id, (x, y) in enumerate(corner_positions):
        adjacent_edge_ids = tuple(sorted(corner_to_edges.get(corner_id, [])))
        adjacent_nodes = set()
        for edge_id in adjacent_edge_ids:
            a, b = edges[edge_id].intersection_ids
            adjacent_nodes.add(a if b == corner_id else b)
        intersections.append(
            Intersection(
                id=corner_id,
                x=x,
                y=y,
                adjacent_hex_ids=tuple(sorted(corner_to_hexes.get(corner_id, []))),
                adjacent_edge_ids=adjacent_edge_ids,
                adjacent_intersection_ids=tuple(sorted(adjacent_nodes)),
                coastal=len(corner_to_hexes.get(corner_id, [])) < 3,
            )
        )

    hexes: List[HexTile] = []
    for hex_id, ((q, r), terrain) in enumerate(zip(axial_positions, terrain_deck)):
        center_x, center_y = axial_to_pixel(q, r)
        hexes.append(
            HexTile(
                id=hex_id,
                q=q,
                r=r,
                x=center_x,
                y=center_y,
                terrain=terrain,
                resource=TERRAIN_TO_RESOURCE[terrain],
                token=tokens[hex_id],
                corner_ids=hex_corner_ids[hex_id],
                edge_ids=hex_edge_ids[hex_id],
                neighbor_hex_ids=tuple(sorted(hex_neighbors[hex_id])),
            )
        )

    harbors = assign_harbors(edges, intersections, rng)

    return BoardTopology(
        hexes=tuple(hexes),
        intersections=tuple(intersections),
        edges=tuple(edges),
        harbors=tuple(harbors),
    )


def assign_number_tokens(
    axial_positions: Sequence[Tuple[int, int]],
    terrains: Sequence[Terrain],
    rng: random.Random,
) -> Dict[int, Optional[int]]:
    token_values = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
    non_desert_hex_ids = [i for i, terrain in enumerate(terrains) if terrain != Terrain.DESERT]
    neighbor_map = build_hex_neighbor_map(axial_positions)

    for _ in range(500):
        rng.shuffle(token_values)
        assignment = dict(zip(non_desert_hex_ids, token_values))
        if not any(
            assignment.get(a) in (6, 8) and assignment.get(b) in (6, 8)
            for a, neighbors in neighbor_map.items()
            for b in neighbors
            if a < b
        ):
            tokens: Dict[int, Optional[int]] = {
                i: None for i, terrain in enumerate(terrains) if terrain == Terrain.DESERT
            }
            tokens.update(assignment)
            return tokens

    tokens = {i: None for i, terrain in enumerate(terrains) if terrain == Terrain.DESERT}
    tokens.update(dict(zip(non_desert_hex_ids, token_values)))
    return tokens


def build_hex_neighbor_map(
    axial_positions: Sequence[Tuple[int, int]]
) -> Dict[int, Tuple[int, ...]]:
    position_to_id = {pos: i for i, pos in enumerate(axial_positions)}
    neighbor_offsets = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))
    result: Dict[int, Tuple[int, ...]] = {}
    for hex_id, (q, r) in enumerate(axial_positions):
        neighbors = []
        for dq, dr in neighbor_offsets:
            other = position_to_id.get((q + dq, r + dr))
            if other is not None:
                neighbors.append(other)
        result[hex_id] = tuple(sorted(neighbors))
    return result


def assign_harbors(
    edges: Sequence[Edge], intersections: Sequence[Intersection], rng: random.Random
) -> List[Harbor]:
    coastal_edges = [edge for edge in edges if edge.coastal]
    coastal_edges.sort(
        key=lambda edge: math.atan2(
            midpoint(intersections, edge)[1], midpoint(intersections, edge)[0]
        )
    )

    harbor_slots = []
    step = len(coastal_edges) / 9
    for i in range(9):
        index = int(round(i * step)) % len(coastal_edges)
        harbor_slots.append(coastal_edges[index].id)

    harbor_slots = dedupe_preserve_order(harbor_slots)
    while len(harbor_slots) < 9:
        for edge in coastal_edges:
            if edge.id not in harbor_slots:
                harbor_slots.append(edge.id)
            if len(harbor_slots) == 9:
                break

    harbor_types: List[Tuple[int, Optional[Resource]]] = [
        (3, None),
        (3, None),
        (3, None),
        (3, None),
        (2, Resource.BRICK),
        (2, Resource.LUMBER),
        (2, Resource.ORE),
        (2, Resource.GRAIN),
        (2, Resource.WOOL),
    ]
    rng.shuffle(harbor_types)

    return [
        Harbor(id=i, edge_id=edge_id, rate=rate, resource=resource)
        for i, (edge_id, (rate, resource)) in enumerate(zip(harbor_slots, harbor_types))
    ]


def dedupe_preserve_order(values: Iterable[int]) -> List[int]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def normalize_resource_dict(resources: object) -> Dict[str, int]:
    if not isinstance(resources, dict):
        raise ActionError("Resource payload must be a dict.")
    normalized = {resource_name: 0 for resource_name in RESOURCE_LIST}
    for resource_name, amount in resources.items():
        if resource_name not in normalized:
            raise ActionError(f"Unknown resource: {resource_name}")
        value = int(amount)
        if value < 0:
            raise ActionError("Resource amounts cannot be negative.")
        normalized[resource_name] = value
    return normalized


def year_of_plenty_choices(bank_resources: Dict[str, int]) -> List[List[str]]:
    available = [resource_name for resource_name, amount in bank_resources.items() if amount > 0]
    choices = []
    for i, first in enumerate(available):
        for second in available[i:]:
            if first == second and bank_resources[first] < 2:
                continue
            choices.append([first, second])
    return choices


def enumerate_discards(resources: Dict[str, int], total_to_discard: int) -> List[Dict[str, int]]:
    resource_names = list(RESOURCE_LIST)
    results: List[Dict[str, int]] = []

    def backtrack(index: int, remaining: int, current: Dict[str, int]) -> None:
        if index == len(resource_names):
            if remaining == 0:
                results.append(dict(current))
            return
        resource_name = resource_names[index]
        max_amount = min(resources[resource_name], remaining)
        for amount in range(max_amount + 1):
            current[resource_name] = amount
            backtrack(index + 1, remaining - amount, current)
        current.pop(resource_name, None)

    backtrack(0, total_to_discard, {})
    return [normalize_resource_dict(choice) for choice in results]


def axial_to_pixel(q: int, r: int) -> Tuple[float, float]:
    return (
        math.sqrt(3) * (q + r / 2),
        1.5 * r,
    )


def midpoint(intersections: Sequence[Intersection], edge: Edge) -> Tuple[float, float]:
    a, b = edge.intersection_ids
    return (
        (intersections[a].x + intersections[b].x) / 2,
        (intersections[a].y + intersections[b].y) / 2,
    )


class SvgRenderer:
    def __init__(self, board: BoardTopology, state: Optional[GameState] = None) -> None:
        self.board = board
        self.state = state
        self.scale = 90
        self.margin = 100

    def write(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.write_text(self.render(), encoding="utf-8")
        return output_path

    def render(self) -> str:
        xs = [node.x for node in self.board.intersections]
        ys = [node.y for node in self.board.intersections]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = (max_x - min_x) * self.scale + 2 * self.margin
        height = (max_y - min_y) * self.scale + 2 * self.margin

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
            '<rect width="100%" height="100%" fill="#f7f2e8" />',
        ]

        for harbor in self.board.harbors:
            edge = self.board.edges[harbor.edge_id]
            mx, my = self._point(*midpoint(self.board.intersections, edge), min_x, min_y)
            label = f"{harbor.rate}:1"
            stroke_color = RESOURCE_COLORS["generic"]
            if harbor.resource:
                label = f"{harbor.resource.value[:2].upper()} {label}"
                stroke_color = RESOURCE_COLORS[harbor.resource.value]
            svg_parts.append(
                f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="18" fill="#fdf8ef" stroke="{stroke_color}" stroke-width="2" />'
            )
            svg_parts.append(
                f'<text x="{mx:.1f}" y="{my + 5:.1f}" text-anchor="middle" font-size="12" font-family="Verdana">{label}</text>'
            )

        for tile in self.board.hexes:
            points = [
                self._point(
                    self.board.intersections[corner_id].x,
                    self.board.intersections[corner_id].y,
                    min_x,
                    min_y,
                )
                for corner_id in tile.corner_ids
            ]
            point_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            fill = TERRAIN_COLORS[tile.terrain.value]
            svg_parts.append(
                f'<polygon points="{point_str}" fill="{fill}" stroke="#47341f" stroke-width="2" />'
            )
            tx, ty = self._point(tile.x, tile.y, min_x, min_y)
            svg_parts.append(
                f'<text x="{tx:.1f}" y="{ty - 18:.1f}" text-anchor="middle" font-size="12" font-family="Verdana">{tile.terrain.value}</text>'
            )
            if tile.token is not None:
                svg_parts.append(
                    f'<circle cx="{tx:.1f}" cy="{ty + 6:.1f}" r="16" fill="#faf4dd" stroke="#47341f" stroke-width="2" />'
                )
                token_color = "#b11f18" if tile.token in (6, 8) else "#222222"
                svg_parts.append(
                    f'<text x="{tx:.1f}" y="{ty + 11:.1f}" text-anchor="middle" font-size="16" font-weight="bold" fill="{token_color}" font-family="Verdana">{tile.token}</text>'
                )
            if self.state and self.state.robber_hex_id == tile.id:
                svg_parts.append(
                    f'<rect x="{tx - 8:.1f}" y="{ty - 52:.1f}" width="16" height="24" rx="4" fill="#1f1f1f" />'
                )

        for edge in self.board.edges:
            a, b = edge.intersection_ids
            x1, y1 = self._point(self.board.intersections[a].x, self.board.intersections[a].y, min_x, min_y)
            x2, y2 = self._point(self.board.intersections[b].x, self.board.intersections[b].y, min_x, min_y)
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            owner = self.state.road_owners.get(edge.id) if self.state else None
            stroke = PLAYER_COLORS[owner] if owner is not None else "#786858"
            width_px = 8 if owner is not None else 3
            svg_parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width_px}" stroke-linecap="round" />'
            )
            svg_parts.append(
                f'<text x="{mx + 6:.1f}" y="{my + 12:.1f}" font-size="9" fill="#5a4e40" font-family="Verdana">E{edge.id}</text>'
            )

        for node in self.board.intersections:
            x, y = self._point(node.x, node.y, min_x, min_y)
            radius = 5
            fill = "#f7f2e8"
            owner = None
            if self.state:
                if node.id in self.state.city_owners:
                    owner = self.state.city_owners[node.id]
                    radius = 11
                elif node.id in self.state.settlement_owners:
                    owner = self.state.settlement_owners[node.id]
                    radius = 8
            if owner is not None:
                fill = PLAYER_COLORS[owner]
            svg_parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="#222222" stroke-width="2" />'
            )
            svg_parts.append(
                f'<text x="{x + 10:.1f}" y="{y - 4:.1f}" font-size="10" fill="#333333" font-family="Verdana">I{node.id}</text>'
            )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def _point(self, x: float, y: float, min_x: float, min_y: float) -> Tuple[float, float]:
        return (
            (x - min_x) * self.scale + self.margin,
            (y - min_y) * self.scale + self.margin,
        )


def summarize_board(board: BoardTopology) -> Dict[str, int]:
    return {
        "hexes": len(board.hexes),
        "intersections": len(board.intersections),
        "edges": len(board.edges),
        "harbors": len(board.harbors),
    }


def run_scripted_opening(state: GameState, turns: int = 12) -> None:
    for _ in range(turns):
        actions = state.legal_actions()
        if not actions:
            return
        chosen = choose_reasonable_action(state, actions)
        state.apply_action(chosen)
        if state.pending_phase == "game_over":
            return


def choose_reasonable_action(state: GameState, actions: Sequence[Action]) -> Action:
    priorities = {
        "build_settlement": 100,
        "build_city": 95,
        "build_road_free": 90,
        "build_road": 80,
        "buy_development_card": 70,
        "play_year_of_plenty": 65,
        "play_road_building": 64,
        "play_knight": 63,
        "trade_maritime": 50,
        "roll_dice": 40,
        "discard_resources": 30,
        "move_robber": 20,
        "finish_road_building": 10,
        "end_turn": 0,
    }
    return max(actions, key=lambda action: priorities.get(action.action_type, 0))


def print_state_summary(state: GameState) -> None:
    print()
    print(
        f"Phase={state.pending_phase} | Active Player={state.active_player} | "
        f"Turn={state.turn_number} | Last Roll={state.last_roll} | Winner={state.winner}"
    )
    for player in state.players:
        resources = ", ".join(
            f"{resource_name[:2]}:{amount}" for resource_name, amount in player.resources.items()
        )
        print(
            f"{player_label(player.player_id)} | VP={state.total_victory_points(player.player_id)} "
            f"(visible {state.visible_victory_points(player.player_id)}) | "
            f"Roads={15 - player.roads_remaining} Settlements={5 - player.settlements_remaining} "
            f"Cities={4 - player.cities_remaining} | Resources [{resources}]"
        )


def print_legal_actions(actions: Sequence[Action]) -> None:
    print()
    print("Legal actions:")
    for index, action in enumerate(actions):
        print(f"  [{index}] {action.action_type} {json.dumps(action.params, sort_keys=True)}")


def current_prompt_hint(state: GameState, actions: Sequence[Action]) -> str:
    if not actions:
        return "Choose action"
    action_types = {action.action_type for action in actions}
    if action_types == {"build_settlement"}:
        return "Choose settlement: intersection id (e.g. 21 or i21), or a<index>"
    if action_types in ({"build_road"}, {"build_road_free"}):
        return "Choose road: edge id (e.g. 17 or e17), or a<index>"
    if action_types == {"build_city"}:
        return "Choose city upgrade: intersection id (e.g. 21 or i21), or a<index>"
    return "Choose action"


def resolve_cli_selection(
    raw: str,
    state: GameState,
    actions: Sequence[Action],
) -> Optional[Action]:
    action_types = {action.action_type for action in actions}
    placement_by_intersection = action_types in ({"build_settlement"}, {"build_city"})
    placement_by_edge = action_types in ({"build_road"}, {"build_road_free"})

    if raw.lower().startswith("a") and raw[1:].isdigit():
        index = int(raw[1:])
        if 0 <= index < len(actions):
            return actions[index]
        print("Action index out of range.")
        return None

    if raw.isdigit():
        numeric_id = int(raw)
        if placement_by_intersection:
            matches = [
                action
                for action in actions
                if action.params.get("intersection_id") == numeric_id
            ]
            if len(matches) == 1:
                return matches[0]
            if 0 <= numeric_id < len(actions):
                return actions[numeric_id]
            print(
                f"Intersection I{numeric_id} is not legal right now, and there is no action index {numeric_id}."
            )
            return None
        if placement_by_edge:
            matches = [
                action for action in actions if action.params.get("edge_id") == numeric_id
            ]
            if len(matches) == 1:
                return matches[0]
            if 0 <= numeric_id < len(actions):
                return actions[numeric_id]
            print(
                f"Edge E{numeric_id} is not legal right now, and there is no action index {numeric_id}."
            )
            return None
        index = numeric_id
        if 0 <= index < len(actions):
            return actions[index]
        print("Action index out of range.")
        return None

    normalized = raw.lower()
    if normalized.startswith("i") and normalized[1:].isdigit():
        intersection_id = int(normalized[1:])
        matches = [
            action
            for action in actions
            if action.params.get("intersection_id") == intersection_id
        ]
        if len(matches) == 1:
            return matches[0]
        print(f"Intersection I{intersection_id} is not a legal choice right now.")
        return None

    if normalized.startswith("e") and normalized[1:].isdigit():
        edge_id = int(normalized[1:])
        matches = [
            action for action in actions if action.params.get("edge_id") == edge_id
        ]
        if len(matches) == 1:
            return matches[0]
        print(f"Edge E{edge_id} is not a legal choice right now.")
        return None

    print("Please enter a valid board id like 21 / i21 / 17 / e17, or use a<index>.")
    return None


def interactive_cli(seed: int = 7, num_players: int = 4) -> None:
    state = build_game(num_players=num_players, seed=seed)
    last_printed_event_index = 0
    print(f"Started Catan CLI with seed={seed} and players={num_players}.")
    print("Commands: board id (like 21, i21, 17, e17), a<index>, svg, obs, log, auto, help, quit")

    while True:
        state.write_svg("catan_board.svg")
        Path("catan_observation_sample.json").write_text(
            json.dumps(state.to_observation(state.active_player), indent=2),
            encoding="utf-8",
        )
        Path("catan_event_log.txt").write_text("\n".join(state.event_log), encoding="utf-8")

        print_state_summary(state)
        if state.pending_phase == "game_over":
            print(f"Game over. Player {state.winner} wins.")
            return

        if last_printed_event_index < len(state.event_log):
            print()
            print("Recent events:")
            for event in state.event_log[last_printed_event_index:]:
                print(f"  - {event}")
            last_printed_event_index = len(state.event_log)

        actions = state.legal_actions()
        print_legal_actions(actions)
        raw = input(f"\n{current_prompt_hint(state, actions)}> ").strip()

        if raw in {"q", "quit", "exit"}:
            print("Exiting CLI.")
            return
        if raw == "help":
            print("In placement phases, plain numbers are treated as board ids.")
            print("Use i21 or 21 for intersections, e17 or 17 for edges, and a3 for action index 3.")
            print("Other commands: svg, obs, log, auto, quit")
            continue
        if raw == "svg":
            print("Board SVG refreshed at catan_board.svg")
            continue
        if raw == "obs":
            print(json.dumps(state.to_observation(state.active_player), indent=2))
            continue
        if raw == "log":
            print("\n".join(state.event_log[-20:]) or "(no events yet)")
            continue
        if raw == "auto":
            chosen = choose_reasonable_action(state, actions)
            print(f"Auto-playing: {chosen.action_type} {chosen.params}")
            state.apply_action(chosen)
            continue

        chosen = resolve_cli_selection(raw, state, actions)
        if chosen is None:
            continue

        try:
            state.apply_action(chosen)
        except ActionError as exc:
            print(f"Action failed: {exc}")


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "play":
        interactive_cli()
        return

    state = build_game(seed=7)
    print(json.dumps(summarize_board(state.board), indent=2))
    run_scripted_opening(state, turns=24)
    svg_path = Path("catan_board.svg")
    state.write_svg(svg_path)
    print(f"Wrote {svg_path}")
    observation = state.to_observation(state.active_player)
    Path("catan_observation_sample.json").write_text(
        json.dumps(observation, indent=2),
        encoding="utf-8",
    )
    print("Wrote catan_observation_sample.json")
    Path("catan_event_log.txt").write_text("\n".join(state.event_log), encoding="utf-8")
    print("Wrote catan_event_log.txt")


if __name__ == "__main__":
    main()
