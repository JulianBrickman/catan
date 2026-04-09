import unittest

from catan import Action, ActionError, Terrain, build_game


def find_edge_path(board, length):
    def neighbors(edge_id):
        edge = board.edges[edge_id]
        adjacent = set()
        for node_id in edge.intersection_ids:
            adjacent.update(board.intersections[node_id].adjacent_edge_ids)
        adjacent.discard(edge_id)
        return sorted(adjacent)

    def dfs(path, used):
        if len(path) == length:
            return path
        for next_edge in neighbors(path[-1]):
            if next_edge in used:
                continue
            result = dfs(path + [next_edge], used | {next_edge})
            if result:
                return result
        return None

    for start_edge in range(len(board.edges)):
        result = dfs([start_edge], {start_edge})
        if result:
            return result
    raise AssertionError(f"Could not find a simple path of {length} edges.")


def shared_node(board, edge_a, edge_b):
    nodes = set(board.edges[edge_a].intersection_ids) & set(board.edges[edge_b].intersection_ids)
    if len(nodes) != 1:
        raise AssertionError(f"Expected one shared node between edges {edge_a} and {edge_b}.")
    return next(iter(nodes))


def find_fork_triplet(board):
    for node in board.intersections:
        if len(node.adjacent_edge_ids) >= 3:
            return tuple(node.adjacent_edge_ids[:3])
    raise AssertionError("Could not find a fork node.")


class CatanRulesTest(unittest.TestCase):
    def test_setup_distance_rule_removes_adjacent_intersections(self):
        state = build_game(seed=1)
        chosen = 10
        state.place_settlement(0, chosen, is_setup=True)

        legal = set(state.legal_initial_settlement_ids())
        self.assertNotIn(chosen, legal)
        for neighbor in state.board.intersections[chosen].adjacent_intersection_ids:
            self.assertNotIn(neighbor, legal)

    def test_second_setup_settlement_grants_adjacent_resources(self):
        state = build_game(seed=2)
        node_id = next(
            node.id
            for node in state.board.intersections
            if any(state.board.hexes[hex_id].terrain != Terrain.DESERT for hex_id in node.adjacent_hex_ids)
        )
        expected = sum(
            1 for hex_id in state.board.intersections[node_id].adjacent_hex_ids
            if state.board.hexes[hex_id].terrain != Terrain.DESERT
        )

        state.active_player = 0
        state.setup_round = 2
        state.pending_phase = "setup_road"
        state.place_settlement(0, node_id, is_setup=True)
        state.last_settlement_placed = node_id

        edge_id = state.legal_setup_road_ids()[0]
        state.apply_action(Action("build_road", {"edge_id": edge_id}))

        self.assertEqual(state.players[0].total_resources(), expected)

    def test_roll_distributes_resources_to_all_players_on_matching_hexes(self):
        state = build_game(seed=3)
        tile = next(tile for tile in state.board.hexes if tile.resource is not None and tile.token is not None)
        node_a, node_b = tile.corner_ids[0], tile.corner_ids[2]

        state.place_settlement(0, node_a, is_setup=True)
        state.place_settlement(1, node_b, is_setup=True)
        state.pending_phase = "turn_start"
        state.active_player = 2
        state.robber_hex_id = next(other.id for other in state.board.hexes if other.id != tile.id)

        state.roll_dice(forced_roll=tile.token)

        resource_name = tile.resource.value
        self.assertEqual(state.players[0].resources[resource_name], 1)
        self.assertEqual(state.players[1].resources[resource_name], 1)
        self.assertEqual(state.pending_phase, "main")

    def test_robber_allows_specific_victim_choice_and_random_resource_from_that_victim(self):
        state = build_game(seed=4)
        tile = next(tile for tile in state.board.hexes if tile.resource is not None)
        node_a, node_b = tile.corner_ids[0], tile.corner_ids[2]

        state.place_settlement(0, node_a, is_setup=True)
        state.place_settlement(2, node_b, is_setup=True)
        state.players[2].resources["wool"] = 1
        state.active_player = 1
        state.pending_phase = "robber_move"
        state.robber_hex_id = next(other.id for other in state.board.hexes if other.id != tile.id)

        state.apply_action(Action("move_robber", {"hex_id": tile.id, "victim_id": 2}))

        self.assertEqual(state.players[1].resources["wool"], 1)
        self.assertEqual(state.players[2].resources["wool"], 0)
        self.assertEqual(state.pending_phase, "turn_start")

    def test_longest_road_is_removed_when_opponent_blocks_the_path(self):
        state = build_game(seed=5)
        path = find_edge_path(state.board, 5)
        for edge_id in path:
            state.place_road(0, edge_id)

        state.update_longest_road()
        self.assertEqual(state.longest_road_owner, 0)
        self.assertEqual(state.compute_longest_road_length(0), 5)

        blocker_node = shared_node(state.board, path[2], path[3])
        state.place_settlement(1, blocker_node, is_setup=False)
        state.update_longest_road()

        self.assertIsNone(state.longest_road_owner)
        self.assertLess(state.compute_longest_road_length(0), 5)

    def test_largest_army_transfers_to_player_with_more_played_knights(self):
        state = build_game(seed=6)
        state.players[0].played_knights = 3
        state.update_largest_army()
        self.assertEqual(state.largest_army_owner, 0)

        state.players[1].played_knights = 4
        state.update_largest_army()
        self.assertEqual(state.largest_army_owner, 1)

    def test_longest_road_counts_only_best_branch_at_a_fork(self):
        state = build_game(seed=9)
        edge_a, edge_b, edge_c = find_fork_triplet(state.board)
        for edge_id in (edge_a, edge_b, edge_c):
            state.place_road(0, edge_id)

        self.assertEqual(state.compute_longest_road_length(0), 2)

    def test_longest_road_counts_a_loop_without_reusing_edges(self):
        state = build_game(seed=10)
        cycle = list(state.board.hexes[0].edge_ids)
        for edge_id in cycle:
            state.place_road(0, edge_id)

        self.assertEqual(state.compute_longest_road_length(0), len(cycle))

    def test_longest_road_tie_does_not_transfer_from_current_owner(self):
        state = build_game(seed=11)
        path_a = find_edge_path(state.board, 5)
        used = set(path_a)
        path_b = None
        for start in range(len(state.board.edges)):
            if start in used:
                continue

            def dfs(path, seen):
                if len(path) == 5:
                    return path
                edge = state.board.edges[path[-1]]
                next_edges = set()
                for node_id in edge.intersection_ids:
                    next_edges.update(state.board.intersections[node_id].adjacent_edge_ids)
                for nxt in sorted(next_edges):
                    if nxt in seen or nxt in used:
                        continue
                    result = dfs(path + [nxt], seen | {nxt})
                    if result:
                        return result
                return None

            candidate = dfs([start], {start})
            if candidate:
                path_b = candidate
                break

        if path_b is None:
            self.skipTest("Could not find a disjoint second path of length 5 on this board.")

        for edge_id in path_a:
            state.place_road(0, edge_id)
        state.update_longest_road()
        self.assertEqual(state.longest_road_owner, 0)

        for edge_id in path_b:
            state.place_road(1, edge_id)
        state.update_longest_road()

        self.assertEqual(state.compute_longest_road_length(0), 5)
        self.assertEqual(state.compute_longest_road_length(1), 5)
        self.assertEqual(state.longest_road_owner, 0)

    def test_svg_contains_edge_and_intersection_labels(self):
        state = build_game(seed=7)
        svg = state.write_svg("test_board.svg").read_text(encoding="utf-8")
        self.assertIn("I0", svg)
        self.assertIn("E0", svg)

    def test_domestic_trade_offer_can_be_accepted_by_a_later_player(self):
        state = build_game(seed=8)
        state.pending_phase = "main"
        state.active_player = 0
        state.players[0].resources["brick"] = 1
        state.players[1].resources["wool"] = 1
        state.players[2].resources["wool"] = 1

        legal_trades = state.legal_domestic_trades(0)
        self.assertIn(
            {"give_resource": "brick", "receive_resource": "wool"},
            legal_trades,
        )

        state.apply_action(
            Action(
                "propose_trade",
                {"give_resource": "brick", "receive_resource": "wool"},
            )
        )
        self.assertEqual(state.pending_phase, "trade_response")
        self.assertEqual(state.pending_trade_responder, 1)

        state.apply_action(Action("decline_trade", {"player_id": 1}))
        self.assertEqual(state.pending_trade_responder, 2)

        state.apply_action(Action("accept_trade", {"player_id": 2}))

        self.assertEqual(state.players[0].resources["brick"], 0)
        self.assertEqual(state.players[0].resources["wool"], 1)
        self.assertEqual(state.players[2].resources["brick"], 1)
        self.assertEqual(state.players[2].resources["wool"], 0)
        self.assertEqual(state.pending_phase, "main")


class CatanExtensiveScenarioTest(unittest.TestCase):
    def test_setup_snake_order_and_transition_to_turn_start(self):
        """Setup uses 0-1-2-3 then 3-2-1-0 settlement order, then starts normal turns."""
        state = build_game(seed=12)
        settlement_order = []
        while state.pending_phase in {"setup_settlement", "setup_road"}:
            if state.pending_phase == "setup_settlement":
                settlement_order.append(state.active_player)
            state.apply_action(state.legal_actions()[0])
        self.assertEqual(settlement_order, [0, 1, 2, 3, 3, 2, 1, 0])
        self.assertEqual(state.pending_phase, "turn_start")
        self.assertEqual(state.active_player, 0)

    def test_roll_seven_enters_discard_and_then_robber_move(self):
        """A rolled 7 should queue only players above 7 cards for discards, then robber move."""
        state = build_game(seed=13)
        state.pending_phase = "turn_start"
        state.active_player = 0

        state.players[0].resources["brick"] = 8
        state.players[2].resources["grain"] = 9
        state.players[1].resources["wool"] = 7
        state.players[3].resources["ore"] = 2

        state.roll_dice(forced_roll=7)

        self.assertEqual(state.pending_phase, "robber_discard")
        self.assertEqual(state.pending_discard_players, [0, 2])

        state.apply_action(Action("discard_resources", {"player_id": 0, "resources": {"brick": 4}}))
        self.assertEqual(state.pending_phase, "robber_discard")
        self.assertEqual(state.pending_discard_players, [2])

        state.apply_action(Action("discard_resources", {"player_id": 2, "resources": {"grain": 4}}))
        self.assertEqual(state.pending_phase, "robber_move")
        self.assertEqual(state.pending_discard_players, [])

    def test_discard_validation_enforces_turn_order_and_exact_count(self):
        """Discard action must come from the required player and match the exact required count."""
        state = build_game(seed=14)
        state.pending_phase = "robber_discard"
        state.pending_discard_players = [1]
        state.players[1].resources["brick"] = 8

        with self.assertRaises(ActionError):
            state.apply_action(Action("discard_resources", {"player_id": 0, "resources": {"brick": 4}}))
        with self.assertRaises(ActionError):
            state.apply_action(Action("discard_resources", {"player_id": 1, "resources": {"brick": 3}}))

        state.apply_action(Action("discard_resources", {"player_id": 1, "resources": {"brick": 4}}))
        self.assertEqual(state.pending_phase, "robber_move")

    def test_robber_blocks_production_on_target_hex(self):
        """A hex with the robber should produce no resources when its number is rolled."""
        state = build_game(seed=15)
        tile = next(tile for tile in state.board.hexes if tile.token is not None and tile.resource is not None)
        node_id = tile.corner_ids[0]
        state.place_settlement(0, node_id, is_setup=True)
        state.robber_hex_id = tile.id
        state.pending_phase = "turn_start"
        state.active_player = 1

        state.roll_dice(forced_roll=tile.token)

        self.assertEqual(state.players[0].resources[tile.resource.value], 0)
        self.assertEqual(state.pending_phase, "main")

    def test_bank_shortage_distributes_all_remaining_cards_with_random_tiebreak(self):
        """If supply is short, all remaining cards are distributed across claim units with a random tie-break."""
        state = build_game(seed=16)
        tile = next(tile for tile in state.board.hexes if tile.token is not None and tile.resource is not None)
        node_a, node_b, node_c, node_d = tile.corner_ids[0], tile.corner_ids[1], tile.corner_ids[2], tile.corner_ids[3]
        resource_name = tile.resource.value

        state.place_settlement(0, node_a, is_setup=True)
        state.place_settlement(1, node_b, is_setup=True)
        state.place_settlement(2, node_c, is_setup=True)
        state.place_settlement(3, node_d, is_setup=True)
        state.robber_hex_id = next(other.id for other in state.board.hexes if other.id != tile.id)
        state.bank_resources[resource_name] = 3
        state.pending_phase = "turn_start"
        state.active_player = 2

        state.roll_dice(forced_roll=tile.token)

        awarded = sum(player.resources[resource_name] for player in state.players)
        self.assertEqual(awarded, 3)
        self.assertEqual(state.bank_resources[resource_name], 0)
        self.assertTrue(all(player.resources[resource_name] in {0, 1} for player in state.players))

    def test_bank_shortage_single_player_claim_is_limited_by_remaining_supply(self):
        """If one player has multiple claims, they receive the remaining supply up to their total claim."""
        state = build_game(seed=17)
        tile = next(tile for tile in state.board.hexes if tile.token is not None and tile.resource is not None)
        node_a, node_b = tile.corner_ids[0], tile.corner_ids[2]
        resource_name = tile.resource.value

        state.place_settlement(0, node_a, is_setup=True)
        state.place_settlement(0, node_b, is_setup=True)
        state.upgrade_city(0, node_b)
        state.robber_hex_id = next(other.id for other in state.board.hexes if other.id != tile.id)
        state.bank_resources[resource_name] = 2
        state.pending_phase = "turn_start"
        state.active_player = 1

        state.roll_dice(forced_roll=tile.token)

        self.assertEqual(state.players[0].resources[resource_name], 2)
        self.assertEqual(state.bank_resources[resource_name], 0)

    def test_buy_development_card_is_hidden_until_end_turn_reveal(self):
        """Newly bought development cards stay in new-card storage until end of that player's turn."""
        state = build_game(seed=18)
        state.pending_phase = "main"
        state.active_player = 0
        state.development_deck = ["knight"]
        state.players[0].resources["ore"] = 1
        state.players[0].resources["grain"] = 1
        state.players[0].resources["wool"] = 1

        state.apply_action(Action("buy_development_card", {}))

        self.assertEqual(state.players[0].new_development_cards["knight"], 1)
        self.assertEqual(state.players[0].development_cards["knight"], 0)

        state.end_turn()
        self.assertEqual(state.players[0].new_development_cards["knight"], 0)
        self.assertEqual(state.players[0].development_cards["knight"], 1)

    def test_played_development_card_consumes_turn_dev_play_budget(self):
        """After one dev card is played, main-phase legal actions should not include additional dev-card plays."""
        state = build_game(seed=19)
        state.pending_phase = "main"
        state.active_player = 0
        state.players[0].development_cards["knight"] = 1
        state.players[0].development_cards["monopoly"] = 1

        robber_action = next(action for action in state.legal_development_card_actions() if action.action_type == "play_knight")
        state.apply_action(robber_action)

        self.assertTrue(state.dev_card_played_this_turn)
        self.assertFalse(
            any(action.action_type.startswith("play_") for action in state.legal_main_phase_actions())
        )

    def test_playing_knight_at_three_claims_largest_army(self):
        """Playing a third knight should immediately award Largest Army if uniquely qualified."""
        state = build_game(seed=20)
        state.pending_phase = "main"
        state.active_player = 0
        state.players[0].played_knights = 2
        state.players[0].development_cards["knight"] = 1

        action = next(action for action in state.legal_development_card_actions() if action.action_type == "play_knight")
        state.apply_action(action)

        self.assertEqual(state.players[0].played_knights, 3)
        self.assertEqual(state.largest_army_owner, 0)

    def test_largest_army_tie_does_not_transfer(self):
        """Matching the current owner's knight count should not steal Largest Army."""
        state = build_game(seed=21)
        state.players[0].played_knights = 3
        state.update_largest_army()
        self.assertEqual(state.largest_army_owner, 0)

        state.players[1].played_knights = 3
        state.update_largest_army()
        self.assertEqual(state.largest_army_owner, 0)

    def test_road_building_places_two_free_roads_and_returns_to_main(self):
        """Road Building should open a free-road phase and consume up to two free placements."""
        state = build_game(seed=22)
        state.pending_phase = "main"
        state.active_player = 0
        state.rolled_this_turn = True
        state.players[0].development_cards["road_building"] = 1
        state.place_settlement(0, 0, is_setup=True)

        state.apply_action(Action("play_road_building", {}))
        self.assertEqual(state.pending_phase, "road_building")

        initial_resources = dict(state.players[0].resources)
        first = next(action for action in state.legal_actions() if action.action_type == "build_road_free")
        state.apply_action(first)
        second = next(action for action in state.legal_actions() if action.action_type == "build_road_free")
        state.apply_action(second)

        self.assertEqual(state.pending_phase, "main")
        self.assertEqual(state.players[0].roads_remaining, 13)
        self.assertEqual(state.players[0].resources, initial_resources)

    def test_road_building_allows_finishing_early_when_no_legal_roads(self):
        """If no free-road placements are legal, Road Building can be ended explicitly."""
        state = build_game(seed=23)
        state.pending_phase = "main"
        state.active_player = 0
        state.rolled_this_turn = True
        state.players[0].development_cards["road_building"] = 1

        state.apply_action(Action("play_road_building", {}))
        actions = state.legal_actions()
        self.assertIn(Action("finish_road_building", {}), actions)
        self.assertEqual([action for action in actions if action.action_type == "build_road_free"], [])

        state.apply_action(Action("finish_road_building", {}))
        self.assertEqual(state.pending_phase, "main")

    def test_player_harbor_rates_apply_generic_and_specific_discounts(self):
        """Owning harbor-adjacent structures should improve maritime rates correctly."""
        state = build_game(seed=24)
        generic = next(h for h in state.board.harbors if h.resource is None)
        specific = next(h for h in state.board.harbors if h.resource is not None)

        generic_node = state.board.edges[generic.edge_id].intersection_ids[0]
        specific_node = state.board.edges[specific.edge_id].intersection_ids[0]
        state.place_settlement(0, generic_node, is_setup=True)
        if specific_node != generic_node:
            state.place_settlement(0, specific_node, is_setup=True)

        rates = state.player_harbor_rates(0)
        self.assertTrue(all(rate <= 3 for rate in rates.values()))
        self.assertEqual(rates[specific.resource.value], 2)

    def test_opponent_harbor_does_not_grant_discount(self):
        """You only get a harbor discount from your own adjacent settlement/city, not an opponent's."""
        state = build_game(seed=241)
        harbor = next(h for h in state.board.harbors if h.resource is None)
        node_id = state.board.edges[harbor.edge_id].intersection_ids[0]
        state.place_settlement(1, node_id, is_setup=True)

        rates = state.player_harbor_rates(0)
        self.assertTrue(all(rate == 4 for rate in rates.values()))

    def test_city_produces_two_resources(self):
        """A city should produce two resources on a matching unblocked roll."""
        state = build_game(seed=242)
        tile = next(tile for tile in state.board.hexes if tile.token is not None and tile.resource is not None)
        node_id = tile.corner_ids[0]
        state.place_settlement(0, node_id, is_setup=True)
        state.upgrade_city(0, node_id)
        state.robber_hex_id = next(other.id for other in state.board.hexes if other.id != tile.id)
        state.pending_phase = "turn_start"
        state.active_player = 1

        state.roll_dice(forced_roll=tile.token)

        self.assertEqual(state.players[0].resources[tile.resource.value], 2)

    def test_maritime_trade_does_not_offer_resources_missing_from_bank(self):
        """Maritime offers must exclude receive-resources with empty bank supply."""
        state = build_game(seed=25)
        state.pending_phase = "main"
        state.active_player = 0
        state.players[0].resources["brick"] = 4
        state.bank_resources["ore"] = 0

        trades = state.legal_maritime_trades(0)
        self.assertFalse(any(trade["receive_resource"] == "ore" for trade in trades))

    def test_settlement_requires_own_road_after_setup(self):
        """Outside setup, a settlement is legal only if connected to one of your own roads."""
        state = build_game(seed=26)
        node_id = 0
        self.assertFalse(state.can_place_settlement(0, node_id, setup=False))

        edge_id = state.board.intersections[node_id].adjacent_edge_ids[0]
        state.place_road(0, edge_id)
        self.assertTrue(state.can_place_settlement(0, node_id, setup=False))

    def test_opponent_settlement_blocks_road_continuation_through_that_node(self):
        """Road continuation cannot pass through an intersection occupied by an opponent structure."""
        state = build_game(seed=27)
        node = next(node for node in state.board.intersections if len(node.adjacent_edge_ids) >= 2)
        edge_a, edge_b = node.adjacent_edge_ids[:2]

        state.place_road(0, edge_a)
        state.place_settlement(1, node.id, is_setup=False)

        self.assertFalse(state.can_place_road(0, edge_b, setup=False))

    def test_city_upgrade_changes_piece_supply_and_points(self):
        """Upgrading to a city should replace a settlement, return one settlement piece, and add 1 VP."""
        state = build_game(seed=28)
        state.place_settlement(0, 0, is_setup=True)
        state.upgrade_city(0, 0)

        self.assertNotIn(0, state.settlement_owners)
        self.assertEqual(state.city_owners[0], 0)
        self.assertEqual(state.players[0].visible_victory_points, 2)
        self.assertEqual(state.players[0].settlements_remaining, 5)
        self.assertEqual(state.players[0].cities_remaining, 3)

    def test_winner_requires_active_player_turn(self):
        """A non-active player at 10+ VP should not win until that player becomes active."""
        state = build_game(seed=29)
        state.pending_phase = "main"
        state.active_player = 0
        state.rolled_this_turn = True
        state.players[1].visible_victory_points = 10

        state.check_for_winner()
        self.assertIsNone(state.winner)

        state.active_player = 1
        state.rolled_this_turn = True
        state.check_for_winner()
        self.assertEqual(state.winner, 1)
        self.assertEqual(state.pending_phase, "game_over")

    def test_player_at_ten_does_not_win_at_start_of_turn_before_rolling(self):
        """A player starting at 10 does not win until after the required roll is resolved."""
        state = build_game(seed=30)
        state.pending_phase = "turn_start"
        state.active_player = 1
        state.players[1].visible_victory_points = 10

        state.check_for_winner()
        self.assertIsNone(state.winner)

        state.roll_dice(forced_roll=7)
        self.assertIsNone(state.winner)

        no_victim_hex = next(
            tile.id for tile in state.board.hexes
            if tile.id != state.robber_hex_id and not state.legal_robber_victims(tile.id)
        )
        state.apply_action(Action("move_robber", {"hex_id": no_victim_hex, "victim_id": None}))

        self.assertEqual(state.winner, 1)
        self.assertEqual(state.pending_phase, "game_over")

    def test_buying_victory_point_card_can_win_on_current_turn(self):
        """Victory point cards are passive and can immediately complete a win on the buyer's turn."""
        state = build_game(seed=300)
        state.pending_phase = "main"
        state.active_player = 0
        state.rolled_this_turn = True
        state.players[0].visible_victory_points = 9
        state.development_deck = ["victory_point"]
        state.players[0].resources["ore"] = 1
        state.players[0].resources["grain"] = 1
        state.players[0].resources["wool"] = 1

        state.apply_action(Action("buy_development_card", {}))

        self.assertEqual(state.winner, 0)
        self.assertEqual(state.pending_phase, "game_over")

    def test_pre_roll_development_card_can_win_before_roll(self):
        """A pre-roll dev-card play that grants Largest Army can win before the dice are rolled."""
        state = build_game(seed=301)
        state.pending_phase = "turn_start"
        state.active_player = 0
        state.players[0].visible_victory_points = 8
        state.players[0].played_knights = 2
        state.players[0].development_cards["knight"] = 1
        target_hex = next(
            tile.id
            for tile in state.board.hexes
            if tile.id != state.robber_hex_id and not state.legal_robber_victims(tile.id)
        )

        state.apply_action(Action("play_knight", {"hex_id": target_hex, "victim_id": None}))

        self.assertEqual(state.winner, 0)
        self.assertEqual(state.pending_phase, "game_over")

    def test_longest_road_requires_five_segments_to_claim(self):
        """Longest Road should not be awarded below length 5 and should award at length 5."""
        state = build_game(seed=31)
        path = find_edge_path(state.board, 5)
        for edge_id in path[:4]:
            state.place_road(0, edge_id)
        state.update_longest_road()
        self.assertIsNone(state.longest_road_owner)

        state.place_road(0, path[4])
        state.update_longest_road()
        self.assertEqual(state.longest_road_owner, 0)

    def test_robber_victims_exclude_zero_resource_players(self):
        """Only opponents with cards in hand are legal robbery victims."""
        state = build_game(seed=32)
        tile = next(tile for tile in state.board.hexes if tile.resource is not None)
        node_a, node_b = tile.corner_ids[0], tile.corner_ids[2]
        state.place_settlement(1, node_a, is_setup=True)
        state.place_settlement(2, node_b, is_setup=True)
        state.players[2].resources["wool"] = 1
        state.active_player = 0

        self.assertEqual(state.legal_robber_victims(tile.id), [2])

    def test_robber_can_move_to_desert_and_steal_if_victim_is_adjacent(self):
        """Moving the robber to desert is legal and can still steal when an adjacent victim exists."""
        state = build_game(seed=33)
        desert = next(tile for tile in state.board.hexes if tile.terrain == Terrain.DESERT)
        state.robber_hex_id = next(tile.id for tile in state.board.hexes if tile.id != desert.id)
        victim_node = desert.corner_ids[0]
        state.place_settlement(1, victim_node, is_setup=True)
        state.players[1].resources["ore"] = 1
        state.pending_phase = "robber_move"
        state.active_player = 0
        state.rolled_this_turn = True

        state.apply_action(Action("move_robber", {"hex_id": desert.id, "victim_id": 1}))

        self.assertEqual(state.robber_hex_id, desert.id)
        self.assertEqual(state.players[0].resources["ore"], 1)
        self.assertEqual(state.players[1].resources["ore"], 0)
        self.assertEqual(state.pending_phase, "main")

    def test_robber_move_requires_victim_when_robbable_players_exist(self):
        """Robber move should reject victim_id=None if at least one valid victim is adjacent."""
        state = build_game(seed=34)
        tile = next(tile for tile in state.board.hexes if tile.resource is not None)
        state.robber_hex_id = next(other.id for other in state.board.hexes if other.id != tile.id)
        state.place_settlement(1, tile.corner_ids[0], is_setup=True)
        state.players[1].resources["wool"] = 1
        state.pending_phase = "robber_move"
        state.active_player = 0

        with self.assertRaises(ActionError):
            state.apply_action(Action("move_robber", {"hex_id": tile.id, "victim_id": None}))

    def test_knight_after_rolled_seven_can_move_robber_again(self):
        """After resolving a rolled 7 robber move, a played Knight can move the robber a second time."""
        state = build_game(seed=35)
        state.pending_phase = "turn_start"
        state.active_player = 0
        state.players[0].development_cards["knight"] = 1

        state.roll_dice(forced_roll=7)
        first_destination = state.legal_robber_hex_ids()[0]
        state.apply_action(Action("move_robber", {"hex_id": first_destination, "victim_id": None}))
        self.assertEqual(state.pending_phase, "main")

        knight_action = next(
            action for action in state.legal_main_phase_actions() if action.action_type == "play_knight"
        )
        state.apply_action(knight_action)

        self.assertNotEqual(state.robber_hex_id, first_destination)
        self.assertEqual(state.players[0].played_knights, 1)
        self.assertEqual(state.players[0].development_cards["knight"], 0)
        self.assertEqual(state.pending_phase, "main")

    def test_knight_pre_roll_does_not_trigger_discard_flow(self):
        """Playing Knight before rolling should not force robber discard handling for large hands."""
        state = build_game(seed=36)
        state.pending_phase = "turn_start"
        state.active_player = 0
        state.players[0].development_cards["knight"] = 1
        state.players[1].resources["brick"] = 10
        target_hex = state.legal_robber_hex_ids()[0]

        state.apply_action(Action("play_knight", {"hex_id": target_hex, "victim_id": None}))

        self.assertEqual(state.pending_discard_players, [])
        self.assertEqual(state.pending_phase, "turn_start")
        self.assertTrue(state.dev_card_played_this_turn)

    def test_newly_bought_dev_card_is_not_playable_same_turn(self):
        """A development card bought this turn should not appear in same-turn play actions."""
        state = build_game(seed=37)
        state.pending_phase = "main"
        state.active_player = 0
        state.rolled_this_turn = True
        state.development_deck = ["knight"]
        state.players[0].resources["ore"] = 1
        state.players[0].resources["grain"] = 1
        state.players[0].resources["wool"] = 1

        state.apply_action(Action("buy_development_card", {}))

        self.assertFalse(
            any(action.action_type == "play_knight" for action in state.legal_main_phase_actions())
        )

    def test_two_victory_point_cards_bought_same_turn_can_complete_win(self):
        """Multiple bought VP cards in one turn should stack and trigger a win at 10+ total VP."""
        state = build_game(seed=38)
        state.pending_phase = "main"
        state.active_player = 0
        state.rolled_this_turn = True
        state.players[0].visible_victory_points = 8
        state.development_deck = ["victory_point", "victory_point"]
        state.players[0].resources["ore"] = 2
        state.players[0].resources["grain"] = 2
        state.players[0].resources["wool"] = 2

        state.apply_action(Action("buy_development_card", {}))
        self.assertIsNone(state.winner)
        state.apply_action(Action("buy_development_card", {}))

        self.assertEqual(state.winner, 0)
        self.assertEqual(state.pending_phase, "game_over")

    def test_build_onto_harbor_allows_immediate_same_turn_harbor_trade(self):
        """In combined main phase, a newly built harbor settlement grants immediate maritime discount."""
        state = build_game(seed=39)
        generic_harbor = next(harbor for harbor in state.board.harbors if harbor.resource is None)
        harbor_node = state.board.edges[generic_harbor.edge_id].intersection_ids[0]
        road_edge = state.board.intersections[harbor_node].adjacent_edge_ids[0]

        state.pending_phase = "main"
        state.active_player = 0
        state.rolled_this_turn = True
        state.place_road(0, road_edge)
        state.players[0].resources["brick"] = 1
        state.players[0].resources["lumber"] = 1
        state.players[0].resources["grain"] = 1
        state.players[0].resources["wool"] = 4

        state.apply_action(Action("build_settlement", {"intersection_id": harbor_node}))

        legal_trades = state.legal_maritime_trades(0)
        harbor_trade = next(
            trade for trade in legal_trades
            if trade["give_resource"] == "wool" and trade["receive_resource"] == "ore"
        )
        self.assertEqual(harbor_trade["rate"], 3)
        state.apply_action(Action("trade_maritime", harbor_trade))

        self.assertEqual(state.players[0].resources["wool"], 0)
        self.assertEqual(state.players[0].resources["ore"], 1)

    def test_trade_offer_with_no_possible_responders_is_rejected(self):
        """Proposing a trade should fail when no opponent has the requested resource."""
        state = build_game(seed=40)
        state.pending_phase = "main"
        state.active_player = 0
        state.rolled_this_turn = True
        state.players[0].resources["brick"] = 1

        with self.assertRaises(ActionError):
            state.apply_action(Action("propose_trade", {"give_resource": "brick", "receive_resource": "wool"}))

    def test_trade_response_declined_by_all_returns_to_main_and_clears_offer(self):
        """If every responder declines, trade state resets and phase returns to main."""
        state = build_game(seed=41)
        state.pending_phase = "main"
        state.active_player = 0
        state.rolled_this_turn = True
        state.players[0].resources["brick"] = 1
        state.players[1].resources["wool"] = 1
        state.players[2].resources["wool"] = 1
        state.players[3].resources["wool"] = 1

        state.apply_action(Action("propose_trade", {"give_resource": "brick", "receive_resource": "wool"}))
        state.apply_action(Action("decline_trade", {"player_id": 1}))
        state.apply_action(Action("decline_trade", {"player_id": 2}))
        state.apply_action(Action("decline_trade", {"player_id": 3}))

        self.assertEqual(state.pending_phase, "main")
        self.assertIsNone(state.pending_trade_offer)
        self.assertIsNone(state.pending_trade_responder)
        self.assertEqual(state.pending_trade_responders, [])

    def test_trade_accept_fails_if_offerer_loses_give_resource_before_later_response(self):
        """A later responder cannot accept if the offerer no longer holds the offered resource."""
        state = build_game(seed=42)
        state.pending_phase = "main"
        state.active_player = 0
        state.rolled_this_turn = True
        state.players[0].resources["brick"] = 1
        state.players[1].resources["wool"] = 1
        state.players[2].resources["wool"] = 1

        state.apply_action(Action("propose_trade", {"give_resource": "brick", "receive_resource": "wool"}))
        state.apply_action(Action("decline_trade", {"player_id": 1}))
        state.players[0].resources["brick"] = 0

        with self.assertRaises(ActionError):
            state.apply_action(Action("accept_trade", {"player_id": 2}))
        self.assertEqual(state.pending_phase, "trade_response")
        self.assertEqual(state.pending_trade_responder, 2)

    def test_trade_accept_fails_if_partner_loses_requested_resource_before_accept(self):
        """Acceptance should fail if the current responder no longer has the requested resource."""
        state = build_game(seed=43)
        state.pending_phase = "main"
        state.active_player = 0
        state.rolled_this_turn = True
        state.players[0].resources["brick"] = 1
        state.players[1].resources["wool"] = 1

        state.apply_action(Action("propose_trade", {"give_resource": "brick", "receive_resource": "wool"}))
        state.players[1].resources["wool"] = 0

        with self.assertRaises(ActionError):
            state.apply_action(Action("accept_trade", {"player_id": 1}))
        self.assertEqual(state.pending_phase, "trade_response")

    def test_road_building_pre_roll_finishes_back_in_turn_start(self):
        """When Road Building is played pre-roll, finishing it should return phase to turn_start."""
        state = build_game(seed=44)
        state.pending_phase = "turn_start"
        state.active_player = 0
        state.players[0].development_cards["road_building"] = 1

        state.apply_action(Action("play_road_building", {}))
        self.assertEqual(state.pending_phase, "road_building")
        state.apply_action(Action("finish_road_building", {}))
        self.assertEqual(state.pending_phase, "turn_start")


if __name__ == "__main__":
    unittest.main()
