import unittest

from catan import Action, Terrain, build_game


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
        self.assertEqual(state.pending_phase, "main")

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

    def test_domestic_trade_swaps_resources_between_active_player_and_partner(self):
        state = build_game(seed=8)
        state.pending_phase = "main"
        state.active_player = 0
        state.players[0].resources["brick"] = 1
        state.players[1].resources["wool"] = 1

        legal_trades = state.legal_domestic_trades(0)
        self.assertIn(
            {"partner_id": 1, "give_resource": "brick", "receive_resource": "wool"},
            legal_trades,
        )

        state.apply_action(
            Action(
                "trade_domestic",
                {"partner_id": 1, "give_resource": "brick", "receive_resource": "wool"},
            )
        )

        self.assertEqual(state.players[0].resources["brick"], 0)
        self.assertEqual(state.players[0].resources["wool"], 1)
        self.assertEqual(state.players[1].resources["brick"], 1)
        self.assertEqual(state.players[1].resources["wool"], 0)


if __name__ == "__main__":
    unittest.main()
