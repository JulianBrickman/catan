import unittest

from catan_rl_env import (
    CatanEnvConfig,
    CatanRLEnv,
    OBSERVATION_SCHEMA_VERSION,
    TENSOR_SCHEMA_VERSION,
)


class CatanRLEnvTest(unittest.TestCase):
    def test_reset_returns_versioned_observation_schema(self):
        env = CatanRLEnv(CatanEnvConfig(seed=101))
        obs = env.reset()

        self.assertEqual(obs["schema_version"], OBSERVATION_SCHEMA_VERSION)
        self.assertEqual(obs["tensor_schema_version"], TENSOR_SCHEMA_VERSION)
        self.assertEqual(obs["observer_player_id"], env.state.active_player)
        self.assertEqual(len(obs["hexes"]), 19)
        self.assertEqual(len(obs["intersections"]), 54)
        self.assertEqual(len(obs["edges"]), 72)
        self.assertEqual(len(obs["players"]), 4)
        self.assertEqual(len(obs["legal_actions"]), len(obs["legal_action_mask"]))
        self.assertEqual(len(obs["legal_actions"]), len(obs["action_features"]))
        self.assertIn("tensor_spec", obs)
        self.assertIn("tensor_obs", obs)
        self.assertIn("action_tensor_spec", obs)

    def test_tensor_spec_matches_tensorized_shapes(self):
        env = CatanRLEnv(CatanEnvConfig(seed=106))
        obs = env.reset()
        spec = obs["tensor_spec"]
        tensor_obs = obs["tensor_obs"]

        self.assertEqual(spec["global_features"]["size"], len(tensor_obs["global_features"]))
        self.assertEqual(spec["hex_features"]["shape"][0], len(tensor_obs["hex_features"]))
        self.assertEqual(spec["hex_features"]["shape"][1], len(tensor_obs["hex_features"][0]))
        self.assertEqual(spec["intersection_features"]["shape"][0], len(tensor_obs["intersection_features"]))
        self.assertEqual(spec["intersection_features"]["shape"][1], len(tensor_obs["intersection_features"][0]))
        self.assertEqual(spec["edge_features"]["shape"][0], len(tensor_obs["edge_features"]))
        self.assertEqual(spec["edge_features"]["shape"][1], len(tensor_obs["edge_features"][0]))
        self.assertEqual(spec["player_features"]["shape"][0], len(tensor_obs["player_features"]))
        self.assertEqual(spec["player_features"]["shape"][1], len(tensor_obs["player_features"][0]))
        self.assertEqual(spec["private_features"]["size"], len(tensor_obs["private_features"]))

    def test_tensorized_private_features_hide_other_players_hands(self):
        env = CatanRLEnv(CatanEnvConfig(seed=107))
        obs = env.reset()

        self.assertEqual(len(obs["private"]["resources"]), 5)
        self.assertEqual(len(obs["tensor_obs"]["private_features"]), obs["tensor_spec"]["private_features"]["size"])
        self.assertNotIn("resources", obs["players"][1])

    def test_tensorized_global_features_include_phase_one_hot(self):
        env = CatanRLEnv(CatanEnvConfig(seed=108))
        obs = env.reset()
        phase_one_hot = obs["tensor_obs"]["global_features"][-9:]

        self.assertEqual(sum(phase_one_hot), 1.0)

    def test_action_feature_rows_match_action_tensor_spec(self):
        env = CatanRLEnv(CatanEnvConfig(seed=109))
        obs = env.reset()

        self.assertEqual(
            len(obs["action_features"][0]),
            obs["action_tensor_spec"]["action_feature_size"],
        )

    def test_domestic_trades_are_filtered_when_disabled(self):
        env = CatanRLEnv(CatanEnvConfig(seed=102, allow_domestic_trade=False))
        env.reset()
        env.state.pending_phase = "main"
        env.state.active_player = 0
        env.state.players[0].resources["brick"] = 1
        env.state.players[1].resources["wool"] = 1

        actions = env.legal_actions()

        self.assertFalse(any(action.action_type == "propose_trade" for action in actions))

    def test_step_uses_phase_local_action_index(self):
        env = CatanRLEnv(CatanEnvConfig(seed=103))
        env.reset()

        obs, reward, done, info = env.step(0)

        self.assertIn("global", obs)
        self.assertIn("reward_components", info)
        self.assertFalse(done)
        self.assertIsInstance(reward, float)

    def test_terminal_rewards_include_rank_focused_scores(self):
        env = CatanRLEnv(CatanEnvConfig(seed=104))
        env.reset()
        env.state.pending_phase = "main"
        env.state.active_player = 0
        env.state.rolled_this_turn = True
        env.state.players[0].visible_victory_points = 9
        env.state.players[0].resources["brick"] = 1
        env.state.players[0].resources["lumber"] = 1
        env.state.players[0].resources["wool"] = 1
        env.state.players[0].resources["grain"] = 1
        start_node = 0
        start_edge = env.state.board.intersections[start_node].adjacent_edge_ids[0]
        env.state.place_settlement(0, start_node, is_setup=True)
        env.state.place_road(0, start_edge)
        far_node = next(
            node_id
            for node_id in env.state.board.edges[start_edge].intersection_ids
            if node_id != start_node
        )
        second_edge = next(
            edge_id
            for edge_id in env.state.board.intersections[far_node].adjacent_edge_ids
            if edge_id != start_edge
        )
        env.state.place_road(0, second_edge)
        legal_target = env.state.legal_settlement_ids(0)[0]

        actions = env.legal_actions()
        action_index = next(
            index
            for index, action in enumerate(actions)
            if action.action_type == "build_settlement"
            and action.params["intersection_id"] == legal_target
        )

        _, reward, done, info = env.step(action_index)

        self.assertTrue(done)
        self.assertEqual(env.state.winner, 0)
        self.assertEqual(info["placements"][0], 1)
        self.assertEqual(info["terminal_reward_components"][0]["placement_score"], 1.0)
        self.assertGreaterEqual(reward, 1.0)

    def test_terminal_rewards_support_truncation_without_winner(self):
        env = CatanRLEnv(CatanEnvConfig(seed=105, max_turns=0))
        env.reset()

        _, _, done, info = env.step(0)

        self.assertTrue(done)
        self.assertTrue(info["truncated"])
        self.assertIn("terminal_rewards_by_player", info)


if __name__ == "__main__":
    unittest.main()
