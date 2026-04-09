import tempfile
import unittest
from pathlib import Path

import torch

from catan_model import CatanPolicyValueNet
from catan_ppo import (
    CatanPPOTrainer,
    PPOConfig,
    collate_step_records,
    compute_gae,
    explained_variance_from_batch,
)
from catan_rl_env import CatanEnvConfig, CatanRLEnv
from train_catan_ppo import build_arg_parser, build_reward_config, save_episode_artifacts


class CatanPPOUtilityTest(unittest.TestCase):
    def test_compute_gae_matches_simple_terminal_case(self):
        advantages, returns = compute_gae(
            rewards=[1.0, 2.0],
            values=[0.0, 0.0],
            dones=[False, True],
            gamma=1.0,
            gae_lambda=1.0,
        )

        self.assertEqual(returns, [3.0, 2.0])
        self.assertEqual(advantages, [3.0, 2.0])

    def test_explained_variance_is_one_for_perfect_prediction(self):
        targets = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        predictions = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        value = explained_variance_from_batch(targets, predictions)

        self.assertAlmostEqual(value, 1.0, places=6)

    def test_collate_step_records_pads_variable_action_counts(self):
        action_dim = CatanRLEnv(CatanEnvConfig(seed=300)).reset()["action_tensor_spec"]["action_feature_size"]
        records = [
            {
                "tensor_obs": {
                    "global_features": [0.0] * 21,
                    "hex_features": [[0.0] * 20 for _ in range(19)],
                    "intersection_features": [[0.0] * 16 for _ in range(54)],
                    "edge_features": [[0.0] * 9 for _ in range(72)],
                    "player_features": [[0.0] * 11 for _ in range(4)],
                    "private_features": [0.0] * 15,
                },
                "action_features": [[0.1] * action_dim, [0.2] * action_dim],
                "action_mask": [1, 1],
                "action_index": 1,
                "old_log_prob": -0.1,
                "advantage": 0.5,
                "return": 1.0,
                "value_target": 1.0,
                "phase": "main",
            },
            {
                "tensor_obs": {
                    "global_features": [0.0] * 21,
                    "hex_features": [[0.0] * 20 for _ in range(19)],
                    "intersection_features": [[0.0] * 16 for _ in range(54)],
                    "edge_features": [[0.0] * 9 for _ in range(72)],
                    "player_features": [[0.0] * 11 for _ in range(4)],
                    "private_features": [0.0] * 15,
                },
                "action_features": [[0.3] * action_dim],
                "action_mask": [1],
                "action_index": 0,
                "old_log_prob": -0.2,
                "advantage": -0.5,
                "return": 0.0,
                "value_target": 0.0,
                "phase": "turn_start",
            },
        ]

        batch = collate_step_records(records, device="cpu")

        self.assertEqual(tuple(batch["action_features"].shape), (2, 2, action_dim))
        self.assertEqual(tuple(batch["action_mask"].shape), (2, 2))
        self.assertEqual(float(batch["action_mask"][1, 1].item()), 0.0)

    def test_cli_reward_flags_override_reward_config(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--reward-placement-score",
                "1.0",
                "--reward-vp-gain",
                "0.05",
                "--reward-final-vp",
                "0.01",
                "--reward-win",
                "0.25",
            ]
        )

        reward_config = build_reward_config(args)

        self.assertEqual(reward_config.weights["placement_score"], 1.0)
        self.assertEqual(reward_config.weights["vp_gain"], 0.05)
        self.assertEqual(reward_config.weights["final_vp"], 0.01)
        self.assertEqual(reward_config.weights["win"], 0.25)


class CatanPPOTrainerTest(unittest.TestCase):
    def test_trainer_collects_and_updates(self):
        env = CatanRLEnv(CatanEnvConfig(seed=301, max_turns=3))
        obs = env.reset()
        model = CatanPolicyValueNet(
            obs["tensor_spec"],
            action_feature_dim=obs["action_tensor_spec"]["action_feature_size"],
        )
        trainer = CatanPPOTrainer(
            model,
            env_config=CatanEnvConfig(seed=301, max_turns=3),
            ppo_config=PPOConfig(
                rollout_episodes=2,
                update_epochs=1,
                minibatch_size=8,
                device="cpu",
            ),
        )

        episodes = trainer.collect_episodes(num_episodes=2, seed=301)
        metrics = trainer.train_update(episodes)

        self.assertEqual(metrics.episodes_collected, 2)
        self.assertGreater(metrics.steps_collected, 0)
        self.assertIsInstance(metrics.policy_loss, float)
        self.assertIsInstance(metrics.explained_variance, float)
        self.assertIsInstance(metrics.truncation_count, int)
        self.assertIsInstance(metrics.average_final_vp, float)

    def test_trainer_runs_deterministic_evaluation(self):
        env = CatanRLEnv(CatanEnvConfig(seed=303, max_turns=3))
        obs = env.reset()
        model = CatanPolicyValueNet(
            obs["tensor_spec"],
            action_feature_dim=obs["action_tensor_spec"]["action_feature_size"],
        )
        trainer = CatanPPOTrainer(
            model,
            env_config=CatanEnvConfig(seed=303, max_turns=3),
            ppo_config=PPOConfig(device="cpu"),
        )

        metrics = trainer.evaluate(num_episodes=2, seed=303, deterministic=True)

        self.assertEqual(metrics.episodes_evaluated, 2)
        self.assertGreater(metrics.steps_evaluated, 0)
        self.assertIsInstance(metrics.average_first_place_rate, float)
        self.assertIsInstance(metrics.truncation_count, int)
        self.assertIsInstance(metrics.average_final_vp, float)

    def test_episode_artifacts_include_player_breakdown(self):
        env = CatanRLEnv(CatanEnvConfig(seed=305, max_turns=2))
        obs = env.reset()
        model = CatanPolicyValueNet(
            obs["tensor_spec"],
            action_feature_dim=obs["action_tensor_spec"]["action_feature_size"],
        )
        trainer = CatanPPOTrainer(
            model,
            env_config=CatanEnvConfig(seed=305, max_turns=2),
            ppo_config=PPOConfig(device="cpu"),
        )
        episode = trainer.collect_episodes(num_episodes=1, seed=305, deterministic=True)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_episode_artifacts(Path(tmpdir), "eval_final_episode", episode)
            player_json = Path(tmpdir) / "eval_final_episode" / "player_stats.json"
            player_md = Path(tmpdir) / "eval_final_episode" / "player_breakdown.md"

            self.assertTrue(player_json.exists())
            self.assertTrue(player_md.exists())
            self.assertIn("Final Player Breakdown", player_md.read_text(encoding="utf-8"))

    def test_trainer_saves_checkpoint(self):
        env = CatanRLEnv(CatanEnvConfig(seed=302, max_turns=2))
        obs = env.reset()
        model = CatanPolicyValueNet(
            obs["tensor_spec"],
            action_feature_dim=obs["action_tensor_spec"]["action_feature_size"],
        )
        trainer = CatanPPOTrainer(
            model,
            env_config=CatanEnvConfig(seed=302, max_turns=2),
            ppo_config=PPOConfig(device="cpu"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = trainer.save_checkpoint(Path(tmpdir) / "checkpoint.pt")
            payload = torch.load(checkpoint_path, map_location="cpu")

        self.assertIn("model_state_dict", payload)
        self.assertIn("optimizer_state_dict", payload)
        self.assertIn("ppo_config", payload)

    def test_trainer_loads_checkpoint_and_restores_update_step(self):
        env = CatanRLEnv(CatanEnvConfig(seed=304, max_turns=2))
        obs = env.reset()
        model = CatanPolicyValueNet(
            obs["tensor_spec"],
            action_feature_dim=obs["action_tensor_spec"]["action_feature_size"],
        )
        trainer = CatanPPOTrainer(
            model,
            env_config=CatanEnvConfig(seed=304, max_turns=2),
            ppo_config=PPOConfig(device="cpu"),
        )
        trainer.update_step = 7

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = trainer.save_checkpoint(Path(tmpdir) / "checkpoint.pt")

            reloaded_model = CatanPolicyValueNet(
                obs["tensor_spec"],
                action_feature_dim=obs["action_tensor_spec"]["action_feature_size"],
            )
            reloaded_trainer = CatanPPOTrainer(
                reloaded_model,
                env_config=CatanEnvConfig(seed=304, max_turns=2),
                ppo_config=PPOConfig(device="cpu"),
            )
            payload = reloaded_trainer.load_checkpoint(checkpoint_path)

        self.assertEqual(reloaded_trainer.update_step, 7)
        self.assertEqual(payload["update_step"], 7)


if __name__ == "__main__":
    unittest.main()
