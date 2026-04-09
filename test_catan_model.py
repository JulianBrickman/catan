import unittest

import torch

from catan_model import (
    CatanPolicyValueNet,
    PolicyHeadConfig,
    SharedCatanEncoder,
    SharedEncoderConfig,
    action_features_to_torch,
    tensor_obs_to_torch,
)
from catan_rl_env import CatanEnvConfig, CatanRLEnv
from catan_rollout import CatanRolloutCollector


class SharedCatanEncoderTest(unittest.TestCase):
    def test_tensor_obs_to_torch_adds_batch_dimension(self):
        env = CatanRLEnv(CatanEnvConfig(seed=201))
        obs = env.reset()

        tensors = tensor_obs_to_torch(obs["tensor_obs"])

        self.assertEqual(tuple(tensors["global_features"].shape), (1, 21))
        self.assertEqual(tuple(tensors["hex_features"].shape), (1, 19, 20))
        self.assertEqual(tuple(tensors["intersection_features"].shape), (1, 54, 16))
        self.assertEqual(tuple(tensors["edge_features"].shape), (1, 72, 9))
        self.assertEqual(tuple(tensors["player_features"].shape), (1, 4, 11))
        self.assertEqual(tuple(tensors["private_features"].shape), (1, 15))

    def test_action_features_to_torch_adds_batch_dimension(self):
        env = CatanRLEnv(CatanEnvConfig(seed=202))
        obs = env.reset()

        action_tensor = action_features_to_torch(obs["action_features"])

        self.assertEqual(action_tensor.ndim, 3)
        self.assertEqual(action_tensor.shape[0], 1)
        self.assertEqual(action_tensor.shape[1], len(obs["legal_actions"]))
        self.assertEqual(action_tensor.shape[2], obs["action_tensor_spec"]["action_feature_size"])

    def test_shared_encoder_forward_returns_latent(self):
        env = CatanRLEnv(CatanEnvConfig(seed=203))
        obs = env.reset()
        tensors = tensor_obs_to_torch(obs["tensor_obs"])
        encoder = SharedCatanEncoder(obs["tensor_spec"], SharedEncoderConfig(latent_dim=128))

        outputs = encoder(tensors)

        self.assertEqual(tuple(outputs["latent"].shape), (1, 128))
        self.assertEqual(tuple(outputs["global_embedding"].shape), (1, 64))
        self.assertEqual(tuple(outputs["hex_embedding"].shape), (1, 64))

    def test_shared_encoder_accepts_batched_inputs(self):
        env_a = CatanRLEnv(CatanEnvConfig(seed=204))
        env_b = CatanRLEnv(CatanEnvConfig(seed=205))
        obs_a = env_a.reset()
        obs_b = env_b.reset()
        tensors_a = tensor_obs_to_torch(obs_a["tensor_obs"])
        tensors_b = tensor_obs_to_torch(obs_b["tensor_obs"])
        batched = {
            key: torch.cat([tensors_a[key], tensors_b[key]], dim=0)
            for key in tensors_a
        }
        encoder = SharedCatanEncoder(obs_a["tensor_spec"])

        outputs = encoder(batched)

        self.assertEqual(tuple(outputs["latent"].shape), (2, 256))
        self.assertEqual(tuple(outputs["player_embedding"].shape), (2, 64))

    def test_shared_encoder_requires_all_tensor_keys(self):
        env = CatanRLEnv(CatanEnvConfig(seed=206))
        obs = env.reset()
        tensors = tensor_obs_to_torch(obs["tensor_obs"])
        del tensors["edge_features"]
        encoder = SharedCatanEncoder(obs["tensor_spec"])

        with self.assertRaises(KeyError):
            encoder(tensors)


class CatanPolicyValueNetTest(unittest.TestCase):
    def test_policy_value_net_returns_logits_and_value(self):
        env = CatanRLEnv(CatanEnvConfig(seed=207))
        obs = env.reset()
        tensor_obs = tensor_obs_to_torch(obs["tensor_obs"])
        action_features = action_features_to_torch(obs["action_features"])
        action_mask = torch.tensor([obs["legal_action_mask"]], dtype=torch.float32)
        model = CatanPolicyValueNet(
            obs["tensor_spec"],
            action_feature_dim=obs["action_tensor_spec"]["action_feature_size"],
            encoder_config=SharedEncoderConfig(latent_dim=128),
            head_config=PolicyHeadConfig(action_hidden_dim=64, value_hidden_dim=64),
        )

        outputs = model(tensor_obs, action_features, phase=obs["global"]["phase"], action_mask=action_mask)

        self.assertEqual(tuple(outputs["policy_logits"].shape), (1, len(obs["legal_actions"])))
        self.assertEqual(tuple(outputs["value"].shape), (1,))
        self.assertEqual(tuple(outputs["latent"].shape), (1, 128))

    def test_sample_action_returns_legal_phase_local_index(self):
        env = CatanRLEnv(CatanEnvConfig(seed=208))
        obs = env.reset()
        tensor_obs = tensor_obs_to_torch(obs["tensor_obs"])
        action_features = action_features_to_torch(obs["action_features"])
        action_mask = torch.tensor([obs["legal_action_mask"]], dtype=torch.float32)
        model = CatanPolicyValueNet(
            obs["tensor_spec"],
            action_feature_dim=obs["action_tensor_spec"]["action_feature_size"],
        )

        sample = model.sample_action(
            tensor_obs,
            action_features,
            phase=obs["global"]["phase"],
            action_mask=action_mask,
            deterministic=True,
        )

        action_index = int(sample["action_index"].item())
        self.assertGreaterEqual(action_index, 0)
        self.assertLess(action_index, len(obs["legal_actions"]))


class CatanRolloutCollectorTest(unittest.TestCase):
    def test_rollout_collector_runs_deterministic_episode(self):
        env = CatanRLEnv(CatanEnvConfig(seed=209, max_turns=3))
        obs = env.reset()
        model = CatanPolicyValueNet(
            obs["tensor_spec"],
            action_feature_dim=obs["action_tensor_spec"]["action_feature_size"],
        )
        collector = CatanRolloutCollector(model, CatanEnvConfig(seed=209, max_turns=3))

        episode = collector.run_episode(seed=209, deterministic=True)

        self.assertGreater(len(episode.steps), 0)
        self.assertIsNotNone(episode.placements)
        self.assertTrue(all(step.legal_action_count > 0 for step in episode.steps))


if __name__ == "__main__":
    unittest.main()
