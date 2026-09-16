import importlib.util
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class AgentReplayTests(unittest.TestCase):
    def test_ddpg_update_and_checkpoint_roundtrip(self):
        import torch

        from breadboard_reach.agent import DDPGAgent
        from breadboard_reach.config import TrainConfig

        cfg = TrainConfig(total_steps=1, warmup_steps=0, batch_size=4)
        agent = DDPGAgent(cfg)
        observation = {
            "observation": torch.zeros((4, 28)),
            "desired_goal": torch.tensor([[0.4, 0, 0.3, 0, 1, 0, 0]] * 4),
        }
        batch = {
            "curr_obs": observation,
            "next_obs": observation,
            "actions": torch.zeros((4, 1, 7)),
            "rewards": torch.tensor([[-1], [-1], [0], [-6]]),
            "terminations": torch.tensor([[0], [0], [0], [1]]),
            "truncations": torch.tensor([[0], [1], [0], [0]]),
        }
        result = agent.update(batch)
        self.assertEqual(agent.updates, 1)
        self.assertTrue(all(np.isfinite(value) for value in result.values()))
        restored = DDPGAgent(cfg)
        restored.load_state_dict(agent.state_dict())
        self.assertEqual(restored.updates, 1)
        single = {
            "observation": np.zeros(28, dtype=np.float32),
            "desired_goal": np.array([0.4, 0, 0.3, 0, 1, 0, 0], dtype=np.float32),
        }
        np.testing.assert_allclose(agent.act(single), restored.act(single), atol=1e-7)

    def test_rlinf_trajectory_packing(self):
        from breadboard_reach.her import Transition
        from breadboard_reach.replay import to_rlinf_trajectory

        class FakeTrajectory:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        obs = {
            "observation": np.zeros(28, dtype=np.float32),
            "achieved_goal": np.array([0, 0, 0.3, 1, 0, 0, 0], dtype=np.float32),
            "desired_goal": np.array([0.4, 0, 0.3, 1, 0, 0, 0], dtype=np.float32),
        }
        transition = Transition(
            observation=obs,
            action=np.zeros(7, dtype=np.float32),
            reward=-1,
            next_observation=obs,
            terminated=False,
            truncated=True,
            collision=False,
            episode_id=0,
            step_index=0,
        )
        with patch("breadboard_reach.replay._rlinf_types", return_value=(FakeTrajectory, object)):
            packed = to_rlinf_trajectory([transition], 100)
        self.assertEqual(tuple(packed.actions.shape), (1, 1, 1, 7))
        self.assertEqual(tuple(packed.curr_obs["observation"].shape), (1, 1, 28))
        self.assertEqual(tuple(packed.rewards.shape), (1, 1, 1))
        self.assertTrue(bool(packed.truncations[0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
