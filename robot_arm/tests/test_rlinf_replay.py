import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("rlinf"), "RLinf is not installed")
class RlinfReplayTests(unittest.TestCase):
    def test_episode_relabel_sample_and_checkpoint(self):
        from breadboard_reach.config import TaskConfig
        from breadboard_reach.geometry import compute_reward
        from breadboard_reach.her import Transition
        from breadboard_reach.replay import RlinfHERReplay

        cfg = TaskConfig(max_episode_steps=2, hold_steps=1)

        def obs(x):
            return {
                "observation": np.zeros(28, dtype=np.float32),
                "achieved_goal": np.array([x, 0, 0.3, 1, 0, 0, 0], dtype=np.float32),
                "desired_goal": np.array([0.4, 0, 0.3, 1, 0, 0, 0], dtype=np.float32),
            }

        episode = [
            Transition(obs(0), np.zeros(7, dtype=np.float32), -1, obs(0.01), False, False, False, 0, 0),
            Transition(obs(0.01), np.zeros(7, dtype=np.float32), -1, obs(0.02), False, True, False, 0, 1),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            replay = RlinfHERReplay(
                root / "working", seed=1, window_episodes=10, cache_episodes=2, max_episode_steps=2
            )
            try:
                count = replay.add_episode(
                    episode,
                    future_k=4,
                    rng=np.random.default_rng(1),
                    reward_fn=lambda a, d, info: compute_reward(a, d, info, cfg),
                )
                self.assertEqual(count, 10)
                self.assertEqual(replay.total_samples, 10)
                batch = replay.sample(4)
                self.assertEqual(batch["curr_obs"]["desired_goal"].shape[-1], 7)
                self.assertEqual(batch["actions"].shape[-1], 7)
                replay.save_checkpoint(root / "snapshot")
            finally:
                replay.close()

            restored = RlinfHERReplay(
                root / "continued", seed=1, window_episodes=10, cache_episodes=2, max_episode_steps=2
            )
            try:
                restored.load_checkpoint(root / "snapshot")
                self.assertEqual(restored.total_samples, 10)
                self.assertEqual(restored.sample(4)["rewards"].shape[0], 4)
            finally:
                restored.close()


if __name__ == "__main__":
    unittest.main()
