import importlib.util
import unittest

import numpy as np

from breadboard_reach.config import TaskConfig
from breadboard_reach.geometry import target_pose


@unittest.skipUnless(
    importlib.util.find_spec("gymnasium") and importlib.util.find_spec("mujoco") and importlib.util.find_spec("mujoco_menagerie"),
    "MuJoCo task dependencies are not installed",
)
class EnvironmentTests(unittest.TestCase):
    def test_reset_action_bounds_and_goal_reward(self):
        from breadboard_reach.env import BreadboardReachEnv

        cfg = TaskConfig(reset_joint_noise=0.0)
        env = BreadboardReachEnv(cfg)
        try:
            hole = env.available_holes[0]
            observation, info = env.reset(seed=1, options={"hole_id": hole.hole_id})
            self.assertEqual(info["hole_id"], hole.hole_id)
            np.testing.assert_allclose(observation["desired_goal"], target_pose(cfg, hole))
            self.assertTrue(env.observation_space.contains(observation))
            self.assertEqual(float(env.compute_reward(env.target, env.target, {})), 0.0)
            for _ in range(5):
                observation, reward, terminated, truncated, _ = env.step(np.ones(7, dtype=np.float32))
                self.assertTrue(env.observation_space.contains(observation))
                self.assertTrue(np.isfinite(reward))
                positions = env.data.qpos[env._qpos_addr]
                self.assertTrue(np.all(positions >= env._joint_low - 1e-4))
                self.assertTrue(np.all(positions <= env._joint_high + 1e-4))
                if terminated or truncated:
                    break
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
