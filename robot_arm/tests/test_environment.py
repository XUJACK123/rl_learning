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
            self.assertEqual(float(env.compute_reward(env.target, env.target, {})), cfg.success_bonus)
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

    def test_zero_action_holds_pose_and_shield_prevents_descent(self):
        from breadboard_reach.diagnostics import _jacobian_action, generate_demonstration
        from breadboard_reach.env import BreadboardReachEnv

        env = BreadboardReachEnv(TaskConfig(reset_joint_noise=0.0), split="all")
        try:
            initial, _ = env.reset(seed=1, options={"hole_id": "R+0C+0"})
            start_tcp = initial["achieved_goal"][:3].copy()
            for _ in range(100):
                observation, _, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
                self.assertFalse(terminated)
                self.assertFalse(info["collision"])
            self.assertTrue(truncated)
            self.assertLess(np.linalg.norm(observation["achieved_goal"][:3] - start_tcp), 0.01)

            _, summary = generate_demonstration(env, "R+0C+0", seed=1)
            self.assertTrue(summary["success"])
            unsafe_goal = env.target
            unsafe_goal[2] -= 0.08
            interventions = 0
            for _ in range(10):
                proposed = _jacobian_action(env, unsafe_goal)
                _, _, terminated, _, info = env.step(proposed)
                self.assertFalse(terminated)
                self.assertFalse(info["collision"])
                interventions += int(info["shielded"])
                if info["shielded"]:
                    self.assertFalse(np.array_equal(proposed, info["executed_action"]))
            self.assertGreater(interventions, 0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
