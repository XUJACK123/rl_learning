import unittest

import numpy as np

from breadboard_reach.config import TaskConfig
from breadboard_reach.geometry import (
    board_to_world,
    compute_reward,
    make_holes,
    pose_error,
    target_pose,
)
from breadboard_reach.her import Transition, relabel_episode


def observation(position: float, goal: float) -> dict[str, np.ndarray]:
    pose = np.array([position, 0, 0.3, 1, 0, 0, 0], dtype=np.float32)
    desired = np.array([goal, 0, 0.3, 1, 0, 0, 0], dtype=np.float32)
    return {
        "observation": np.zeros(28, dtype=np.float32),
        "achieved_goal": pose,
        "desired_goal": desired,
    }


class GeometryTests(unittest.TestCase):
    def test_target_transform_and_split(self):
        cfg = TaskConfig()
        holes = make_holes(cfg)
        self.assertEqual(holes, make_holes(cfg))
        self.assertTrue(any(h.split == "train" for h in holes))
        self.assertTrue(any(h.split == "eval" for h in holes))
        pose = target_pose(cfg, holes[0])
        self.assertAlmostEqual(pose[2], 0.29, places=6)
        transformed = board_to_world(np.array([1.0, 0.0, 0.0]), np.zeros(3), np.pi / 2)
        np.testing.assert_allclose(transformed, [0.0, 1.0, 0.0], atol=1e-7)

    def test_quaternion_sign_and_batch_reward(self):
        cfg = TaskConfig()
        achieved = np.array(
            [[0.0, 0.0, 0.3, 1, 0, 0, 0], [0.1, 0, 0.3, 1, 0, 0, 0]],
            dtype=np.float32,
        )
        desired = np.array(
            [[0.0, 0.0, 0.3, -1, 0, 0, 0], [0.0, 0, 0.3, 1, 0, 0, 0]],
            dtype=np.float32,
        )
        distance, angle = pose_error(achieved, desired)
        np.testing.assert_allclose(distance, [0, 0.1], atol=1e-6)
        np.testing.assert_allclose(angle, [0, 0], atol=1e-6)
        reward = compute_reward(achieved, desired, {"collision": np.array([0, 1])}, cfg)
        np.testing.assert_allclose(reward, [5, -11], atol=1e-6)


class HERTests(unittest.TestCase):
    def test_future_goal_is_local_and_reward_is_recomputed(self):
        cfg = TaskConfig()
        episode = []
        for index in range(4):
            episode.append(
                Transition(
                    observation=observation(index * 0.01, 1.0),
                    action=np.zeros(7, dtype=np.float32),
                    reward=-1.0,
                    next_observation=observation((index + 1) * 0.01, 1.0),
                    terminated=False,
                    truncated=index == 3,
                    collision=False,
                    episode_id=7,
                    step_index=index,
                    shielded=index == 1,
                )
            )
        samples = relabel_episode(
            episode,
            future_k=4,
            rng=np.random.default_rng(9),
            reward_fn=lambda a, d, info: compute_reward(a, d, info, cfg),
        )
        self.assertEqual(len(samples), 16)
        self.assertEqual(sum(t.future_index is None for t in samples), 4)
        for sample in samples[4:]:
            self.assertGreater(sample.future_index, sample.step_index)
            self.assertEqual(sample.episode_id, 7)
            np.testing.assert_array_equal(
                sample.observation["desired_goal"], sample.next_observation["desired_goal"]
            )
            np.testing.assert_array_equal(
                sample.observation["desired_goal"],
                episode[sample.future_index].next_observation["achieved_goal"],
            )
            expected = compute_reward(
                sample.next_observation["achieved_goal"],
                sample.next_observation["desired_goal"],
                {"collision": False, "shielded": sample.shielded},
                cfg,
            )
            self.assertEqual(sample.reward, float(expected))
            self.assertEqual(sample.truncated, sample.step_index == 3)

    def test_rejects_mixed_or_incomplete_episode(self):
        transition = Transition(
            observation=observation(0, 1),
            action=np.zeros(7, dtype=np.float32),
            reward=-1,
            next_observation=observation(0.01, 1),
            terminated=False,
            truncated=False,
            collision=False,
            episode_id=1,
            step_index=0,
        )
        with self.assertRaises(ValueError):
            relabel_episode([transition], future_k=1, rng=np.random.default_rng(0), reward_fn=lambda *_: -1)


if __name__ == "__main__":
    unittest.main()
