"""MuJoCo Panda GoalEnv for a fixed breadboard and changing hole targets."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import gymnasium as gym
import mujoco
import mujoco_menagerie as menagerie
import numpy as np
from gymnasium import spaces

from .config import TaskConfig
from .geometry import (
    Hole,
    canonical_quaternion,
    compute_reward,
    goal_reached,
    make_holes,
    pose_error,
    target_pose,
)


def _vec(values: tuple[float, ...] | np.ndarray) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _model_with_breadboard(cfg: TaskConfig, holes: list[Hole]) -> mujoco.MjModel:
    """Compose the published Panda MJCF with simple task geometry in memory."""
    panda = menagerie.get("franka_emika_panda")
    # 新版 mujoco-menagerie 的 xml() 返回 Path，需要读成字符串再解析
    xml = panda.xml("panda").read_text(encoding="utf-8")
    root = ET.fromstring(xml)
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(cfg.physics_dt))

    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("Panda MJCF has no worldbody")
    hand = next((body for body in root.iter("body") if body.get("name") == "hand"), None)
    if hand is None:
        raise RuntimeError("Panda MJCF has no hand body")
    ET.SubElement(
        hand,
        "site",
        name="breadboard_tcp",
        pos="0 0 0.1034",
        size="0.006",
        rgba="0.95 0.15 0.15 0.8",
        group="3",
    )

    # The robot base sits at z=0. The work surface is raised into a useful
    # reachable region while remaining clear of the base itself.
    ET.SubElement(
        world,
        "geom",
        name="work_table",
        type="box",
        pos="0.5 0 0.125",
        size="0.32 0.3 0.125",
        rgba="0.55 0.48 0.39 1",
        friction="1 0.005 0.0001",
    )
    ET.SubElement(
        world,
        "geom",
        name="breadboard",
        type="box",
        pos=_vec(cfg.board_center),
        size=_vec(cfg.board_half_size),
        rgba="0.92 0.91 0.83 1",
        friction="1 0.005 0.0001",
    )
    top = cfg.board_center[2] + cfg.board_half_size[2]
    for hole in holes:
        ET.SubElement(
            world,
            "site",
            name=f"hole_{hole.hole_id}",
            type="sphere",
            pos=_vec((cfg.board_center[0] + hole.local_xy[0], cfg.board_center[1] + hole.local_xy[1], top + 0.0002)),
            size="0.0018",
            rgba="0.13 0.18 0.29 0.9",
            group="3",
        )
    ET.SubElement(
        world,
        "site",
        name="selected_goal",
        type="sphere",
        pos="0 0 1",
        size="0.009",
        rgba="0.1 0.85 0.23 0.75",
        group="3",
    )
    for index in range(cfg.max_episode_steps + 1):
        ET.SubElement(
            world,
            "site",
            name=f"trace_{index}",
            type="sphere",
            pos="0 0 1",
            size="0.003",
            rgba="0.12 0.45 0.95 0",
            group="3",
        )
    model = mujoco.MjModel.from_xml_string(
        ET.tostring(root, encoding="unicode"), assets=panda.assets("panda")
    )
    return model


class BreadboardReachEnv(gym.Env):
    """Single Panda simulation; ``reset`` samples a new fixed-board hole."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        cfg: TaskConfig | None = None,
        *,
        split: str = "train",
        render_mode: str | None = None,
        render_width: int = 960,
        render_height: int = 720,
    ) -> None:
        super().__init__()
        self.cfg = cfg or TaskConfig()
        self.holes = make_holes(self.cfg)
        self.holes_by_id = {hole.hole_id: hole for hole in self.holes}
        if split not in ("train", "eval", "all"):
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self.available_holes = [h for h in self.holes if split == "all" or h.split == split]
        self.model = _model_with_breadboard(self.cfg, self.holes)
        self.data = mujoco.MjData(self.model)
        self._shadow = mujoco.MjData(self.model) if self.cfg.action_shield else None
        self._state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._state_buffer = np.empty(mujoco.mj_stateSize(self.model, self._state_spec))
        self._renderer: mujoco.Renderer | None = None
        self.render_mode = render_mode
        self.render_width, self.render_height = render_width, render_height
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.lookat[:] = (0.42, 0.0, 0.26)
        self._camera.distance = 1.05
        self._camera.azimuth = 140
        self._camera.elevation = -28
        self._render_option = mujoco.MjvOption()
        self._render_option.sitegroup[:] = 1

        self._joint_ids = np.asarray(
            [self._id(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
        )
        self._qpos_addr = self.model.jnt_qposadr[self._joint_ids]
        self._qvel_addr = self.model.jnt_dofadr[self._joint_ids]
        self._actuator_ids = np.asarray(
            [self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}") for i in range(1, 8)]
        )
        self._finger_actuator_id = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self._tcp_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "breadboard_tcp")
        self._goal_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "selected_goal")
        self._trace_site_ids = np.asarray(
            [self._id(mujoco.mjtObj.mjOBJ_SITE, f"trace_{i}") for i in range(self.cfg.max_episode_steps + 1)]
        )
        self._hazard_geom_ids = {
            self._id(mujoco.mjtObj.mjOBJ_GEOM, "work_table"),
            self._id(mujoco.mjtObj.mjOBJ_GEOM, "breadboard"),
        }
        self._joint_low = self.model.jnt_range[self._joint_ids, 0].astype(np.float32)
        self._joint_high = self.model.jnt_range[self._joint_ids, 1].astype(np.float32)
        self._substeps = round(self.cfg.control_dt / self.cfg.physics_dt)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, shape=(28,), dtype=np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "desired_goal": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            }
        )
        self._hole: Hole | None = None
        self._goal = np.zeros(7, dtype=np.float32)
        self._last_action = np.zeros(7, dtype=np.float32)
        self._joint_target = np.zeros(7, dtype=np.float64)
        self._step_count = 0
        self._hold_count = 0
        self._success_once = False
        self._trace_count = 0

    def _id(self, kind: mujoco.mjtObj, name: str) -> int:
        identifier = mujoco.mj_name2id(self.model, kind, name)
        if identifier < 0:
            raise RuntimeError(f"MJCF object missing: {name}")
        return identifier

    @property
    def hole_id(self) -> str:
        if self._hole is None:
            raise RuntimeError("Reset the environment before requesting hole_id")
        return self._hole.hole_id

    @property
    def target(self) -> np.ndarray:
        return self._goal.copy()

    def _achieved_goal(self) -> np.ndarray:
        quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, self.data.site_xmat[self._tcp_id].reshape(9))
        return np.concatenate(
            (
                self.data.site_xpos[self._tcp_id],
                canonical_quaternion(quaternion),
            )
        ).astype(np.float32)

    def _observation(self) -> dict[str, np.ndarray]:
        achieved = self._achieved_goal()
        state = np.concatenate(
            (
                self.data.qpos[self._qpos_addr],
                self.data.qvel[self._qvel_addr],
                achieved,
                self._last_action,
            )
        ).astype(np.float32)
        return {
            "observation": state,
            "achieved_goal": achieved,
            "desired_goal": self._goal.copy(),
        }

    def _collision_pair(self, data: mujoco.MjData | None = None) -> tuple[str, str] | None:
        data = self.data if data is None else data
        def label(geom_id: int) -> str:
            name = self.model.geom(geom_id).name
            if name:
                return name
            body_id = int(self.model.geom_bodyid[geom_id])
            return f"{self.model.body(body_id).name}/geom_{geom_id}"

        for index in range(data.ncon):
            contact = data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in self._hazard_geom_ids and self.model.geom_bodyid[geom2] != 0:
                return (label(geom1), label(geom2))
            if geom2 in self._hazard_geom_ids and self.model.geom_bodyid[geom1] != 0:
                return (label(geom2), label(geom1))
        return None

    def _has_collision(self, data: mujoco.MjData | None = None) -> bool:
        return self._collision_pair(data) is not None

    def _safe_action(self, requested: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        if self._shadow is None:
            return requested, False, False
        if self._has_collision():
            return np.zeros_like(requested), True, True
        mujoco.mj_getState(self.model, self.data, self._state_buffer, self._state_spec)
        current_target = self._joint_target.copy()
        candidates = [requested * (0.5**attempt) for attempt in range(self.cfg.max_action_halvings + 1)]
        candidates.extend(-requested * scale for scale in (0.5, 1.0))
        candidates.extend(-self._last_action * scale for scale in (0.5, 1.0))
        candidates.append(np.zeros_like(requested))
        for attempt, action in enumerate(candidates):
            mujoco.mj_setState(self.model, self._shadow, self._state_buffer, self._state_spec)
            self._shadow.ctrl[self._actuator_ids] = np.clip(
                current_target + self.cfg.max_joint_delta * action,
                self._joint_low + 0.01,
                self._joint_high - 0.01,
            )
            self._shadow.ctrl[self._finger_actuator_id] = 255
            mujoco.mj_forward(self.model, self._shadow)
            safe = not self._has_collision(self._shadow)
            for _ in range(self._substeps):
                if not safe:
                    break
                mujoco.mj_step(self.model, self._shadow)
                safe = not self._has_collision(self._shadow)
            if safe:
                return action.astype(np.float32), attempt != 0, False
        return np.zeros_like(requested), True, True

    def _append_trace(self) -> None:
        if self._trace_count >= len(self._trace_site_ids):
            return
        site_id = self._trace_site_ids[self._trace_count]
        self.model.site_pos[site_id] = self.data.site_xpos[self._tcp_id]
        self.model.site_rgba[site_id] = (0.12, 0.45, 0.95, 0.7)
        self.data.site_xpos[site_id] = self.data.site_xpos[self._tcp_id]
        self._trace_count += 1

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        options = options or {}
        requested = options.get("hole_id")
        if requested is not None:
            if requested not in self.holes_by_id:
                raise ValueError(f"Unknown hole: {requested}")
            self._hole = self.holes_by_id[requested]
        else:
            self._hole = self.available_holes[int(self.np_random.integers(len(self.available_holes)))]
        self._goal = target_pose(self.cfg, self._hole)
        self.model.site_pos[self._goal_site_id] = self._goal[:3]
        self.model.site_rgba[self._trace_site_ids] = (0.12, 0.45, 0.95, 0)
        self._trace_count = 0

        home_key = self._id(mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, home_key)
        home_positions = options.get("joint_positions", self.cfg.home_joint_positions)
        if home_positions is not None:
            home_positions = np.asarray(home_positions, dtype=np.float64)
            if home_positions.shape != (7,) or not np.isfinite(home_positions).all():
                raise ValueError("joint_positions must contain seven finite values")
            self.data.qpos[self._qpos_addr] = np.clip(
                home_positions, self._joint_low + 0.02, self._joint_high - 0.02
            )
        self.data.qvel[:] = 0
        joint_noise = float(options.get("joint_noise", self.cfg.reset_joint_noise))
        if joint_noise:
            noise = self.np_random.uniform(
                -joint_noise, joint_noise, size=7
            )
            self.data.qpos[self._qpos_addr] = np.clip(
                self.data.qpos[self._qpos_addr] + noise,
                self._joint_low + 0.02,
                self._joint_high - 0.02,
            )
        self.data.ctrl[self._actuator_ids] = self.data.qpos[self._qpos_addr]
        self._joint_target = self.data.ctrl[self._actuator_ids].copy()
        self.data.ctrl[self._finger_actuator_id] = 255
        self._last_action.fill(0)
        self._step_count = self._hold_count = 0
        self._success_once = False
        mujoco.mj_forward(self.model, self.data)
        self._append_trace()
        return self._observation(), {"hole_id": self.hole_id, "split": self._hole.split}

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict | None
    ) -> np.ndarray:
        return compute_reward(achieved_goal, desired_goal, info, self.cfg)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        if self._hole is None:
            raise RuntimeError("Call reset before step")
        if self._step_count >= self.cfg.max_episode_steps:
            raise RuntimeError("Call reset after episode truncation")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Action must contain seven finite values")
        action = np.clip(action, -1, 1)
        executed_action, shielded, blocked = self._safe_action(action)
        collision = False
        if not blocked:
            target = np.clip(
                self._joint_target + self.cfg.max_joint_delta * executed_action,
                self._joint_low + 0.01,
                self._joint_high - 0.01,
            )
            self._joint_target = target.copy()
            self.data.ctrl[self._actuator_ids] = target
            self.data.ctrl[self._finger_actuator_id] = 255
            for _ in range(self._substeps):
                mujoco.mj_step(self.model, self.data)
                collision = collision or self._has_collision()
                if collision:
                    break
        self._last_action = executed_action
        self._step_count += 1
        mujoco.mj_forward(self.model, self.data)
        self._append_trace()
        obs = self._observation()
        currently_at_goal = bool(goal_reached(obs["achieved_goal"], self._goal, self.cfg))
        self._hold_count = self._hold_count + 1 if currently_at_goal and not collision else 0
        self._success_once = self._success_once or self._hold_count >= self.cfg.hold_steps
        if collision:
            self._success_once = False
        position_error, angular_error = pose_error(obs["achieved_goal"], self._goal)
        reward = float(self.compute_reward(obs["achieved_goal"], self._goal, {"collision": collision, "shielded": shielded}))
        terminated = bool(collision)
        truncated = self._step_count >= self.cfg.max_episode_steps and not terminated
        info = {
            "hole_id": self.hole_id,
            "split": self._hole.split,
            "step": self._step_count,
            "position_error_m": float(position_error),
            "angular_error_deg": float(np.rad2deg(angular_error)),
            "currently_at_goal": currently_at_goal,
            "hold_count": self._hold_count,
            "success": self._success_once,
            "collision": collision,
            "collision_pair": self._collision_pair(),
            "shielded": shielded,
            "blocked": blocked,
            "executed_action": executed_action.copy(),
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        if self._renderer is None:
            # 把离屏缓冲区尺寸设成渲染尺寸，否则 Renderer 默认 640x480 会报错
            self.model.vis.global_.offwidth = self.render_width
            self.model.vis.global_.offheight = self.render_height
            self._renderer = mujoco.Renderer(
                self.model, height=self.render_height, width=self.render_width
            )
        self._renderer.update_scene(
            self.data, camera=self._camera, scene_option=self._render_option
        )
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
