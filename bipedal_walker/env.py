import gymnasium as gym
from gymnasium import spaces
import Box2D
from Box2D.b2 import world, polygonShape, revoluteJointDef
import numpy as np
import pygame

"""
action space: four joint, each joint is [-1, 1]
observation space: 
- angle velocity of the four joint
- angle of the four joint
- x, y velocity
- angle of the body
- state the base touch with the ground
"""

class CustomBipedalWalker(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # 1. the gravity of the world -9.81
        self.world = world(gravity = (0, 9.81), doSleep = True)

        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        self.hull = None
        self.legs = []
        self.joints = []

    def create_environment_(self):
        # Create ground and other environment elements here
        ground = self.world.CreateStaticBody(
            shapes = polygonShape(box = (50, 1)),
            position = (25, 0),
            friction = 0.8
        )

    def create_robot_(self):
        hull_shape = polygonShape(box = (0.4, 0.2))
        self.hull = self.world.CreateDynamicBody(
            position = (2.0, 1.5),
            fixtures = Box2D.b2FixtureDef(shape = hull_shape, density = 5.0, friction = 0.1)
        )
        self.legs = []
        self.joints = []
        # side is the offset of the thigh/leg of init place
        for side in [-1, 1]:
            thigh = self.world.CreateDynamicBody(
                Position = (2.0 + side* 0.1, 1.1),
                fixtures=Box2D.b2FixtureDef(shape=polygonShape(box=(0.08, 0.3)), density=1.0)
            )
            leg = self.world.CreateDynamicBody(
                Position = (2.0 + side* 0.1, 1.1),
                fixtures = Box2D.b2FixtureDef(shape = polygonShape(box = (0.06, 0.3)), density=1.0)
            )
            rjd1 = revoluteJointDef(
                bodyA = self.hull, bodyB = thigh,
                anchor = (2.0 + side * 0.1, 1.3),
                lowerAngle = -1.5, upperAngle = 1.5, enableLimit = True,
                maxMotorTorque = 80
            )
            rjd2 = revoluteJointDef(
                bodyA = thigh, bodyB = leg,
                anchor = (2.0 + side * 0.1, 0.8),
                lowerAngle = -1.6, upperAngle = 1.5, enableLimit=True,
                maxMotorTorque = 60.0, enableMotor=True
            )
            self.joints.append(self.world.CreateJoint(rjd1))
            self.joints.append(self.world.CreateJoint(rjd2))
            self.legs.extend([thigh, leg])

    def get_obs_(self):
        pos = self.hull.position
        vel = self.hull.linearVelocity
        angle = self.hull.angle
        angular_vel = self.hull.angularVelocity
        joint_states = []
        for j in self.joints:
            joint_states.extend([j.angle, j.speed])
        obs = np.array([
            angle, angular_vel, vel.x, vel.y,
            *joint_states,
            0.0, 0.0
        ], dtype = np.float32)
        return obs

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        for body in self.world.bodies:
            self.world.DestroyBody(body)
        self.create_environment_()
        self.create_robot_()
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        for j, a in zip(self.joints, action):
            j.motorSpeed = float(np.clip(a, -1, 1) * 6.0)
        self.world.Step(1.0 / 50.0, 6, 2)
        obs = self.get_obs_()
        # reward for getting front
        forward_reward = self.hull.linearVelocity.x * 0.1
        # penalty for using energy to move
        energy_penalty = -0.001 * np.sum(np.square(action))
        reward = forward_reward + energy_penalty
        terminated = False
        if abs(self.hull.angle) > 1.5:
            reward -= 100.0
            terminated = True
        truncated = False
        return obs, reward, terminated, truncated, {}