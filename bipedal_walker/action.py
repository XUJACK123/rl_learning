import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from custom_bipedal_walker import CustomBipedalWalker

def make_env():
    def _init():
        return CustomBipedalWalker()
    return _init()

def train():
    env = DummyVecEnv