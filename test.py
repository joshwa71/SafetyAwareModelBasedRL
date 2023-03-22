#!/usr/bin/env python3
import safety_gym
import mujoco_py
import gym

env = gym.make('Safexp-PointGoal2-v0')
print(env.action_space)
observation = env.reset()
for i in range(10000):
    env.render()    
    next_observation, reward, done, info = env.step(env.action_space.sample())
    #print(reward)