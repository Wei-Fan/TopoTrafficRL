# this is an example for learning the rl-agents, ttrl_env, and gymnasium.
# The original name for this file is intersection_social_dqn.py

import sys
import os
import json
import importlib
import gymnasium as gym
from ttrl_agent.agents.common.factory import load_environment
from topo_agent import TopoAgent
from simulation import Simulation
from utils import show_videos

# Change the current working directory to scripts/
script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

# use environment configuratino file, which is defined in /configs/IntersectionEnv/env.json
env_config = os.path.join(script_path, 'configs', 'IntersectionEnv', 'env.json')

# use agent # TODO: update config file
agent_config = os.path.join(script_path, 'configs', 'IntersectionEnv', 'agents', 'topo.json')

# Load an environment from the configuration file.
env = load_environment(env_config)

# Load an agent from the class.
# The agent class must have:
# def reset(self): # reset agent
# def seed(self, seed=None): # init agent
# def act(self, state): # plan action at current state. This is where optimization based method implemented.
# def record(self, state, action, reward, next_state, done, info): # Record a transition by performing a Deep Q-Network iteration
# def save(self, filename): save the agent
if not isinstance(agent_config, dict):
    with open(agent_config) as f:
        agent_config = json.loads(f.read())
agent = TopoAgent(env, agent_config)


# Run the simulation.
NUM_EPISODES = 20000  #@param {type: "integer"}
simulation = Simulation(env, agent, num_episodes=NUM_EPISODES, display_env=True)
print(f"Ready to run {agent} on {env}")
simulation.run()


# Record video data.
# TODO: add this into evaluation iteration
# env = load_environment(env_config)
# env.config["offscreen_rendering"] = True
# agent = load_agent(agent_config, env)
# evaluation = Evaluation(env, agent, num_episodes=1000, training = False, recover = True)
# test_path = evaluation.run_directory / "test"
# show_videos(test_path)
