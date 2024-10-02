# this is an example for learning the rl-agents, ttrl_env, and gymnasium.
# The original name for this file is intersection_social_dqn.py

import sys
import os
import json
import importlib
import gymnasium as gym
from ttrl_agent.agents.common.factory import load_environment
from simulation import Simulation
from utils import show_videos

# Change the current working directory to scripts/
script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

# use environment configuratino file, which is defined in /configs/IntersectionEnv/env.json
env_config = os.path.join(script_path, 'configs', 'IntersectionEnv', 'env.json')

# use agent defined by rl-agents, which is defined in /configs/IntersectionEnv/agents/DQNAgent/ego_attention_2h.json
agent_config = os.path.join(script_path, 'configs', 'IntersectionEnv', 'agents', 'DQNAgent', 'ego_attention_2h.json')

# Load an environment from the configuration file.
env = load_environment(env_config)

# Load an agent from the class.
# The agent class must have:
# def reset(self): # reset agent
# def seed(self, seed=None): # init agent
with open(agent_config) as f:
    agent_config = json.loads(f.read())
if "__class__" in agent_config:
    path = agent_config['__class__'].split("'")[1]
    module_name, class_name = path.rsplit(".", 1)
    agent_class = getattr(importlib.import_module(module_name), class_name)
    agent = agent_class(env, agent_config)


# Run the simulation.
# TODO: breakdown evaluation
NUM_EPISODES = 20000  #@param {type: "integer"}
simulation = Simulation(env, agent, num_episodes=NUM_EPISODES, display_env=True, display_agent=True)
print(f"Ready to run {agent} on {env}")
simulation.run()


# Record video data.
# TODO: add this into evaluation iteration
# env = load_environment(env_config)
# env.config["offscreen_rendering"] = True
# agent = load_agent(agent_config, env)
# evaluation = Evaluation(env, agent, num_episodes=1000, training = False, recover = True)
# evaluation.test()
# test_path = evaluation.run_directory / "test"
# show_videos(test_path)
