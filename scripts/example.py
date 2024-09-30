# this is an example for learning the rl-agents, ttrl_env, and gymnasium.
# The original name for this file is intersection_social_dqn.py

import sys
import os
import gymnasium as gym
from ttrl_agent.agents.common.factory import load_agent, load_environment
from ttrl_agent.trainer.evaluation import Evaluation
from utils import show_videos

# Change the current working directory to rl-agents/scripts/
script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

# use environment configuratino file, which is defined in project_ws/rl-agents/scripts/configs/IntersectionEnv/env.json

env_config = os.path.join(script_path, 'configs', 'IntersectionEnv', 'env.json')

# use agent defined by rl-agents, which is defined in project_ws/rl-agents/scripts/configs/IntersectionEnv/agents/DQNAgent/ego_attention_2h.json
agent_config = os.path.join(script_path, 'configs', 'IntersectionEnv', 'agents', 'DQNAgent', 'ego_attention_2h.json')

# Load an environment from the configuration file.
env = load_environment(env_config)

# # Load an agent from the configuration file.
agent = load_agent(agent_config, env)

# # Create the evaluation of an agent interacting with an environment to maximize its expected reward.
NUM_EPISODES = 1000  #@param {type: "integer"}
evaluation = Evaluation(env, agent, num_episodes=NUM_EPISODES, display_env=True, display_agent=True)
print(f"Ready to train {agent} on {env}")

# # Training
evaluation.train()

# # Run the learned policy for a few episodes.
env = load_environment(env_config)
env.config["offscreen_rendering"] = True
agent = load_agent(agent_config, env)
evaluation = Evaluation(env, agent, num_episodes=20, training = False, recover = True)
evaluation.test()
test_path = evaluation.run_directory / "test"
show_videos(test_path)

# os.chdir('../..')