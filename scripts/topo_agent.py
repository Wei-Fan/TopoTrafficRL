import logging
import torch
from gymnasium import spaces

from ttrl_agent.agents.common.models import model_factory, size_model_config, trainable_parameters
from ttrl_agent.agents.common.optimizers import loss_function_factory, optimizer_factory
from ttrl_agent.agents.deep_q_network.abstract import AbstractDQNAgent

from abc import ABC, abstractmethod

from ttrl_agent.configuration import Configurable

logger = logging.getLogger(__name__)


class TopoAgent(Configurable, ABC):
    def __init__(self, env, config=None):
        self.env = env
        assert isinstance(env.action_space, spaces.Discrete) or isinstance(env.action_space, spaces.Tuple), \
            "Only compatible with Discrete action spaces."
        self.training = True
        self.previous_state = None
        size_model_config(self.env, self.config["model"]) # update self.config's size based on observation_space and action_space
        # TODO: init agent model here
        self.steps = 0

    def save(self, filename):
        state = {'state_dict': self.value_net.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)
        return filename

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['state_dict'])
        self.target_net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return filename

    def set_writer(self, writer):
        self.writer = writer
        try:
            self.exploration_policy.set_writer(writer)
        except AttributeError:
            pass
        obs_shape = self.env.observation_space.shape if isinstance(self.env.observation_space, spaces.Box) else \
            self.env.observation_space.spaces[0].shape
        model_input = torch.zeros((1, *obs_shape), dtype=torch.float, device=self.device)
        self.writer.add_graph(self.value_net, input_to_model=(model_input,)),
        self.writer.add_scalar("agent/trainable_parameters", trainable_parameters(self.value_net), 0)

    def seed(self, seed=None):
        return self.exploration_policy.seed(seed)

    def reset(self):
        pass

    def plan(self, state):
        """
            Plan an optimal trajectory from an initial state.

        :param state: s, the initial state of the agent
                      np.ndarray
        :return: [a0, a1, a2...], a sequence of actions to perform, where a is a int
                 ACTIONS_LONGI = {0: "STOP", 1: "SLOWER", 2: "IDLE", 3: "FASTER"}
        """
        self.previous_state = state
        vehicles_data = self.convert_obs_to_dict(state)
        ego = vehicles_data["vehicle_0"]
        # TODO: find an action based on observation
        for vehicle_info in vehicles_data:
            x = vehicle_info["x"]
            y = vehicle_info["y"]
            vx = vehicle_info["vx"]
            vy = vehicle_info["vy"]
            cos_h = vehicle_info["cos_h"]
            sin_h = vehicle_info["sin_h"]

        # return action

    def set_directory(self, directory):
        self.directory = directory

    def convert_obs_to_dict(self, obs_array):
        """
            Convert the observation output (numpy array) into a dictionary of dictionaries.

        :param obs_array: numpy array from the observe() method.
        :return: dict of dicts where each key is 'vehicle_i' and the value is a dict of features.
        """
        features = self.env.config["observation"]["features"]
        vehicles_data = {}
        for i, row in enumerate(obs_array):
            # Map each row to a dictionary with feature names as keys.
            vehicle_dict = {feature: row[j] for j, feature in enumerate(features)}
            # Assign this dictionary to a vehicle key.
            vehicles_data[f'vehicle_{i}'] = vehicle_dict

        return vehicles_data
