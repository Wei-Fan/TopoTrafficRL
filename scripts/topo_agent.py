import logging
import torch
from gymnasium import spaces

from ttrl_agent.agents.common.memory import Transition
from ttrl_agent.agents.common.models import model_factory, size_model_config, trainable_parameters
from ttrl_agent.agents.common.optimizers import loss_function_factory, optimizer_factory
from ttrl_agent.agents.common.utils import choose_device
from ttrl_agent.agents.deep_q_network.abstract import AbstractDQNAgent

from abc import ABC, abstractmethod

from ttrl_agent.configuration import Configurable
from ttrl_agent.agents.common.exploration.abstract import exploration_factory
from ttrl_agent.agents.common.memory import ReplayMemory, Transition

logger = logging.getLogger(__name__)


class TopoAgent(Configurable, ABC):
    def __init__(self, env, config=None):
        self.env = env
        assert isinstance(env.action_space, spaces.Discrete) or isinstance(env.action_space, spaces.Tuple), \
            "Only compatible with Discrete action spaces."
        self.memory = ReplayMemory(self.config)
        self.exploration_policy = exploration_factory(self.config["exploration"], self.env.action_space)
        self.training = True
        self.previous_state = None
        size_model_config(self.env, self.config["model"]) # update self.config based on observation_space and action_space
        # TODO: init agent model here
        self.steps = 0

    def step_optimizer(self, loss):
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def compute_bellman_residual(self, batch, target_state_action_value=None):
        # Compute concatenate the batch elements
        if not isinstance(batch.state, torch.Tensor):
            # logger.info("Casting the batch to torch.tensor")
            state = torch.cat(tuple(torch.tensor([batch.state], dtype=torch.float))).to(self.device)
            action = torch.tensor(batch.action, dtype=torch.long).to(self.device)
            reward = torch.tensor(batch.reward, dtype=torch.float).to(self.device)
            next_state = torch.cat(tuple(torch.tensor([batch.next_state], dtype=torch.float))).to(self.device)
            terminal = torch.tensor(batch.terminal, dtype=torch.bool).to(self.device)
            batch = Transition(state, action, reward, next_state, terminal, batch.info)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.value_net(batch.state)
        state_action_values = state_action_values.gather(1, batch.action.unsqueeze(1)).squeeze(1)

        if target_state_action_value is None:
            with torch.no_grad():
                # Compute V(s_{t+1}) for all next states.
                next_state_values = torch.zeros(batch.reward.shape).to(self.device)
                if self.config["double"]:
                    # Double Q-learning: pick best actions from policy network
                    _, best_actions = self.value_net(batch.next_state).max(1)
                    # Double Q-learning: estimate action values from target network
                    best_values = self.target_net(batch.next_state).gather(1, best_actions.unsqueeze(1)).squeeze(1)
                else:
                    best_values, _ = self.target_net(batch.next_state).max(1)
                next_state_values[~batch.terminal] = best_values[~batch.terminal]
                # Compute the expected Q values
                target_state_action_value = batch.reward + self.config["gamma"] * next_state_values

        # Compute loss
        loss = self.loss_function(state_action_values, target_state_action_value)
        return loss, target_state_action_value, batch

    def get_batch_state_values(self, states):
        values, actions = self.value_net(torch.tensor(states, dtype=torch.float).to(self.device)).max(1)
        return values.data.cpu().numpy(), actions.data.cpu().numpy()

    def get_batch_state_action_values(self, states):
        return self.value_net(torch.tensor(states, dtype=torch.float).to(self.device)).data.cpu().numpy()

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

    def initialize_model(self):
        self.value_net.reset()

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

    def record(self, state, action, reward, next_state, done, info):
        """
            Record a transition by performing a Deep Q-Network iteration

            - push the transition into memory
            - sample a minibatch
            - compute the bellman residual loss over the minibatch
            - perform one gradient descent step
            - slowly track the policy network with the target network
        :param state: a state
        :param action: an action
        :param reward: a reward
        :param next_state: a next state
        :param done: whether state is terminal
        """
        if not self.training:
            return
        if isinstance(state, tuple) and isinstance(action, tuple):  # Multi-agent setting
            [self.memory.push(agent_state, agent_action, reward, agent_next_state, done, info)
             for agent_state, agent_action, agent_next_state in zip(state, action, next_state)]
        else:  # Single-agent setting
            self.memory.push(state, action, reward, next_state, done, info)
        batch = self.sample_minibatch()
        if batch:
            loss, _, _ = self.compute_bellman_residual(batch)
            self.step_optimizer(loss)
            self.update_target_network()

    def act(self, state, step_exploration_time=True):
        """
            Act according to the state-action value model and an exploration policy
        :param state: current state
        :param step_exploration_time: step the exploration schedule
        :return: an action
        """
        self.previous_state = state
        if step_exploration_time:
            self.exploration_policy.step_time()
        # Handle multi-agent observations
        # TODO: it would be more efficient to forward a batch of states
        if isinstance(state, tuple):
            return tuple(self.act(agent_state, step_exploration_time=False) for agent_state in state)

        # Single-agent setting
        values = self.get_state_action_values(state)
        self.exploration_policy.update(values)
        return self.exploration_policy.sample()

    def sample_minibatch(self):
        if len(self.memory) < self.config["batch_size"]:
            return None
        transitions = self.memory.sample(self.config["batch_size"])
        return Transition(*zip(*transitions))

    def update_target_network(self):
        self.steps += 1
        if self.steps % self.config["target_update"] == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())

    def get_state_value(self, state):
        """
        :param state: s, an environment state
        :return: V, its state-value
        """
        values, actions = self.get_batch_state_values([state])
        return values[0], actions[0]

    def get_state_action_values(self, state):
        """
        :param state: s, an environment state
        :return: [Q(a1,s), ..., Q(an,s)] the array of its action-values for each actions
        """
        return self.get_batch_state_action_values([state])[0]

    def step_optimizer(self, loss):
        raise NotImplementedError

    def seed(self, seed=None):
        return self.exploration_policy.seed(seed)

    def reset(self):
        pass

    def action_distribution(self, state):
        self.previous_state = state
        values = self.get_state_action_values(state)
        self.exploration_policy.update(values)
        return self.exploration_policy.get_distribution()

    def set_time(self, time):
        self.exploration_policy.set_time(time)

    def eval(self):
        self.training = False
        self.config['exploration']['method'] = "Greedy"
        self.exploration_policy = exploration_factory(self.config["exploration"], self.env.action_space)

    def plan(self, state):
        """
            Plan an optimal trajectory from an initial state.

        :param state: s, the initial state of the agent
        :return: [a0, a1, a2...], a sequence of actions to perform
        """
        return [self.act(state)]

    def set_directory(self, directory):
        self.directory = directory
