import datetime
import json
import logging
import os
import time
from pathlib import Path
import numpy as np
from tensorboardX import SummaryWriter
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, capped_cubic_video_schedule

import ttrl_agent.trainer.logger
from ttrl_agent.agents.common.graphics import AgentGraphics
from ttrl_agent.configuration import serialize
from ttrl_agent.trainer.graphics import RewardViewer

logger = logging.getLogger(__name__)


class Simulation(object):
    """
        The simulation of an agent interacting with an environment to maximize its expected reward.
    """

    OUTPUT_FOLDER = 'out'
    SAVED_MODELS_FOLDER = 'saved_models'
    RUN_FOLDER = 'run_{}_{}'
    METADATA_FILE = 'metadata.{}.json'
    LOGGING_FILE = 'logging.{}.log'

    def __init__(self,
                 env,
                 agent,
                 directory=None,
                 run_directory=None,
                 num_episodes=1000,
                 training=True,
                 sim_seed=None,
                 recover=None,
                 display_env=True,
                 display_rewards=True,
                 close_env=True,
                 step_callback_fn=None):
        """

        :param env: The environment to be solved, possibly wrapping an AbstractEnv environment
        :param AbstractAgent agent: The agent solving the environment
        :param Path directory: Workspace directory path
        :param Path run_directory: Run directory path
        :param int num_episodes: Number of episodes run
        !param training: Whether the agent is being trained or tested
        :param sim_seed: The seed used for the environment/agent randomness source
        :param recover: Recover the agent parameters from a file.
                        - If True, it the default latest save will be used.
                        - If a string, it will be used as a path.
        :param display_env: Render the environment, and have a monitor recording its videos
        :param display_agent: Add the agent graphics to the environment viewer, if supported
        :param display_rewards: Display the performances of the agent through the episodes
        :param close_env: Should the environment be closed when the evaluation is closed
        :param step_callback_fn: A callback function called after every environment step. It takes the following
               arguments: (episode, env, agent, transition, writer).

        """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.training = training
        self.sim_seed = sim_seed if sim_seed is not None else np.random.randint(0, 1e6)
        self.close_env = close_env
        self.display_env = display_env
        self.step_callback_fn = step_callback_fn

        self.directory = Path(directory or self.default_directory)
        self.run_directory = self.directory / (run_directory or self.default_run_directory)
        self.wrapped_env = RecordVideo(env,
                                       self.run_directory,
                                       episode_trigger=(None if self.display_env else lambda e: False))
        try:
            self.wrapped_env.unwrapped.set_record_video_wrapper(self.wrapped_env)
        except AttributeError:
            pass
        self.wrapped_env = RecordEpisodeStatistics(self.wrapped_env)
        self.episode = 0
        self.writer = SummaryWriter(str(self.run_directory))
        self.agent.set_writer(self.writer)
        self.agent.evaluation = self
        self.write_logging()
        self.write_metadata()
        self.filtered_agent_stats = 0
        self.best_agent_stats = -np.infty, 0

        self.recover = recover
        if self.recover:
            self.load_agent_model(self.recover)
        self.reward_viewer = None
        if display_rewards:
            self.reward_viewer = RewardViewer()
        self.observation = None

    def run(self):
        self.training = True
        self.run_episodes()
        self.close()

    def test(self):
        """
        Test the agent.

        If applicable, the agent model should be loaded before using the recover option.
        """
        self.training = False
        if self.display_env:
            self.wrapped_env.episode_trigger = lambda e: True
        try:
            self.agent.eval()
        except AttributeError:
            pass
        self.run_episodes()
        self.close()

    def run_episodes(self):
        for self.episode in range(self.num_episodes):
            # Run episode
            terminal = False
            self.reset(seed=self.episode)
            rewards = []
            start_time = time.time()
            while not terminal:
                # Step until a terminal step is reached
                reward, terminal = self.step()
                rewards.append(reward)

                # Catch interruptions
                try:
                    if self.env.unwrapped.done:
                        break
                except AttributeError:
                    pass

            # End of episode
            duration = time.time() - start_time
            self.after_all_episodes(self.episode, rewards, duration)
            self.after_some_episodes(self.episode, rewards)

    def step(self):
        """
            Plan a sequence of actions according to the agent policy, and step the environment accordingly.
        """
        # Query agent for actions sequence
        actions = self.agent.plan(self.observation)
        if not actions:
            raise Exception("The agent did not plan any action")

        # Forward the actions to the environment viewer
        try:
            self.env.unwrapped.viewer.set_agent_action_sequence(actions)
        except AttributeError:
            pass

        # Step the environment
        previous_observation, action = self.observation, actions[0]
        transition = self.wrapped_env.step(action)
        self.observation, reward, done, truncated, info = transition
        terminal = done or truncated

        # Call callback
        if self.step_callback_fn is not None:
            self.step_callback_fn(self.episode, self.wrapped_env, self.agent, transition, self.writer)

        # Record the experience.
        try:
            self.agent.record(previous_observation, action, reward, self.observation, done, info)
        except NotImplementedError:
            pass

        return reward, terminal

    def save_agent_model(self, identifier, do_save=True):
        # Create the folder if it doesn't exist
        permanent_folder = self.directory / self.SAVED_MODELS_FOLDER
        os.makedirs(permanent_folder, exist_ok=True)

        episode_path = None
        if do_save:
            episode_path = Path(self.run_directory) / "checkpoint-{}.tar".format(identifier)
            try:
                self.agent.save(filename=permanent_folder / "latest.tar")
                episode_path = self.agent.save(filename=episode_path)
                if episode_path:
                    logger.info("Saved {} model to {}".format(self.agent.__class__.__name__, episode_path))
            except NotImplementedError:
                pass
        return episode_path

    def load_agent_model(self, model_path):
        if model_path is True:
            model_path = self.directory / self.SAVED_MODELS_FOLDER / "latest.tar"
        if isinstance(model_path, str):
            model_path = Path(model_path)
            if not model_path.exists():
                model_path = self.directory / self.SAVED_MODELS_FOLDER / model_path
        try:
            model_path = self.agent.load(filename=model_path)
            if model_path:
                logger.info("Loaded {} model from {}".format(self.agent.__class__.__name__, model_path))
        except FileNotFoundError:
            logger.warning("No pre-trained model found at the desired location.")
        except NotImplementedError:
            pass

    def after_all_episodes(self, episode, rewards, duration):
        rewards = np.array(rewards)
        gamma = self.agent.config.get("gamma", 1)
        self.writer.add_scalar('episode/length', len(rewards), episode)
        self.writer.add_scalar('episode/total_reward', sum(rewards), episode)
        self.writer.add_scalar('episode/return', sum(r*gamma**t for t, r in enumerate(rewards)), episode)
        self.writer.add_scalar('episode/fps', len(rewards) / max(duration, 1e-6), episode)
        self.writer.add_histogram('episode/rewards', rewards, episode)
        logger.info("Episode {} score: {:.1f}".format(episode, sum(rewards)))

    def after_some_episodes(self, episode, rewards,
                            best_increase=1.1,
                            episodes_window=50):
        if capped_cubic_video_schedule(episode):
            # Save the model
            if self.training:
                self.save_agent_model(episode)

        if self.training:
            # Save best model so far, averaged on a window
            best_reward, best_episode = self.best_agent_stats
            self.filtered_agent_stats += 1 / episodes_window * (np.sum(rewards) - self.filtered_agent_stats)
            if self.filtered_agent_stats > best_increase * best_reward \
                    and episode >= best_episode + episodes_window:
                self.best_agent_stats = (self.filtered_agent_stats, episode)
                self.save_agent_model("best")

    @property
    def default_directory(self):
        return Path(self.OUTPUT_FOLDER) / self.env.unwrapped.__class__.__name__ / self.agent.__class__.__name__

    @property
    def default_run_directory(self):
        return self.RUN_FOLDER.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), os.getpid())

    def write_metadata(self):
        metadata = dict(env=serialize(self.env), agent=serialize(self.agent))
        file_infix = '{}.{}'.format(id(self.wrapped_env), os.getpid())
        file = self.run_directory / self.METADATA_FILE.format(file_infix)
        with file.open('w') as f:
            json.dump(metadata, f, sort_keys=True, indent=4)

    def write_logging(self):
        file_infix = '{}.{}'.format(id(self.wrapped_env), os.getpid())
        ttrl_agent.trainer.logger.configure()
        ttrl_agent.trainer.logger.add_file_handler(self.run_directory / self.LOGGING_FILE.format(file_infix))

    def reset(self, seed=0):
        seed = self.sim_seed + seed if self.sim_seed is not None else None
        self.observation, info = self.wrapped_env.reset()
        self.agent.seed(seed)  # Seed the agent with the main environment seed
        self.agent.reset()

    def close(self):
        """
            Close the evaluation.
        """
        if self.training:
            self.save_agent_model("final")
        self.wrapped_env.close()
        self.writer.close()
        if self.close_env:
            self.env.close()
