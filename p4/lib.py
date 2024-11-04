import gymnasium as gym
from numpy import arctan, pi
from random import random, choice
from json import load, dump
from enum import Enum


Algorithm = Enum("Algorithm", [("Q_LEARNING", True), ("SARSA", False)])


class QTable(dict):
    def __init__(self, file=None):
        """
        Subclass of dict representing q-table
        Parameters:
            file: Q-table from json
        Returns: Q-table for agent
        """
        if file is not None:
            super().__init__(load(open(file, "r")))

    def get(self, state, action, default):
        return super().get(str(tuple([state.values(), action])), default)

    def set(self, state, action, value):
        self[str(tuple([state.values(), action]))] = value


class State(dict):
    def __init__(
        self,
        observation,
        _=None,
        terminated=None,
        truncated=None,
        __=None,
        precision=1e4,
    ):
        """
        Subclass of dict a agent for CartPole Problem
        Parameters:
            observation: Observation object provided by simulator
            _: Unused value from simulator object
            terminated: Check if simulation has achieved end state
            truncated: Check if simulation has achieved truncated state
            __: Unused value from simulator object
            precision: Precision in continuos to discrete values
        Returns: State for agent
        """
        self.terminated = terminated
        self.truncated = truncated
        self["position"] = int(precision * observation[0])  # [-4.8PREC, 4.8PREC]
        self["velocity"] = int(precision * observation[1])  # [-inf, +inf]
        self["angle"] = int(precision * observation[2])  # [-0.41PREC, 0.41PREC]
        self["ang_velocity"] = int(precision * observation[3])  # [-inf, +inf]


class Agent:
    ACTION_SPACE = [0, 1]

    def __init__(
        self,
        lr=0.5,
        er=0.1,
        gamma=0.9,
        alpha_er=1e3,
        file=None,
        train=True,
        algorithm=Algorithm.Q_LEARNING,
        state_precision=1e4,
    ):
        """
        Agent for CartPole problem
        Parameters:
            lr: Learning rate
            er: Exploration rate
            gamma: Bellman Gamma
            alpha_er: Weight for smoothing exploration rate function
            file: Json file for Q-table
            train: Makes agent trainable
            algorithm: Algorithm for training agent
            state_precision: Precision in continuos to discrete values
        Returns: Agent for CartPole Problem
        """
        self.trainable = train
        self.q_table = QTable(file)
        self.lr = lr
        self.er = er if self.trainable else 0
        self.alpha_er = alpha_er
        self.algorithm = algorithm
        self.state_precision = state_precision

        if not self.trainable and file is None:
            raise Exception("Non trainable agents must have a file param")

        self.gamma = gamma

    def next_action(self, state):
        """
        Parameters:
            state: State object to analyze
        Returns: Best action from state
        """
        if random() > self.er and self.trainable:
            return choice(Agent.ACTION_SPACE)

        return max(
            Agent.ACTION_SPACE,
            key=lambda action: self.q_table.get(state, action, 0),
        )

    reward = lambda self, state: (
        1 if state["angle"] == 0 else (-1 if state.terminated else 0)
    )
    reward.__doc__ = """Parameters:
        state: State object to analyze
    Returns: Reward from reward function
    """

    def update(self, old_state, action, state):
        """
        Updates agent's Q-table
        Parameters:
            old_state: State before taking action
            action: Action taken
            state: State after taking taking action
        """
        q = self.q_table.get(old_state, action, 0)

        match self.algorithm:
            case Algorithm.Q_LEARNING:
                next_q = max(
                    [self.q_table.get(state, i, 0) for i in Agent.ACTION_SPACE],
                    default=0,
                )
            case Algorithm.SARSA:
                next_q = self.q_table.get(state, self.next_action(state), 0)
            case _:
                raise Exception("Invalid algorithm for agent")

        self.q_table.set(
            state,
            action,
            q + self.lr * (self.reward(state) + self.gamma * next_q - q),
        )

    def run_episodes(self, n, seed=None, on_episode_end=None):
        """
        Run simulation for n episodes
        Parameters:
            n: Number of episodes to run
            seed: Seed for each reset
            on_episode_end: Lambda function to run after the end of each episode
        """
        self.episodes = n
        env = gym.make("CartPole-v1", render_mode="human")
        observation, _ = env.reset(seed=seed) if seed is not None else env.reset()

        self.state = State(observation, precision=self.state_precision)
        for i in range(n):
            self.current_episode = i
            action = self.next_action(self.state)

            if on_episode_end is not None:
                on_episode_end()

            old_state = self.state
            self.state = State(*env.step(action))

            if self.trainable:
                self.update(old_state, action, self.state)
                self.er = self.er * -arctan(i / (self.alpha_er * n))

            if self.state.terminated or self.state.truncated:
                observation, _ = (
                    env.reset(seed=seed) if seed is not None else env.reset()
                )

        env.close()
        self.episodes = None
        self.current_episode = None

    def episode_count(self):
        """
        Run as on_episode_end parameter to check progress of simulation
        Returns: String showing simulation progress
        """
        if not all([self.episodes, self.current_episode]):
            return "Not running"

        return f"{self.current_episode} / {self.episodes} {int(self.current_episode/self.episodes * 100)}%"

    save = lambda self, file: dump(dict(self.q_table), open(file, "w"))
    save.__doc__ = """Dumps Q-table to json file
    Parameters:
        file: File directory to dump json to
    """
