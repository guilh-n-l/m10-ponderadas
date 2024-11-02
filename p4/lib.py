import gymnasium as gym
from numpy import arctan, pi
from random import random, choice
from json import load, dump
from enum import Enum


Algorithm = Enum("Algorithm", [("Q_LEARNING", True), ("SARSA", False)])


class QTable(dict):
    def __init__(self, file=None):
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
        if random() > self.er and self.trainable:
            return choice(Agent.ACTION_SPACE)

        return max(
            Agent.ACTION_SPACE,
            key=lambda action: self.q_table.get(state, action, 0),
        )

    reward = lambda self, state: (
        1 if state["angle"] == 0 else (-1 if state.terminated else 0)
    )

    def update(self, old_state, action, state):
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
        if not all([self.episodes, self.current_episode]):
            return "Not running"

        return f"{self.current_episode} / {self.episodes} {int(self.current_episode/self.episodes * 100)}%"

    save = lambda self, file: dump(dict(self.q_table), open(file, "w"))
