import gymnasium as gym
from numpy import arctan, pi
from random import random, choice


class QTable(dict):
    def get(self, state, action, default):
        return super().get(tuple([state.values(), action]), default)

    def set(self, state, action, value):
        self[tuple([state.values(), action])] = value


class State(dict):
    def __init__(self, observation, _=None, terminated=None, truncated=None, __=None):
        self.terminated = terminated
        self.truncated = truncated
        self["position"] = int(100 * observation[0])  # [-480, 480]
        self["velocity"] = int(100 * observation[1])  # [-inf, +inf]
        self["angle"] = int(100 * observation[2])  # [-41, 41]
        self["ang_velocity"] = int(100 * observation[3])  # [-inf, +inf]


class Agent:
    ACTION_SPACE = [0, 1]

    def __init__(self, lr, er, gamma, alpha_er=1e3):
        self.q_table = QTable()
        self.lr = lr
        self.er = er
        self.alpha_er = alpha_er

        self.gamma = gamma

    def next_action(self, state):
        if random() > self.er:
            return choice(Agent.ACTION_SPACE)
        return max(
            Agent.ACTION_SPACE,
            key=lambda action: self.q_table.get(state, action, random()),
        )

    reward = lambda self, state: (
        1 if state["angle"] == 0 else (-1 if state.terminated else 0)
    )

    def update(self, old_state, action, state):
        q = self.q_table.get(old_state, action, 0)
        max_q_pred = max(
            [self.q_table.get(state, i, 0) for i in Agent.ACTION_SPACE], default=0
        )
        self.q_table.set(
            state,
            action,
            q + self.lr * (self.reward(state) + self.gamma * max_q_pred - q),
        )

    def run_episodes(self, n, seed=None, on_episode_end=None):
        env = gym.make("CartPole-v1", render_mode="human")
        observation, _ = env.reset(seed=seed) if seed is not None else env.reset()
        state = State(observation)
        for i in range(n):
            action = self.next_action(state)
            old_state = state
            state = State(*env.step(action))
            self.update(old_state, action, state)

            if state.terminated or state.truncated:
                observation, _ = (
                    env.reset(seed=seed) if seed is not None else env.reset()
                )
            self.er = (self.er / (pi / 2)) * (
                -arctan((1 / (self.alpha_er * n)) * i) + (pi / 2)
            )
            if on_episode_end is not None:
                on_episode_end()
        env.close()


if __name__ == "__main__":
    agent = Agent(0.5, 0.1, 0.9)
    agent.run_episodes(int(1e5), seed=20, on_episode_end=lambda: print(agent.er))
