import gymnasium as gym
from numpy import argmax, exp, max as nmax, array
from numpy.random import choice, rand
from math import inf
from enum import Enum

env = None

Environ = Enum(
    "Environ",
    [
        ("STATE", 0),
        ("REWARD", 1),
        ("TERMINATED", 2),
        ("TRUNCATED", 3),
        ("INFO", 4),
        ("OBSERVATION", 5),
    ],
)


class Policy(Enum):
    GREEDY = 0
    SOFTMAX = 1
    EPS_GREEDY = 2

    _invalid_policy_exception = lambda cls: ValueError("Policy not recognized")

    def init(self, **kwargs):
        """
        Initializes the policy with the given parameters.

        Parameters:
            **kwargs: Policy-specific parameters (epsilon for EPS_GREEDY).

        Returns:
            The initialized policy object.
        """
        match self:
            case Policy.EPS_GREEDY:
                self.epsilon = kwargs.get("epsilon", 0.1)
            case Policy.SOFTMAX:
                pass
            case Policy.GREEDY:
                pass
            case _:
                raise Policy._invalid_policy_exception()
        return self


class State:
    def __init__(self, obs, reward, term, trunc, info):
        """
        Initializes the state

        Parameters:
            obs: Observation from the environment.
            reward: The reward from the environment.
            term: Boolean indicating whether the episode is terminated.
            trunc: Boolean indicating whether the episode is truncated.
            info: Additional info returned by the environment.
        """
        self.reward = reward
        self.terminated = term
        self.truncated = trunc
        self.info = info
        self.observation = obs

    def __str__(self):
        """
        Returns a string representation of the state.
        """
        return f"""State(
    reward: {self.reward}
    terminated: {self.terminated}
    truncated: {self.truncated}
    info: {self.info}
    observation: {self.observation}
)"""


class Algorithm(Enum):
    TD = True

    _invalid_algorithm_exception = lambda cls: ValueError("Algorithm not recognized")

    def init(self, **kwargs):
        """
        Initializes algorithm

        Parameters:
            **kwargs: Algorithm-specific parameters (n, alpha, gamma, seed).
        """
        match self:
            case Algorithm.TD:
                self.n = kwargs.get("n", 0) + 1
                self.alpha = kwargs.get("alpha", 0.1)
                self.gamma = kwargs.get("gamma", 0.9)
                self.seed = kwargs.get("seed", None)
                self.policy_type = kwargs.get("p_type", Policy.GREEDY)
                self.v_table = [0] * 48
                self.states = []
            case _:
                raise Algorithm._invalid_algorithm_exception()
        return self

    def _run_step_end(self, on_step_end, on_step_end_args):
        """
        Executes a callback at the end of a step.

        Parameters:
            on_step_end: Callback function to run at the end of each step.
            on_step_end_params: Parameters passed to the callback function.
        """
        if on_step_end is not None:
            environ_vars = [
                Environ.STATE,
                Environ.REWARD,
                Environ.TERMINATED,
                Environ.TRUNCATED,
                Environ.INFO,
                Environ.OBSERVATION,
            ]
            environ_replacements = [
                self.states[-1],
                self.states[-1].reward,
                self.states[-1].terminated,
                self.states[-1].truncated,
                self.states[-1].info,
                self.states[-1].observation,
            ]
            args = []
            match on_step_end_args:
                case None:
                    on_step_end()
                case on_step_end_args if any(
                    item in environ_vars for item in on_step_end_args
                ):
                    args = [
                        (
                            environ_replacements[environ_vars.index(i)]
                            if i in environ_vars
                            else i
                        )
                        for i in on_step_end_args
                    ]
                    on_step_end(*args)
                case _:
                    on_step_end(*on_step_end_args)

    def _env_reset_if_terminated(self, state):
        """
        Resets the environment if the episode has terminated or been truncated.

        Parameters:
            state: Current state of the environment.

        Returns:
            True if the environment was reset, False otherwise.
        """
        if state.terminated or state.truncated:
            global env
            env.reset() if self.seed is None else env.reset(seed=self.seed)
            return True
        return False

    def _temporal_difference(self, n_steps, on_step_end, on_step_end_args):
        """
        Runs the Temporal Difference (TD) algorithm for the specified number of steps.

        Parameters:
            n_steps: Number of steps to run the algorithm for.
            on_step_end: Callback function to run at the end of each step.
            on_step_end_params: Parameters passed to the callback function.
        """

        def update_v_table(state, next_state):
            """
            Updates the value table (v_table) based on the Temporal Difference (TD) rule.

            Parameters:
                state: The current state.
                next_state: The next state in the sequence.
            """
            self.v_table[next_state] += self.alpha * (
                (
                    sum([self.gamma**i * rewards[i] for i in range(len(rewards))])
                    + self.gamma**self.n * self.v_table[next_state]
                )
                - self.v_table[next_state]
            )

        def update_rewards(state):
            """
            Updates the reward list by appending the current reward while maintaining a fixed size.

            Parameters:
                state: The current state with its reward.
            """
            if len(rewards) >= self.n:
                rewards.pop(0)
            rewards.append(state.reward)

        def run_next_step(action, next_state):
            """
            Runs a single step in the environment, updates the state, and handles termination.

            Parameters:
                action: The action to take.
                next_state: The state to transition to.

            Returns:
                The updated state after taking the action.
            """
            state = State(*env.step(action))
            self._env_reset_if_terminated(state)

            if state.observation == 47:
                state.reward = 100

            if len(self.states) >= self.n:
                self.states.pop(0)

            self.states.append(state)

            if len(next_states) >= self.n:
                next_states.pop(0)

            next_states.append(next_state)

            update_rewards(state)

            self._run_step_end(on_step_end, on_step_end_args)
            return state

        def choose_action(state):
            """
            Chooses the best action based on the current state using the algorithm's policy.

            Parameters:
                state: The current state of the environment.

            Returns:
                action: The chosen action.
                next_state: The next state after taking the action.
            """

            def softmax(arr):
                """
                Applies the softmax function to an array of values.

                Parameters:
                    arr: Array of values.

                Returns:
                    ndarray: The softmax probabilities for each action.
                """
                e = exp(arr - nmax(arr))
                return e / e.sum()

            obs = 36 if state is None else state.observation
            match obs:
                case obs if obs <= 11:
                    adjacent = [
                        obs,
                        obs if obs == 11 else obs + 1,
                        obs + 12,
                        obs if obs == 0 else obs - 1,
                    ]
                case obs if obs <= 23:
                    adjacent = [
                        obs - 12,
                        obs if obs == 23 else obs + 1,
                        obs + 12,
                        obs if obs == 12 else obs - 1,
                    ]
                case obs if obs <= 35:
                    adjacent = [
                        obs - 12,
                        obs if obs == 35 else obs + 1,
                        obs + 12,
                        obs if obs == 24 else obs - 1,
                    ]
                case _:
                    adjacent = [
                        obs - 12,
                        obs if obs + 1 == 48 else obs + 1,
                        obs,
                        obs if obs - 1 == 35 else obs - 1,
                    ]
            values = [self.v_table[i] if i != obs else -inf for i in adjacent]
            match self.policy_type:
                case Policy.GREEDY:
                    act = (
                        adjacent.index(adjacent[argmax(values)])
                        if len(set(values)) != 1
                        else env.action_space.sample()
                    )
                case Policy.SOFTMAX:
                    act = choice(
                        array(list(range(4))),
                        p=softmax(array([self.v_table[i] for i in adjacent])),
                    )
                case Policy.EPS_GREEDY:
                    if rand() < self.policy_type.epsilon:
                        act = choice(array(list(range(4))))
                    else:
                        act = (
                            adjacent.index(adjacent[argmax(values)])
                            if len(set(values)) != 1
                            else env.action_space.sample()
                        )
                case _:
                    raise Policy._invalid_policy_exception()
            return act, adjacent[act]

        rewards = []
        next_states = []
        step = 0
        action, next_state = choose_action(None)
        state = None

        while step < self.n - 1:
            state = run_next_step(action, next_state)
            action, next_state = choose_action(state)
            step += 1

        while step < int(n_steps):
            state = run_next_step(action, next_state)
            update_v_table(self.states[-self.n], next_states[-self.n])
            action, next_state = choose_action(state)
            step += 1

    def run(self, n_steps, on_step_end=None, on_step_end_args=None):
        """
        Runs the selected algorithm for a given number of steps.

        Parameters:
            n_steps: Number of steps to run the algorithm.
            on_step_end: Optional callback to be executed at the end of each step.
            on_step_end_params: Parameters to be passed to the callback function.
        """
        match self:
            case Algorithm.TD:
                self._temporal_difference(n_steps, on_step_end, on_step_end_args)
                self.states = []
            case _:
                raise Algorithm._invalid_algorithm_exception()

    def print_table(self):
        """
        Prints the value table (v_table) of the algorithm.
        This will display the current values for each state in the grid.
        """
        match self:
            case Algorithm.TD:
                for i in range(len(self.v_table)):
                    print(self.v_table[i], end="\n" if (i + 1) % 12 == 0 else " ")
                print()
            case _:
                raise Algorithm._invalid_algorithm_exception()

    def __str__(self):
        """
        Returns a string representation of the algorithm.
        """
        str = ""
        match self:
            case Algorithm.TD:
                str += "Algorithm<TEMPORAL DIFFERENCE>("
                str += f"n={self.n - 1} "
                str += f"alpha={self.alpha} "
                str += f"gamma={self.gamma} "
                str += f"seed={self.seed} "
                str += f"policy_type={self.policy_type})"
                return str
            case _:
                raise Algorithm._invalid_algorithm_exception()


def get_env(seed=None, human=False):
    """
    Initializes and returns the environment for the CliffWalking-v0 task.

    This function checks if the environment has already been created, and if not, it creates it.
    It will reset the environment with the provided seed if specified.

    Parameters:
        seed (optional): A random seed for initializing the environment's state. If not provided, the environment will be initialized with a random seed.

    Returns:
        gym.Env: The initialized gym environment for the "CliffWalking-v0" task.
    """
    global env
    if env is None:
        env = (
            gym.make("CliffWalking-v0", render_mode="human")
            if human
            else gym.make("CliffWalking-v0")
        )
        env.reset() if seed is None else env.reset(seed)
        return env
    return env


def track_episode_reward(step_reward_list, state, rewards):
    """
    Tracks and updates the cumulative reward for an episode.

    This function appends the reward of each step to `step_reward_list` during the episode.
    When the episode ends (terminated or truncated), it sums the rewards, adds the total
    to `rewards`, and clears the `step_reward_list` for the next episode.

    Parameters:
        step_reward_list: A deque that accumulates rewards for the current episode.
        state (State): The current state, containing reward and termination info.
        rewards: List to store total rewards for each completed episode.
    """
    if state.terminated or state.truncated:
        rewards.append(sum(step_reward_list) + state.reward)
        step_reward_list.clear()

    else:
        step_reward_list.append(state.reward)


def main():
    env = get_env(human=True)
    # env = get_env()

    alg = Algorithm.TD.init(n=10, gamma=0.2, p_type=Policy.GREEDY)
    print(alg)
    step_reward_list = []
    reward_list = []
    alg.run(
        n_steps=1e10,
        on_step_end=track_episode_reward,
        on_step_end_args=[step_reward_list, Environ.STATE, reward_list],
    )
    print(reward_list)


if __name__ == "__main__":
    main()
