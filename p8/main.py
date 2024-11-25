from keras.layers import Dense, Input
from keras.models import Sequential, clone_model
from numpy import argmax, array
from random import random, sample
from enum import Enum
import gymnasium as gym

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
    observation: {[format(i, ".2f") for i in self.observation]}
)"""


class DeepQNet:
    def __init__(
        self,
        input_shape=(8,),
        n_dense=3,
        dense_sizes=[64, 32],
        activation="relu",
        output_size=4,
        output_activation="linear",
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["accuracy"],
        model=None,
    ):
        """
        Initializes a Deep Q-Network (DQN) model. This network learns to estimate Q-values for each possible action in a given state

        Parameters:
            input_shape (8,): The shape of the input state.
            n_dense (3): The number of dense layers.
            dense_sizes (64, 32): Sizes of the dense layers (default is a list of sizes decreasing by powers of 2).
            activation ('relu'): The activation function for the hidden layers.
            output_size (4): The number of actions.
            output_activation ('linear'): The activation function for the output layer.
            optimizer ('adam'): The optimizer used for training.
            loss ('mean_squared_error'): The loss function used for training.
            metrics (['accuracy']): The metrics used for evaluating the model during training.
            model (None): An existing model to copy.
        """
        if model is not None:
            self.net = DeepQNet._copy_model(model)
            return
        self.net = DeepQNet._get_model(
            input_shape,
            n_dense,
            dense_sizes,
            activation,
            output_size,
            output_activation,
            optimizer,
            loss,
            metrics,
        )
        self.compile_params = {"optimizer": optimizer, "loss": loss, "metrics": metrics}

    def _get_model(
        input_shape,
        n_dense,
        dense_sizes,
        activation,
        output_size,
        output_activation,
        optimizer,
        loss,
        metrics,
    ):
        """
        Parameters:
            input_shape: The shape of the input.
            n_dense: Number of dense layers.
            dense_sizes: List specifying the size of each dense layer.
            activation: The activation function used for hidden layers.
            output_size: The number of output neurons.
            output_activation: The activation function for the output layer.
            optimizer: The optimizer used to compile the model.
            loss: The loss function used to compile the model.
            metrics: Metrics to evaluate the model during training.

        Returns:
            A Keras Sequential model.
        """
        if len(dense_sizes) != n_dense - 1:
            raise ValueError("dense_sizes list length must equal n_dense - 1")

        model = Sequential()

        model.add(Input(shape=input_shape))
        for i in range(n_dense - 1):
            model.add(Dense(dense_sizes[i], activation=activation))
        model.add(Dense(output_size, activation=output_activation))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model

    def _copy_model(model):
        """
        Copies the architecture from another DeepQNet model.

        Parameters:
            model: The model to copy.

        Returns:
            A new DeepQNet instance with the same architecture.
        """
        n_model = clone_model(model.net)
        n_model.compile(**model.compile_params)
        return n_model

    copy_weights = lambda self, model: self.net.set_weights(model.net.get_weights())

    copy_weights.__doc__ = """
        Copies the weights from another model to this one.

        Parameters:
            model: The model to copy weights from.
    """


def env_reset_if_terminated(state):
    """
    Resets the environment based on the status of termination or truncated from state

    Parameters:
        state: State to check whether terminated/truncated or not
    Returns:
        If environment was terminated or truncated
    """
    if state.terminated or state.truncated:
        global env
        env.reset()
        return True
    return False


def get_env(seed=None, human=False):
    """
    Initializes and returns the environment for the LunarLander-v3 task.

    This function checks if the environment has already been created, and if not, it creates it.
    It will reset the environment with the provided seed if specified.

    Parameters:
        seed (None): A random seed for initializing the environment's state. If not provided, the environment will be initialized with a random seed.

    Returns:
        gym.Env: The initialized gym environment for the "LunarLander-v3" task.
    """

    global env
    if env is None:
        env = (
            gym.make("LunarLander-v3", render_mode="human")
            if human
            else gym.make("LunarLander-v3")
        )
        env.reset() if seed is None else env.reset(seed)
        return env
    return env


class Agent:
    def __init__(
        self,
        neural_net,
        max_buffer_size=1e6,
        batch_size=100,
        update_target=1e3,
        epsilon_0=0.3,
        gamma=0.1,
        alpha=0.5,
        on_episode_end=None,
        on_episode_end_args=None,
    ):
        self.neural_net = neural_net
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size if type(batch_size) is int else int(batch_size)
        self.update_target = update_target
        self.epsilon_0 = epsilon_0
        self.gamma = gamma
        self.alpha = alpha
        self.rewards = []
        self.loss = []
        self.accuracy = []
        self.n_episodes = 0
        self.on_episode_end = on_episode_end
        self.on_episode_end_args = on_episode_end_args

    def run_n_steps(self, n_steps, seed=None, human=False):
        self.n_episodes = 0

        def epsilon_greedy(step, n_steps, state):
            """
            Selects an action based on the epsilon-greedy policy.

            Parameters:
                step: The current step for the whole execution.
                n_steps: The total number of steps in the whole execution.
                model: The Deep Q-Network model used to predict Q-values.
                state: The current state of the environment.

            Returns:
                action: The selected action (either based on Q-values or random exploration).
            """

            def is_greedy(step, n_steps):
                """
                Determines if the agent should choose a greedy action or explore.

                Parameters:
                    step: The current step for the whole execution.
                    n_steps: The total number of steps in the whole execution.

                Returns:
                    if the agent should choose a greedy action
                """
                if random() < max(
                    -(1.25 * self.epsilon_0 / n_steps) * step + self.epsilon_0, 0
                ):
                    return True
                return False

            if is_greedy(step, n_steps):
                return argmax(
                    self.neural_net.net.predict(
                        state.observation.reshape(1, -1), verbose=0
                    )
                )
            else:
                global env
                return env.action_space.sample()

        def update_buffer(buf, uple_5, step):
            """
            Updates the buffer with a new experience and ensures its size does not exceed the maximum buffer size.

            Parameters:
                buf: The experience replay buffer.
                uple_5: A list containing the current state, action, reward, next state, and done flag.
                step: The current step in the execution.
            """
            buf.append(uple_5)

            if len(buf) > self.max_buffer_size:
                buf.pop(0)

            if step > 0:
                if len(buf) < 2:
                    raise ValueError("Max buffer size must be greater than 1")

                buf[-2][3] = state.observation

        def train_model(buf, state):
            """
            Trains the model using a mini-batch sampled from the experience replay buffer.

            Parameters:
                buf: The experience replay buffer containing past experiences.
                state: The current state of the environment.
            """
            mini_batch = sample(buf, self.batch_size)
            states = []
            targets = []

            for state, action, reward, next_state, done in mini_batch:
                if next_state is None:
                    continue

                q_values = array(
                    self.neural_net.net.predict(state.reshape(1, -1), verbose=0)
                ).flatten()

                self.rewards.append(reward)
                target = reward

                if not done:
                    next_q_values = target_net.net.predict(
                        next_state.reshape(1, -1), verbose=0
                    )
                    target = reward + self.gamma * max(next_q_values[0])

                q_values[action] = target

                states.append(state)
                targets.append(q_values)

            acc, loss = array(
                list(
                    self.neural_net.net.fit(
                        array(states), array(targets), epochs=1, verbose=0
                    ).history.values()
                )
            ).flatten()
            self.accuracy.append(acc)
            self.loss.append(loss)

        env = get_env(seed=seed, human=human)
        target_net = DeepQNet(model=self.neural_net)

        buffer = []

        action = env.action_space.sample()

        for i in range(n_steps):
            state = State(*env.step(action))
            action = epsilon_greedy(i, n_steps, state)
            update_buffer(
                buffer,
                [
                    state.observation,
                    action,
                    state.reward,
                    None,
                    state.terminated or state.truncated,
                ],
                i,
            )

            if env_reset_if_terminated(state):
                if self.on_episode_end:
                    (
                        self.on_episode_end(*self.on_episode_end_args)
                        if self.on_episode_end_args
                        else self.on_episode_end()
                    )
                self.n_episodes += 1

            if len(buffer) >= self.batch_size:
                train_model(buffer, state)

            if i % self.update_target == 0:
                target_net.copy_weights(self.neural_net)

            print(f"Step {i}")
        return self.n_episodes


# def main():
#     get_env(human=True)
#     agent_net = DeepQNet(n_dense=5, dense_sizes=[64, 32, 32, 16])
#     ag = Agent(agent_net, on_episode_end=print, on_episode_end_args=["acabei"])
#     ag.run_n_steps(int(1e12), human=True)
#
#
# if __name__ == "__main__":
#     main()
