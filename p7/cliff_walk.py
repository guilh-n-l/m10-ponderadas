from enum import Enum

Action = Enum("Action", [("UP", 0), ("RIGHT", 1), ("DOWN", 2), ("LEFT", 3)])


class CliffWalk:
    def __init__(self):
        """
        Initializes the CliffWalk environment with a 4x12 grid.
        Agent starts at position 36, and rewards are set for cliffs and the goal.
        """
        self.array = [0] * 48
        self.rewards = [-1] * 48
        for i in range(37, 48):
            self.rewards[i] = -100 if i < 47 else 100
        self.position = 36
        self.terminated = False

    def reset(self):
        """
        Resets the agent to the starting position
        """
        self.position = 36
        self.terminated = False

    def walk(self, action):
        """
        Moves the agent based on the action (UP, RIGHT, DOWN, LEFT).
        Ends the episode if the agent falls into a cliff or reaches the goal.
        The player cannot enter hole squares, nor can they clip into walls.

        Parameters:
            action (Action): The action to take (UP, RIGHT, DOWN, LEFT).

        Raises:
            ValueError: If the action is invalid.
        """
        if action not in [i for i in Action]:
            raise ValueError("Invalid action")

        p = self.position
        a = action.value
        match p:
            case p if p <= 11:
                arr = [p, p if p == 11 else p + 1, p + 12, p if p == 0 else p - 1]
            case p if p <= 23:
                arr = [p - 12, p if p == 23 else p + 1, p + 12, p if p == 12 else p - 1]
            case p if p <= 35:
                arr = [p - 12, p if p == 35 else p + 1, p + 12, p if p == 24 else p - 1]
            case _:
                arr = [p - 12, p if p == 47 else p + 1, p, p if p == 36 else p - 1]

        in_hole = arr[a] in [i for i in range(37, 47)]
        self.position = p if in_hole else arr[a]
        self.terminated = in_hole or arr[a] == 47

    def print_board(self):
        """
        Prints the current board with agent's position, cliff, and goal.
        """
        for i in range(len(self.array)):
            print(
                (
                    (("[ ]" if i != 47 else "[$]") if i not in range(37, 47) else "[#]")
                    if i != self.position
                    else "[*]"
                ),
                end=" " if (i + 1) % 12 != 0 else "\n",
            )
        print("0: Move up | 1: Move right | 2: Move down | 3: Move left")


if __name__ == "__main__":
    cw = CliffWalk()
    cw.print_board()
    while True:
        action = Action(int(input()))
        cw.walk(action)
        cw.print_board()
        print(cw.terminated)
