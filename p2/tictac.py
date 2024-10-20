from sys import maxsize as MAX_INTEGER
from random import choice, random

WINNING_BOARDS = [
    set([0, 1, 2]),
    set([3, 4, 5]),
    set([6, 7, 8]),
    set([0, 3, 6]),
    set([1, 4, 7]),
    set([2, 5, 8]),
    set([0, 4, 8]),
    set([2, 4, 6]),
]

get_available = lambda board: [idx for idx, i in enumerate(board) if i is None]

get_available.__doc__ = """Parameters:
    board: TicTacToeGame board
Returns: List of available squares
"""


def replace_(list_, i, value):
    """
    Parameters:
        list_: List to replace value from
        i: index of value to replace
        element: value to replace old value with
    Returns: New list with value replaced
    """
    new_list = list_.copy()
    new_list[i] = value
    return new_list


check_if_won = lambda isP1, board: (
    isP1
    if any(
        i.issubset(set([idx for idx, i in enumerate(board) if i is isP1]))
        for i in WINNING_BOARDS
    )
    else None
)

check_if_won.__doc__ = """Parameters:
    isP1: True if P1's turn
    board: TicTacToeGame board to check if game was already won
Returns: True if won by P1, False if won by P2, if no one wins, None
"""

check_if_double_win = lambda isP1, board: (
    len(
        [
            True
            for i in get_available(board)
            if check_if_won(isP1, replace_(board, i, isP1)) is not None
        ]
    )
    > 1
)

check_if_won.__doc__ = """Parameters:
    isP1: True if P1's turn
    board: TicTacToeGame board to check if game had double win
Returns: True if player will win for sure in their next turn
"""


def hamming(isP1, board, depth=0):
    """
    Parameters:
        isP1: True if P1's turn
        board: TicTacToeGame board to check distance to winning state
        depth: Recursion depth
    Returns: Distance to winning state (1-3, 3 if game will tie)
    """
    min_hamming = MAX_INTEGER
    for square in get_available(board):
        new_board = board.copy()
        new_board[square] = isP1
        if check_if_won(isP1, new_board) is not None:
            min_hamming = min(min_hamming, 1)
        if depth == 0 and hamming(isP1, new_board, depth=1) == 1:
            min_hamming = min(min_hamming, 2)
    return min(min_hamming, 3)


STRING_MAPPER = {True: "X", False: "O", None: "*"}
board_string = (
    lambda board: f"""{STRING_MAPPER[board[0]]} {STRING_MAPPER[board[1]]} {STRING_MAPPER[board[2]]}
{STRING_MAPPER[board[3]]} {STRING_MAPPER[board[4]]} {STRING_MAPPER[board[5]]}
{STRING_MAPPER[board[6]]} {STRING_MAPPER[board[7]]} {STRING_MAPPER[board[8]]}
"""
)

board_string.__doc__ = """Parameters:
    board: TicTacToeGame board to turn into readable string
Returns: Human readable string of the board
"""


def get_best_move(isP1, board):
    """
    Parameters:
        isP1: True if P1's turn
        board: TicTacToeGame board to get best move from
    Returns: Best square that player can play
    """
    available = get_available(board)
    points = [calculate_move_points(isP1, board, square) for square in available]
    return available[points.index(max(points))]


def calculate_move_points(isP1, board, selected):
    """
    Parameters:
        isP1: True if P1's turn
        board: TicTacToeGame board to get best move from
        selected: Selected square to calculate move points from
    Returns: Points given by selected square
    """
    if board[selected] is not None:
        raise Exception("Selected invalid square")

    def is_blocking(isP1, board, selected):
        new_board = board.copy()
        new_board[selected] = not isP1

        return check_if_won(not isP1, new_board) is not None

    new_board = board.copy()
    new_board[selected] = isP1
    if check_if_won(isP1, new_board) is not None:
        return MAX_INTEGER
    if check_if_double_win(isP1, new_board):
        return 20
    if is_blocking(isP1, board, selected):
        return 10
    if hamming(not isP1, new_board) == 1:
        return -MAX_INTEGER
    return 4 - hamming(isP1, new_board)


class TicTacToeGame:
    def __init__(self, multiplayer=True, bot_level=1.0):
        """
        TicTacToe Game
        Parameters:
            multiplayer(True): If game mode is multiplayer
            bot_level(1.0): Level of the bot to play in single player mode (.0 to 1.)
        Returns: TicTacToeGame object
        """
        self.board = [None] * 9
        self.isP1 = choice([True, False])
        self.winner = None
        self.ended = False
        self.multiplayer = multiplayer
        self.history = [board_string(self.board)]
        self.bot_level = bot_level

        if not self.isP1 and not self.multiplayer:
            self.choose_square(
                get_best_move(False, self.board)
                if random() < bot_level
                else choice(get_available(self.board))
            )

    def play_round(self, square):
        """
        Parameters:
            square: Square to play in current game board
        Returns: True if the play was made, False if it failed
        Note: If multiplayer is True, each round is 1 function call, else it is 1 function call for each player
        """
        if not self.ended:
            self.choose_square(square)
            if not self.multiplayer and None in self.board:
                self.choose_square(
                    get_best_move(False, self.board)
                    if random() < self.bot_level
                    else choice(get_available(self.board))
                )
            return True
        return False

    def choose_square(self, square):
        """
        Parameters:
            square: Square to play in current game board
        """
        if self.board[square] is not None:
            raise Exception("Selected invalid square")

        self.board[square] = self.isP1
        self.history.append(board_string(self.board))

        if check_if_won(self.isP1, self.board) is not None:
            self.winner = self.isP1
            self.ended = True
            return

        if None not in self.board:
            self.winner = None
            self.ended = True
            return

        self.isP1 = not self.isP1
