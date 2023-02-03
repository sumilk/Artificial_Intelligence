from abc import ABC, abstractmethod


class GameState(ABC):
    """Abstract class for representing game state.
    Concrete subclasses will define the state for specific games.
    """

    @abstractmethod
    def update(self, move):
        pass

    @abstractmethod
    def is_win(self):
        pass

    @abstractmethod
    def is_draw(self):
        pass

    @abstractmethod
    def available_moves(self):
        pass

    @abstractmethod
    def clone(self):
        pass


class PacmanState(GameState):
    """Concrete implementation of the game state for Pacman."""

    def __init__(self, board, pacman_position, exit_position, red_pellets, green_pellets):
        self.board = board
        self.pacman_position = pacman_position
        self.exit_position = exit_position
        self.red_pellets = red_pellets
        self.green_pellets = green_pellets

    def update(self, move):
        pass

    def is_win(self):
        pass

    def is_draw(self):
        pass

    def available_moves(self):
        pass

    def clone(self):
        pass


class TicTacToeState(GameState):
    """Concrete implementation of the game state for Tic Tac Toe."""

    def __init__(self, board):
        self.board = board

    def update(self, move):
        pass

    def is_win(self):
        pass

    def is_draw(self):
        pass

    def available_moves(self):
        pass

    def clone(self):
        pass


class GameSolver(ABC):
    """Abstract class for solving a game.
    Concrete subclasses will implement the solving algorithm for specific games.
    """

    @abstractmethod
    def solve(self, game_state):
        pass


class AStarSolver(GameSolver):
    """Concrete implementation of the solving algorithm for games like Pacman using A*."""

    def __init__(self, heuristic):
        self.heuristic = heuristic

    def solve(self, game_state):
        pass


class MinMaxSolver(GameSolver):
    """Concrete implementation of the solving algorithm for games like Tic Tac Toe using MinMax."""

    def solve(self, game_state):
        pass


class GameObserver(ABC):
    """Abstract class for observing a game.
    Concrete subclasses will define the observer for specific games.
    """

    @abstractmethod
    def update(self, game_state):
        pass


class ConsoleObserver(GameObserver):
    """Concrete implementation of the observer for displaying the game state in the console."""

    def update(self, game_state):
        print(game_state.board)


class GameManager:
    def __init__(self, game_state, game_solver, game_observer):
        self.game_state = game_state
        self.game_solver = game_solver
        self.game_observer = game_observer

    def play(self):
        while not self.game_state.is_win() and not self.game_state.is_draw():
            self.game_observer.update(self.game_state)
            move = self.game_solver.solve(self.game_state)
            self.game_state.update(move)
        self.game_observer.update(self.game_state)
        if self.game_state.is_win():
            print("Player wins!")
        elif self.game_state.is_draw():
            print("Draw!")