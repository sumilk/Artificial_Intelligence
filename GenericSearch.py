from abc import ABC, abstractmethod

from queue import PriorityQueue
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
        pacman_row, pacman_col = self.pacman_position
        if move == "up":
            pacman_row -= 1
        elif move == "down":
            pacman_row += 1
        elif move == "left":
            pacman_col -= 1
        elif move == "right":
            pacman_col += 1
        else:
            raise ValueError("Invalid move")

        if pacman_row < 0 or pacman_row >= len(self.board) or \
           pacman_col < 0 or pacman_col >= len(self.board[0]):
            raise ValueError("Move is out of bounds")

        if self.board[pacman_row][pacman_col] == "#":
            raise ValueError("Move is blocked")

        if (pacman_row, pacman_col) in self.red_pellets:
            self.red_pellets.remove((pacman_row, pacman_col))

        self.pacman_position = (pacman_row, pacman_col)

    def is_win(self):
        return self.pacman_position == self.exit_position

    def is_draw(self):
        return len(self.red_pellets) == 0

    def available_moves(self):
        moves = ["up", "down", "left", "right"]
        pacman_row, pacman_col = self.pacman_position
        if pacman_row == 0:
            moves.remove("up")
        if pacman_row == len(self.board) - 1:
            moves.remove("down")
        if pacman_col == 0:
            moves.remove("left")
        if pacman_col == len(self.board[0]) - 1:
            moves.remove("right")
        return moves

    def clone(self):
        return PacmanState(self.board, self.pacman_position, self.exit_position,
                           self.red_pellets, self.green_pellets)


class TicTacToeState(GameState):
    def __init__(self, n: int, board=None):
        self.n = n
        self.board = board if board is not None else [[0] * n for _ in range(n)]

        self.turn = 0

    def update(self, move):
        x, y = move
        if self.board[x][y] is not None:
            raise Exception("Invalid move")
        self.board[x][y] = self.turn
        self.turn = (self.turn + 1) % 2

    def is_win(self):
        # check rows
        for i in range(self.n):
            if all(self.board[i][j] == self.turn for j in range(self.n)):
                return True
        # check columns
        for j in range(self.n):
            if all(self.board[i][j] == self.turn for i in range(self.n)):
                return True
        # check diagonals
        if all(self.board[i][i] == self.turn for i in range(self.n)):
            return True
        if all(self.board[i][self.n-i-1] == self.turn for i in range(self.n)):
            return True
        return False

    def is_draw(self):
        return all(all(cell is not None for cell in row) for row in self.board) and not self.is_win()

    def available_moves(self):
        moves = []
        for i in range(self.n):
            for j in range(self.n):
                if self.board[i][j] is None:
                    moves.append((i, j))
        return moves

    def clone(self):
        return TicTacToeState(self.n, [row[:] for row in self.board])



class GameSolver(ABC):
    """Abstract class for solving a game.
    Concrete subclasses will implement the solving algorithm for specific games.
    """

    @abstractmethod
    def solve(self, game_state):
        pass



class AStarSolver(GameSolver):
    """A* search algorithm for solving game problems like pacman"""
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def solve(self, initial_state):
        """Solve the game problem using A* search algorithm.

        Args:
        initial_state: the initial state of the game.

        Returns:
        A list of moves that leads to the solution.
        """
        priority_queue = PriorityQueue()
        priority_queue.put((0, initial_state, []))
        visited = set()

        while not priority_queue.empty():
            (cost, current_state, path) = priority_queue.get()
            if current_state.is_win():
                return path
            if current_state in visited:
                continue
            visited.add(current_state)
            for next_state in current_state.next_states():
                heuristic_cost = self.heuristic(next_state)
                priority_queue.put((cost + heuristic_cost, next_state, path + [next_state.last_move()]))



class MinMaxSolver(GameSolver):
    """Concrete implementation of the solving algorithm for games like Tic Tac Toe using MinMax."""
    def __init__(self, max_depth, player_id):
        self.max_depth = max_depth
        self.player_id = player_id

    def solve(self, state):
        def min_max(state: GameState, depth, player_id):
            if state.is_win():
                return (None, -1e6) if state.is_win() == self.player_id else (None, 1e6)
            if depth == self.max_depth or state.is_draw():
                return (None, 0)

            best_move = None
            best_score = -1e6 if player_id == self.player_id else 1e6

            for move in state.available_moves():
                new_state = state.clone()
                new_state.update(move)
                _, score = min_max(new_state, depth + 1, 3 - player_id)
                if player_id == self.player_id:
                    if score > best_score:
                        best_move = move
                        best_score = score
                else:
                    if score < best_score:
                        best_move = move
                        best_score = score

            return (best_move, best_score)

        move, _ = min_max(state, 0, self.player_id)
        return move




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



class GameStateNode:
    """Class for representing a node in the state space tree for a game state."""

    def __init__(self, game_state, parent=None):
        self.game_state = game_state
        self.parent = parent
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)


class GameManager:
    """Abstract class for managing a game.
    Concrete subclasses will define the management for specific games.
    """

    def __init__(self, game_state, game_solver, game_observer):
        self.game_state_node = GameStateNode(game_state)
        self.game_solver = game_solver
        self.game_observer = game_observer

    @abstractmethod
    def play(self):
        pass

    def expand_state_space(self, game_state_node):
        """Expand the state space tree from the given node."""
        for move in game_state_node.game_state.available_moves():
            child_state = game_state_node.game_state.clone()
            child_state.update(move)
            child_node = GameStateNode(child_state, parent=game_state_node)
            game_state_node.add_child(child_node)
            self.expand_state_space(child_node)



class PacmanManager(GameManager):
    """Class for managing a Pacman game."""
    def __init__(self, game_state, game_solver, game_observer):
        super().__init__(game_state, game_solver, game_observer)

    def play(self):
        """Plays the Pacman game."""
        self.expand_state_space(self.game_state_node)
        best_move = self.game_solver.solve(self.game_state_node)
        self.game_state_node = self.game_state_node.get_child(best_move)
        self.game_observer.update(self.game_state_node.game_state)
        while not self.game_state_node.game_state.is_win() and not self.game_state_node.game_state.is_draw():
            self.expand_state_space(self.game_state_node)
            best_move = self.game_solver.solve(self.game_state_node)
            self.game_state_node = self.game_state_node.get_child(best_move)
            self.game_observer.update(self.game_state_node.game_state)


# class TicTacToeManager(GameManager):
#     """Concrete implementation of the management for Tic Tac Toe."""
#
#     def play(self):
#         self.expand_state_space(self.game_state_node)

class TicTacToeManager(GameManager):
    """Concrete implementation of the manager for Tic Tac Toe game."""

    def play(self):
        """Play the game until it ends (win or draw)."""
        while not self.game_state_node.game_state.is_win() and not self.game_state_node.game_state.is_draw():
            self.game_observer.update(self.game_state_node.game_state)
            move = self.game_solver.solve(self.game_state_node.game_state)
            child_state = self.game_state_node.game_state.clone()
            child_state.update(move)
            self.game_state_node = GameStateNode(child_state, parent=self.game_state_node)
        self.game_observer.update(self.game_state_node.game_state)
        if self.game_state_node.game_state.is_win():
            print("Player {} wins!".format(self.game_state_node.game_state.get_current_player()))
        else:
            print("Draw")

class GameFactory:
    """Factory for creating instances of specific games and their related classes."""

    @staticmethod
    def create_game(game_type, **kwargs):
        if game_type == "Pacman":
            board = kwargs.get('board')
            pacman_position = kwargs.get('pacman_position')
            exit_position = kwargs.get('exit_position')
            red_pellets = kwargs.get('red_pellets')
            green_pellets = kwargs.get('green_pellets')
            game_state = PacmanState(board, pacman_position, exit_position, red_pellets, green_pellets)
            game_solver = AStarSolver()
            game_observer=ConsoleObserver()
            return PacmanManager(game_state, game_solver, game_observer)
        elif game_type == "TicTacToe":
            board = kwargs.get('board')
            n = kwargs.get('n')
            max_depth = kwargs.get('max_depth', 3)
            player_id = kwargs.get('player_id', 1)
            game_state = TicTacToeState(n,board)
            game_solver = MinMaxSolver(max_depth, player_id)
            game_observer = ConsoleObserver()
            return TicTacToeManager(game_state, game_solver, game_observer)

if __name__ == '__main__':

    # game_factory = GameFactory()
    # game = game_factory.create_game("Pacman", board=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    #                                  pacman_position=(0,0), exit_position=(2,2),
    #                                  red_pellets=[(1,1)], green_pellets=[(0,2)])
    # game.play()

    game = GameFactory.create_game("TicTacToe", n = 3, board=[["X", "O", ""], ["", "X", "O"], ["O", "", "X"]])
    game.play()
