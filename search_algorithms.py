"""
Assignment Q1: Game Search Algorithms
Implements: Minimax, Alpha-Beta Pruning, Heuristic Alpha-Beta, Monte-Carlo Tree Search
Game: Tic-Tac-Toe (3x3)
"""

import math
import random
import time
from copy import deepcopy


# ──────────────────────────────────────────────
# GAME: Tic-Tac-Toe
# ──────────────────────────────────────────────

class TicTacToe:
    """3x3 Tic-Tac-Toe board. X=1, O=-1, Empty=0"""

    def __init__(self, board=None):
        self.board = board if board else [[0] * 3 for _ in range(3)]
        self.current_player = 1  # X starts

    def get_actions(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == 0]

    def make_move(self, row, col):
        new = TicTacToe(deepcopy(self.board))
        new.board[row][col] = self.current_player
        new.current_player = -self.current_player
        return new

    def is_terminal(self):
        return self.winner() is not None or not self.get_actions()

    def winner(self):
        b = self.board
        lines = (
            [b[r] for r in range(3)] +
            [[b[r][c] for r in range(3)] for c in range(3)] +
            [[b[0][0], b[1][1], b[2][2]], [b[0][2], b[1][1], b[2][0]]]
        )
        for line in lines:
            if all(x == 1 for x in line): return 1
            if all(x == -1 for x in line): return -1
        return None

    def utility(self):
        w = self.winner()
        return w if w else 0

    def heuristic(self):
        """Simple heuristic: count potential winning lines"""
        score = 0
        b = self.board
        lines = (
            [b[r] for r in range(3)] +
            [[b[r][c] for r in range(3)] for c in range(3)] +
            [[b[0][0], b[1][1], b[2][2]], [b[0][2], b[1][1], b[2][0]]]
        )
        for line in lines:
            xs = line.count(1)
            os = line.count(-1)
            if xs > 0 and os > 0:
                continue  # blocked
            if xs == 2: score += 10
            if xs == 1: score += 1
            if os == 2: score -= 10
            if os == 1: score -= 1
        return score

    def display(self):
        symbols = {1: 'X', -1: 'O', 0: '.'}
        for row in self.board:
            print(' '.join(symbols[c] for c in row))
        print()


# ──────────────────────────────────────────────
# ALGORITHM 1: Minimax
# ──────────────────────────────────────────────

class Minimax:
    """
    Standard Minimax Algorithm.
    MAX player (X=1) maximises; MIN player (O=-1) minimises.
    No pruning – explores the full game tree.
    """

    def __init__(self):
        self.nodes_explored = 0

    def minimax(self, state, is_maximising):
        self.nodes_explored += 1
        if state.is_terminal():
            return state.utility()

        if is_maximising:
            best = -math.inf
            for action in state.get_actions():
                val = self.minimax(state.make_move(*action), False)
                best = max(best, val)
            return best
        else:
            best = math.inf
            for action in state.get_actions():
                val = self.minimax(state.make_move(*action), True)
                best = min(best, val)
            return best

    def best_move(self, state):
        self.nodes_explored = 0
        best_val, best_action = -math.inf, None
        for action in state.get_actions():
            val = self.minimax(state.make_move(*action), False)
            if val > best_val:
                best_val, best_action = val, action
        return best_action, best_val


# ──────────────────────────────────────────────
# ALGORITHM 2: Alpha-Beta Pruning
# ──────────────────────────────────────────────

class AlphaBeta:
    """
    Alpha-Beta Pruning (exact, no depth limit).
    Prunes branches that cannot influence the final decision.
    Alpha = best value MAX can guarantee so far.
    Beta  = best value MIN can guarantee so far.
    """

    def __init__(self):
        self.nodes_explored = 0
        self.nodes_pruned = 0

    def alpha_beta(self, state, alpha, beta, is_maximising):
        self.nodes_explored += 1
        if state.is_terminal():
            return state.utility()

        if is_maximising:
            val = -math.inf
            for action in state.get_actions():
                val = max(val, self.alpha_beta(state.make_move(*action), alpha, beta, False))
                alpha = max(alpha, val)
                if beta <= alpha:
                    self.nodes_pruned += 1
                    break  # Beta cut-off
            return val
        else:
            val = math.inf
            for action in state.get_actions():
                val = min(val, self.alpha_beta(state.make_move(*action), alpha, beta, True))
                beta = min(beta, val)
                if beta <= alpha:
                    self.nodes_pruned += 1
                    break  # Alpha cut-off
            return val

    def best_move(self, state):
        self.nodes_explored = 0
        self.nodes_pruned = 0
        best_val, best_action = -math.inf, None
        alpha, beta = -math.inf, math.inf
        for action in state.get_actions():
            val = self.alpha_beta(state.make_move(*action), alpha, beta, False)
            if val > best_val:
                best_val, best_action = val, action
            alpha = max(alpha, best_val)
        return best_action, best_val


# ──────────────────────────────────────────────
# ALGORITHM 3: Heuristic Alpha-Beta (depth-limited)
# ──────────────────────────────────────────────

class HeuristicAlphaBeta:
    """
    Alpha-Beta with a depth cutoff and heuristic evaluation function.
    Allows pruning huge game trees (e.g., Chess) where full search is infeasible.
    """

    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        self.nodes_explored = 0

    def h_alpha_beta(self, state, depth, alpha, beta, is_maximising):
        self.nodes_explored += 1
        if state.is_terminal():
            return state.utility() * 100  # terminal wins matter most

        if depth == 0:
            return state.heuristic()  # use heuristic at cutoff

        if is_maximising:
            val = -math.inf
            for action in state.get_actions():
                val = max(val, self.h_alpha_beta(state.make_move(*action), depth - 1, alpha, beta, False))
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return val
        else:
            val = math.inf
            for action in state.get_actions():
                val = min(val, self.h_alpha_beta(state.make_move(*action), depth - 1, alpha, beta, True))
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return val

    def best_move(self, state):
        self.nodes_explored = 0
        best_val, best_action = -math.inf, None
        for action in state.get_actions():
            val = self.h_alpha_beta(state.make_move(*action), self.max_depth, -math.inf, math.inf, False)
            if val > best_val:
                best_val, best_action = val, action
        return best_action, best_val


# ──────────────────────────────────────────────
# ALGORITHM 4: Monte-Carlo Tree Search (MCTS)
# ──────────────────────────────────────────────

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = state.get_actions()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return self.state.is_terminal()

    def ucb1(self, c=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, c=1.41):
        return max(self.children, key=lambda n: n.ucb1(c))

    def expand(self):
        action = self.untried_actions.pop(random.randint(0, len(self.untried_actions) - 1))
        child_state = self.state.make_move(*action)
        child = MCTSNode(child_state, parent=self, action=action)
        self.children.append(child)
        return child

    def rollout(self):
        """Random simulation to terminal state"""
        state = self.state
        while not state.is_terminal():
            action = random.choice(state.get_actions())
            state = state.make_move(*action)
        return state.utility()

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)


class MCTS:
    """
    Monte-Carlo Tree Search.
    Four phases: Selection (UCB1) → Expansion → Simulation (rollout) → Backpropagation.
    No domain heuristic needed; learns through random play.
    """

    def __init__(self, iterations=1000):
        self.iterations = iterations

    def best_move(self, state):
        root = MCTSNode(state)
        for _ in range(self.iterations):
            node = self._select(root)
            if not node.is_terminal():
                node = node.expand()
            result = node.rollout()
            node.backpropagate(result)

        best = max(root.children, key=lambda n: n.visits)
        return best.action, best.wins / best.visits if best.visits else 0

    def _select(self, node):
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            node = node.best_child()
        return node


# ──────────────────────────────────────────────
# DEMO & COMPARISON
# ──────────────────────────────────────────────

def run_all(state, label="Initial Board"):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    state.display()

    algos = [
        ("Minimax",              Minimax()),
        ("Alpha-Beta",           AlphaBeta()),
        ("Heuristic Alpha-Beta", HeuristicAlphaBeta(max_depth=4)),
        ("MCTS (1000 iters)",    MCTS(iterations=1000)),
    ]

    for name, algo in algos:
        t0 = time.time()
        move, val = algo.best_move(state)
        elapsed = time.time() - t0

        extra = ""
        if hasattr(algo, 'nodes_explored'):
            extra += f"  nodes={algo.nodes_explored}"
        if hasattr(algo, 'nodes_pruned'):
            extra += f"  pruned={algo.nodes_pruned}"

        print(f"{name:<25} → move={move}  val={val:.2f}  time={elapsed:.4f}s{extra}")


if __name__ == "__main__":
    # Test 1: empty board
    game = TicTacToe()
    run_all(game, "Empty Board (X to move)")

    # Test 2: mid-game
    mid = TicTacToe([[1, -1, 0], [0, 1, -1], [0, 0, 0]])
    run_all(mid, "Mid-Game Board (X to move)")

    # Test 3: one move from X win – all algos should block/win
    near = TicTacToe([[-1, 0, 0], [1, 1, 0], [-1, 0, 0]])
    run_all(near, "Near-Win Board (X to move – should win at (1,2))")

    print("\nAll algorithms completed successfully.")
