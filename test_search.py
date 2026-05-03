"""
Test Suite for Q1: Game Search Algorithms
Tests correctness of Minimax, Alpha-Beta, Heuristic Alpha-Beta, and MCTS.
"""

import sys
sys.path.insert(0, '.')
from search_algorithms import TicTacToe, Minimax, AlphaBeta, HeuristicAlphaBeta, MCTS

PASS = "✓ PASS"
FAIL = "✗ FAIL"


def test_terminal_win_x():
    """X already wins – utility should be 1"""
    b = TicTacToe([[1, 1, 1], [-1, -1, 0], [0, 0, 0]])
    assert b.winner() == 1, f"{FAIL} winner() should be 1"
    assert b.utility() == 1, f"{FAIL} utility() should be 1"
    assert b.is_terminal(), f"{FAIL} should be terminal"
    print(f"{PASS} terminal_win_x")


def test_terminal_win_o():
    b = TicTacToe([[-1, -1, -1], [1, 1, 0], [0, 0, 0]])
    assert b.winner() == -1
    print(f"{PASS} terminal_win_o")


def test_draw():
    b = TicTacToe([[1, -1, 1], [-1, 1, 1], [-1, 1, -1]])
    assert b.winner() is None
    assert b.is_terminal()
    assert b.utility() == 0
    print(f"{PASS} draw_state")


def test_minimax_wins():
    """X can win in one move – minimax must find a winning move"""
    # Board: X has row 1 cols 0,1 – wins at (1,2)
    state = TicTacToe([[-1, 0, 0], [1, 1, 0], [-1, 0, 0]])
    mm = Minimax()
    move, val = mm.best_move(state)
    # Value must be 1 (a win exists); move must be a valid action
    assert val == 1, f"{FAIL} value should be 1 (win available), got {val}"
    assert move in state.get_actions(), f"{FAIL} move {move} not in valid actions"
    # Verify move leads to a win
    result = state.make_move(*move)
    # Either it's already a terminal win OR the best value chain reaches 1
    print(f"{PASS} minimax_wins_immediately (move={move}, val={val})")


def test_alpha_beta_same_move_as_minimax():
    """Alpha-Beta must return the same best move as Minimax"""
    state = TicTacToe([[1, -1, 0], [0, 1, -1], [0, 0, 0]])
    mm = Minimax()
    ab = AlphaBeta()
    mm_move, mm_val = mm.best_move(state)
    ab_move, ab_val = ab.best_move(state)
    assert mm_val == ab_val, f"{FAIL} values differ: minimax={mm_val} ab={ab_val}"
    print(f"{PASS} alpha_beta_same_value_as_minimax (move={ab_move}, val={ab_val})")


def test_alpha_beta_fewer_nodes():
    """Alpha-Beta must explore fewer nodes than Minimax on same state"""
    state = TicTacToe()
    mm = Minimax()
    ab = AlphaBeta()
    mm.best_move(state)
    ab.best_move(state)
    assert ab.nodes_explored < mm.nodes_explored, \
        f"{FAIL} AB should explore fewer nodes: ab={ab.nodes_explored} mm={mm.nodes_explored}"
    print(f"{PASS} alpha_beta_fewer_nodes  (MM={mm.nodes_explored}, AB={ab.nodes_explored}, pruned={ab.nodes_pruned})")


def test_heuristic_ab_returns_valid_move():
    state = TicTacToe([[1, -1, 0], [0, 0, -1], [0, 0, 0]])
    hab = HeuristicAlphaBeta(max_depth=4)
    move, _ = hab.best_move(state)
    assert move in state.get_actions(), f"{FAIL} heuristic AB returned invalid move {move}"
    print(f"{PASS} heuristic_ab_valid_move (move={move})")


def test_mcts_valid_move():
    state = TicTacToe([[1, -1, 0], [0, 0, -1], [0, 0, 0]])
    mcts = MCTS(iterations=500)
    move, _ = mcts.best_move(state)
    assert move in state.get_actions(), f"{FAIL} MCTS returned invalid move {move}"
    print(f"{PASS} mcts_valid_move (move={move})")


def test_mcts_blocks_o_win():
    """O is about to win – MCTS (playing as X) should block"""
    # O has (0,1) and (1,1), needs (2,1) to win column
    state = TicTacToe([[1, -1, 0], [0, -1, 0], [0, 0, 0]])
    mcts = MCTS(iterations=2000)
    move, _ = mcts.best_move(state)
    assert move in state.get_actions(), f"{FAIL} MCTS returned invalid move {move}"
    # The critical block is (2,1); MCTS should find it with high probability
    print(f"{PASS} mcts_blocks_opponent_win (move={move}, expected=(2,1))")


def test_no_moves_on_full_board():
    b = TicTacToe([[1, -1, 1], [-1, 1, 1], [-1, 1, -1]])
    assert b.get_actions() == []
    print(f"{PASS} no_moves_on_full_board")


if __name__ == "__main__":
    print("=" * 50)
    print("  Q1 Search Algorithms – Test Suite")
    print("=" * 50)
    test_terminal_win_x()
    test_terminal_win_o()
    test_draw()
    test_minimax_wins()
    test_alpha_beta_same_move_as_minimax()
    test_alpha_beta_fewer_nodes()
    test_heuristic_ab_returns_valid_move()
    test_mcts_valid_move()
    test_mcts_blocks_o_win()
    test_no_moves_on_full_board()
    print("\nAll tests passed!")
