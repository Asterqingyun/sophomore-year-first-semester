import argparse
import os
from typing import Tuple

import numpy as np

from Game import GomokuGame
from MCTS import MCTS
from NeuralNet import NeuralNet
from utils import dotdict


def _parse_human_move(s: str) -> Tuple[int, int]:
    s = s.strip().lower()
    if s in {"q", "quit", "exit"}:
        raise KeyboardInterrupt

    # 支持: "x y" / "x,y" / "x，y"
    for sep in [",", "，", " ", "\t"]:
        if sep in s:
            parts = [
                p
                for p in s.replace("，", ",")
                .replace("\t", " ")
                .replace(",", " ")
                .split()
                if p
            ]
            if len(parts) != 2:
                break
            return int(parts[0]), int(parts[1])

    raise ValueError("请输入坐标: x y（例如: 7 7），或输入 q 退出")


def _action_from_xy(x: int, y: int, n: int) -> int:
    return int(x) * int(n) + int(y)


def main():
    parser = argparse.ArgumentParser(
        description="Play Gomoku (15x15) vs trained AlphaZero model"
    )
    parser.add_argument("--size", type=int, default=15, help="board size, default 15")
    parser.add_argument("--sims", type=int, default=200, help="MCTS sims per move")
    parser.add_argument("--cpuct", type=float, default=1.0, help="cpuct")
    parser.add_argument(
        "--human",
        choices=["black", "white"],
        default="black",
        help="human plays black(first) or white(second)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="path to checkpoint .pth.tar (default: Chess/temp/best.pth.tar)",
    )
    args_cli = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    chess_dir = os.path.abspath(os.path.join(base_dir, "..", ".."))
    temp_dir = os.path.join(chess_dir, "temp")
    ckpt_path = args_cli.checkpoint or os.path.join(temp_dir, "best.pth.tar")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    game = GomokuGame(args_cli.size)
    nnet = NeuralNet(game)

    ckpt_folder = os.path.dirname(ckpt_path)
    ckpt_name = os.path.basename(ckpt_path)
    print(f'Loading checkpoint: "{ckpt_folder}/{ckpt_name}"')
    nnet.load_checkpoint(ckpt_folder, ckpt_name)

    mcts_args = dotdict(
        {
            "numMCTSSims": int(args_cli.sims),
            "cpuct": float(args_cli.cpuct),
            "mcts_batch_size": 64,
        }
    )
    mcts = MCTS(game, nnet, mcts_args)

    human_player = 1 if args_cli.human == "black" else -1

    board = game.getInitBoard()
    cur_player = 1

    print("\nEnter moves as: x y   (0-based). Example: 7 7")
    print("Type 'q' to quit.\n")

    while True:
        GomokuGame.display(board)
        ended = game.getGameEnded(board, cur_player)
        if ended != 0:
            if ended == 1:
                winner = cur_player
            elif ended == -1:
                winner = -cur_player
            else:
                winner = 0

            if winner == 0:
                print("Game ended: draw")
            elif winner == human_player:
                print("Game ended: you win")
            else:
                print("Game ended: AI wins")
            return

        if cur_player == human_player:
            while True:
                try:
                    s = input("Your move (x y): ")
                    x, y = _parse_human_move(s)
                    if x < 0 or x >= args_cli.size or y < 0 or y >= args_cli.size:
                        print("Out of range")
                        continue
                    if board[x][y] != 0:
                        print("Position occupied")
                        continue
                    action = _action_from_xy(x, y, args_cli.size)
                    break
                except ValueError as e:
                    print(e)
                except KeyboardInterrupt:
                    print("Quit")
                    return
        else:
            canonical = game.getCanonicalForm(board, cur_player)
            pi = mcts.getActionProb(canonical, temp=0)
            action = int(np.argmax(pi))
            x, y = divmod(action, args_cli.size)
            print(f"AI move: {x} {y}")

        board, cur_player = game.getNextState(board, cur_player, action)


if __name__ == "__main__":
    main()
