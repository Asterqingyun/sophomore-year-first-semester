import numpy as np
import os
import ctypes
from typing import Any, List, Tuple


class Game:
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """

    def __init__(self):
        pass

    def getInitBoard(self) -> Any:
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        raise NotImplementedError()

    def getBoardSize(self) -> Tuple[int, int]:
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        raise NotImplementedError()

    def getActionSize(self) -> int:
        """
        Returns:
            actionSize: number of all possible actions
        """
        raise NotImplementedError()

    def getNextState(self, board: Any, player: int, action: int) -> Tuple[Any, int]:
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        raise NotImplementedError()

    def getValidMoves(self, board: Any, player: int) -> Any:
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        raise NotImplementedError()

    def getGameEnded(self, board: Any, player: int) -> float:
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        raise NotImplementedError()

    def getCanonicalForm(self, board: Any, player: int) -> Any:
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        raise NotImplementedError()

    def getSymmetries(self, board: Any, pi: Any) -> List[Any]:
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        raise NotImplementedError()

    def stringRepresentation(self, board: Any) -> str:
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        raise NotImplementedError()


class GomokuGame(Game):
    """标准五子棋实现：15x15 棋盘，黑白双方对弈。

    约定：
        - 棋盘上 1 表示当前玩家的棋子，-1 表示对手的棋子，0 表示空。
    - Game 接口中的 player 取值为 1 或 -1，表示轮到哪一方。
    """

    def __init__(self, n=15, n_in_row=5, use_forbidden=False):
        self.n = n
        self.n_in_row = n_in_row
        # 训练阶段建议关闭（禁手是“黑方特有的非对称规则”，会与 canonical form/对称增强产生语义冲突）
        self.use_forbidden = bool(use_forbidden)
        # 延迟加载 C 侧禁手规则库
        self._c_lib = None

    # ===================== C 侧禁手桥接 =====================

    def _load_c_lib(self):
        """加载包含 is_forbidden_py 的动态库（Windows 下默认 gomoku_rules.dll）。"""

        if self._c_lib is not None:
            return self._c_lib

        # 默认放在当前 Game.py 所在目录或其父目录下，名称可按需调整
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, "gomoku_rules.dll"),
            os.path.join(os.path.dirname(base_dir), "gomoku_rules.dll"),
        ]

        lib = None
        for path in candidates:
            if os.path.exists(path):
                try:
                    lib = ctypes.CDLL(path)
                    break
                except OSError:
                    continue

        if lib is None:
            self._c_lib = None
            return None

        # 设置函数签名：int is_forbidden_py(const int *board_flat, int current_player, int x, int y)
        lib.is_forbidden_py.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.is_forbidden_py.restype = ctypes.c_int

        self._c_lib = lib
        return self._c_lib

    def _is_forbidden(self, board, player, x, y):
        """调用 C 规则引擎判断 (x,y) 是否为禁手。

        board 中棋子：1 表示先手/黑，-1 表示后手/白，0 表示空。
        C 侧棋子：BLACK=1, WHITE=2。
        """

        if not self.use_forbidden:
            return False

        lib = self._load_c_lib()
        if lib is None:
            # 如果没加载到 DLL，则视为无禁手规则
            return False

        # 映射到 C 侧编码：0,1,2（兼容棋盘里可能存在 2/-2 标记）
        s = np.sign(board)
        c_board = np.zeros_like(board, dtype=np.int32)
        c_board[s == 1] = 1  # BLACK
        c_board[s == -1] = 2  # WHITE

        # player: 1 -> BLACK(1), -1 -> WHITE(2)
        c_player = 1 if player == 1 else 2

        flat = c_board.astype(np.int32, copy=False).ravel()
        ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        res = lib.is_forbidden_py(
            ptr, ctypes.c_int(c_player), ctypes.c_int(x), ctypes.c_int(y)
        )
        return bool(res)

    def getInitBoard(self):
        # 初始空棋盘
        return np.zeros((self.n, self.n), dtype=np.int8)

    def getBoardSize(self):
        return (self.n, self.n)

    def getActionSize(self):
        # 每个格子一个动作，不设“pass”
        return self.n * self.n

    def getNextState(self, board, player, action):
        """在 board 上为 player 执行一次落子，返回新棋盘和下一手玩家。"""

        b = np.copy(board)
        x, y = divmod(int(action), self.n)
        # 防御性检查：不允许在非空位置落子
        if b[x][y] != 0:
            raise ValueError("Invalid move: position already occupied")

        b[x][y] = int(player)
        return b, -player

    def getValidMoves(self, board, player):
        """返回一维 0/1 向量。

        当 use_forbidden=False 时，不进行禁手过滤（更快、更稳定）。
        """

        valids = np.zeros(self.getActionSize(), dtype=np.int8)

        # 若游戏已结束，则不再有合法动作
        if self.getGameEnded(board, player) != 0:
            return valids

        # 不启用禁手：所有空位都合法
        if not self.use_forbidden:
            valids[(board.reshape(-1) == 0)] = 1
            return valids

        # 启用禁手：逐点试落子后调用 C 规则过滤
        for a in range(self.getActionSize()):
            x, y = divmod(a, self.n)
            if board[x][y] != 0:
                continue

            tmp = np.copy(board)
            tmp[x][y] = int(player)

            if self._is_forbidden(tmp, player, x, y):
                continue

            valids[a] = 1

        return valids

    def getGameEnded(self, board, player):
        """判断游戏是否结束。

        返回：
        - 0: 未结束
        - 1: 传入的 player 获胜
        - -1: 传入的 player 落败
        - 1e-4: 和棋
        """

        n = self.n
        k = self.n_in_row

        # 检查是否有一方连成 k 子
        for x in range(n):
            for y in range(n):
                piece = board[x][y]
                if piece == 0:
                    continue

                # 水平
                if y + k <= n and np.all(board[x, y : y + k] == piece):
                    return 1 if piece == player else -1

                # 垂直
                if x + k <= n and np.all(board[x : x + k, y] == piece):
                    return 1 if piece == player else -1

                # 主对角线
                if x + k <= n and y + k <= n:
                    if all(board[x + i, y + i] == piece for i in range(k)):
                        return 1 if piece == player else -1

                # 副对角线
                if x + k <= n and y - k + 1 >= 0:
                    if all(board[x + i, y - i] == piece for i in range(k)):
                        return 1 if piece == player else -1

        # 没人获胜且有空位 -> 未结束
        if np.any(board == 0):
            return 0

        # 棋盘已满且无人获胜 -> 和棋
        return 1e-4

    def getCanonicalForm(self, board, player):
        # 从当前 player 视角看棋盘
        return player * board

    def getSymmetries(self, board, pi):
        """返回所有旋转 / 翻转对称形式，用于数据增强。"""

        assert len(pi) == self.n * self.n
        pi_board = np.reshape(pi, (self.n, self.n))
        symm = []

        for i in range(4):  # 0,90,180,270 度
            newB = np.rot90(board, i)
            newPi = np.rot90(pi_board, i)
            for flip in [False, True]:
                b_sym = np.fliplr(newB) if flip else newB
                pi_sym = np.fliplr(newPi) if flip else newPi
                symm.append((b_sym, pi_sym.ravel().copy()))
        return symm

    def stringRepresentation(self, board):
        # 用于 MCTS 哈希
        return board.tobytes()

    @staticmethod
    def display(board):
        """在终端打印棋盘。符号表示黑白。"""

        n = board.shape[0]
        # 列号
        print("   ", end="")
        for y in range(n):
            print(f"{y:2d}", end=" ")
        print()

        for x in range(n):
            # 行号
            print(f"{x:2d}", end=" ")
            for y in range(n):
                piece = board[x][y]
                if piece == 1:
                    ch = "B"
                elif piece == -1:
                    ch = "W"
                else:
                    ch = "."
                print(f" {ch}", end="")
            print()
