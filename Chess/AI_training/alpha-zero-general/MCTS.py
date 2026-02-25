import logging
import math

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        # 单次批量评估的最大叶子数，若 args 中未配置则给默认值
        # dotdict.__getattr__ 不支持 getattr 默认值，因此用 dict.get
        self.mcts_batch_size = args.get("mcts_batch_size", 16)
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        # stores list of valid action indices for board s (for faster selection)
        self.Vs_actions = {}

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        # 使用批量叶子评估版本的 MCTS
        # 关键：先确保根节点至少被扩展一次。
        # 否则当 mcts_batch_size >= numMCTSSims 时，所有 _select 都会返回 (path=[],'nn',root)
        # 导致回传时没有任何边被更新，最终 counts 全 0 -> 除 0。
        s_root = self.game.stringRepresentation(canonicalBoard)
        if s_root not in self.Ps:
            path0, leaf0 = self._select(canonicalBoard)
            self._evaluate_and_backup([(path0, leaf0)])

        pending = []  # 每项: (path, leaf_info)

        for i in range(self.args.numMCTSSims):
            path, leaf_info = self._select(canonicalBoard)
            pending.append((path, leaf_info))

            # 累积到一定数量，或者到达最后一次模拟时，一起评估并回传
            if len(pending) >= self.mcts_batch_size or i == self.args.numMCTSSims - 1:
                self._evaluate_and_backup(pending)
                pending = []

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [
            self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
            for a in range(self.game.getActionSize())
        ]

        # 兜底：如果因为异常情况导致根节点所有边访问次数仍为 0，返回合法动作的均匀分布
        # （避免 counts_sum=0 的 ZeroDivisionError）
        if sum(counts) == 0:
            valids = self.game.getValidMoves(canonicalBoard, 1).astype(np.float64)
            if valids.sum() > 0:
                return (valids / valids.sum()).tolist()
            # 理论上这意味着无合法步（应当已终局），这里给均匀分布只为防崩
            return [1.0 / len(counts)] * len(counts)

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def _select(self, canonicalBoard):
        """从根开始沿 UCB 选择一路向下，直到遇到
        - 终局节点：返回 (path, ('terminal', s, v_start))
        - 未扩展的新叶子：返回 (path, ('nn', s, board))

        path: 列表 [(s, a), ...]，不包含叶子本身，只包含沿途节点及其所选动作。
        """

        path = []
        board = canonicalBoard

        while True:
            s = self.game.stringRepresentation(board)

            if s not in self.Es:
                self.Es[s] = self.game.getGameEnded(board, 1)

            if self.Es[s] != 0:
                # 终局：与原 search 一致，基值为 -Es[s]
                v_start = -self.Es[s]
                return path, ("terminal", s, v_start)

            if s not in self.Ps:
                # 新叶子：后续批量调用神经网络
                return path, ("nn", s, board)

            # 已扩展过：根据 UCB 继续向下选择
            actions = self.Vs_actions.get(s)
            if actions is None:
                # 兼容老缓存/异常情况：从 valids 重新构建一次动作列表
                valids = self.Vs[s]
                actions = np.flatnonzero(np.asarray(valids)).tolist()
                self.Vs_actions[s] = actions
            cur_best = -float("inf")
            best_act = -1

            # 绑定局部变量，减少 Python 属性/字典查找开销
            Qsa = self.Qsa
            Nsa = self.Nsa
            Ns = self.Ns
            Ps_s = self.Ps[s]
            cpuct = self.args.cpuct
            sqrt_Ns = math.sqrt(Ns[s] + EPS)
            sqrt_Ns_exact = math.sqrt(Ns[s]) if Ns[s] > 0 else 0.0

            for a in actions:
                key = (s, a)
                if key in Qsa:
                    u = Qsa[key] + cpuct * Ps_s[a] * sqrt_Ns_exact / (1 + Nsa[key])
                else:
                    u = cpuct * Ps_s[a] * sqrt_Ns

                if u > cur_best:
                    cur_best = u
                    best_act = a

            # 防护：不应该出现“没有任何合法动作却非终局”的情况。
            # 若出现，直接当作终局返回，避免写入非法 action=-1。
            if best_act == -1:
                log.error(
                    "No valid moves in _select() but state is non-terminal. s=%s, sum(valids)=%s",
                    s,
                    len(actions) if actions is not None else None,
                )
                return path, ("terminal", s, 0.0)

            a = best_act
            path.append((s, a))

            next_s, next_player = self.game.getNextState(board, 1, a)
            board = self.game.getCanonicalForm(next_s, next_player)

    def _evaluate_and_backup(self, pending):
        """对 pending 中所有叶子进行一次性评估，并沿各自路径回传更新 Qsa/Nsa/Ns/Ps/Vs。

        pending: 列表 [(path, leaf_info), ...]
        leaf_info:
            - ('terminal', s, v_start)
            - ('nn', s, board)
        """

        if not pending:
            return

        # 先收集需要神经网络评估的叶子
        nn_boards = []
        nn_indices = []  # pending 中对应的索引
        for idx, (path, leaf_info) in enumerate(pending):
            kind = leaf_info[0]
            if kind == "nn":
                _, s_leaf, board_leaf = leaf_info
                nn_boards.append(board_leaf)
                nn_indices.append(idx)

        # 批量调用神经网络（始终保持为 ndarray，避免 Optional 导致的静态检查/运行时问题）
        ps_batch = np.empty((0,), dtype=np.float32)
        vs_batch = np.empty((0,), dtype=np.float32)
        if nn_boards:
            if hasattr(self.nnet, "predict_batch"):
                ps_batch, vs_batch = self.nnet.predict_batch(nn_boards)
            # 兜底：若批量接口不存在或返回异常，回退为逐个调用 predict
            if ps_batch is None or vs_batch is None:
                ps_list = []
                vs_list = []
                for b in nn_boards:
                    ps, v = self.nnet.predict(b)
                    ps_list.append(ps)
                    vs_list.append(v)
                ps_batch = np.array(ps_list, dtype=np.float32)
                vs_batch = np.array(vs_list, dtype=np.float32)
            # 类型/形状收敛，避免奇怪的返回类型
            ps_batch = np.asarray(ps_batch, dtype=np.float32)
            vs_batch = np.asarray(vs_batch, dtype=np.float32)

        # 为每条模拟路径计算起始值并回传
        nn_cursor = 0
        for idx, (path, leaf_info) in enumerate(pending):
            kind = leaf_info[0]

            if kind == "terminal":
                _, s_leaf, v_start = leaf_info
            else:  # 'nn'
                _, s_leaf, board_leaf = leaf_info

                # 取出对应的网络输出
                ps_s = ps_batch[nn_cursor]
                v_net = vs_batch[nn_cursor]
                nn_cursor += 1

                # 掩蔽非法动作并归一化，逻辑与原 search 完全一致
                valids = self.game.getValidMoves(board_leaf, 1)
                ps_s = ps_s * valids
                sum_ps = np.sum(ps_s)
                if sum_ps > 0:
                    ps_s /= sum_ps
                else:
                    log.error("All valid moves were masked, doing a workaround.")
                    ps_s = ps_s + valids
                    ps_s /= np.sum(ps_s)

                self.Ps[s_leaf] = ps_s
                self.Vs[s_leaf] = valids
                # 缓存合法动作索引，避免 _select 每次扫描全动作空间
                self.Vs_actions[s_leaf] = np.flatnonzero(np.asarray(valids)).tolist()
                self.Ns[s_leaf] = 0

                # 与原 search 一致，叶子基值为 -v_net
                v_start = -v_net

            # 从叶子向根回传，沿途每一层翻转符号
            v = v_start
            for s, a in reversed(path):
                if (s, a) in self.Qsa:
                    self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                        self.Nsa[(s, a)] + 1
                    )
                    self.Nsa[(s, a)] += 1
                else:
                    self.Qsa[(s, a)] = v
                    self.Nsa[(s, a)] = 1

                # 访问计数：每次模拟经过该状态一次
                if s in self.Ns:
                    self.Ns[s] += 1
                else:
                    self.Ns[s] = 1

                v = -v

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float("inf")
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(
                        self.Ns[s]
                    ) / (1 + self.Nsa[(s, a)])
                else:
                    u = (
                        self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
                    )  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                self.Nsa[(s, a)] + 1
            )
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
