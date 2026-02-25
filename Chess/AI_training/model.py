# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

# ==========================================
# Part 1: 基础设置与简易棋盘 (Environment)
# ==========================================

BOARD_WIDTH = 15
BOARD_HEIGHT = 15
N_FEATURE = 3  # 输入通道数: [我方子, 敌方子, 当前回合]


class Board(object):
    """
    五子棋盘逻辑，为了让MCTS能跑起来必须实现的基本功能
    """

    def __init__(self):
        self.width = BOARD_WIDTH
        self.height = BOARD_HEIGHT
        self.states = {}  # 存储棋盘状态 {move: player}
        self.n_in_row = 5
        self.players = [1, 2]  # 1: 黑棋, 2: 白棋

    def init_board(self, start_player=0):
        self.current_player = self.players[start_player]
        self.states = {}
        self.last_move = -1
        # 合法落子列表
        self.availables = list(range(self.width * self.height))

    def move_to_location(self, move):
        """0-224 的数字转为 (h, w)"""
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        """(h, w) 转为 0-224"""
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """
        构造神经网络需要的输入 Feature Map (3, 15, 15)
        Channel 0: 当前玩家的子
        Channel 1: 对手的子
        Channel 2: 当前回合标记 (全0或全1)
        """
        square_state = np.zeros((N_FEATURE, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]

            # 填充 Channel 0 (我方)
            square_state[0][move_curr // self.width, move_curr % self.width] = 1.0
            # 填充 Channel 1 (敌方)
            square_state[1][move_oppo // self.width, move_oppo % self.width] = 1.0
            # 填充 Channel 2 (刚才落子的位置，辅助特征，或者用全1代表先手亦可)
            # 这里简化处理：如果是执黑(player1)，全1；执白，全0
            if self.current_player == 1:
                square_state[2][:, :] = 1.0

        return square_state

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0]
            if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        """简单的胜负判断逻辑"""
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (
                w in range(width - n + 1)
                and len(set(states.get(i, -1) for i in range(m, m + n))) == 1
            ):
                return True, player

            if (
                h in range(height - n + 1)
                and len(set(states.get(i, -1) for i in range(m, m + n * width, width)))
                == 1
            ):
                return True, player

            if (
                w in range(width - n + 1)
                and h in range(height - n + 1)
                and len(
                    set(
                        states.get(i, -1)
                        for i in range(m, m + n * (width + 1), width + 1)
                    )
                )
                == 1
            ):
                return True, player

            if (
                w in range(n - 1, width)
                and h in range(height - n + 1)
                and len(
                    set(
                        states.get(i, -1)
                        for i in range(m, m + n * (width - 1), width - 1)
                    )
                )
                == 1
            ):
                return True, player

        return False, -1

    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not self.availables:
            return True, -1  # 平局
        return False, -1


# ==========================================
# Part 2: 神经网络模型 (Neural Network)
# ==========================================
# 这个结构必须严格对应 C 语言实现的结构


class ResBlock(nn.Module):
    def __init__(self, inplanes=64, planes=64):
        super(ResBlock, self).__init__()
        # Padding=1 保证尺寸不变
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # 核心：残差连接
        out = F.relu(out)
        return out


class PolicyValueNet(nn.Module):
    def __init__(self, board_width=15, board_height=15):
        super(PolicyValueNet, self).__init__()
        self.board_width = board_width
        self.board_height = board_height

        # 1. 初始公共卷积层
        self.conv1 = nn.Conv2d(N_FEATURE, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 2. 残差塔 (5个 Block)
        self.res_blocks = nn.ModuleList([ResBlock(64, 64) for _ in range(5)])

        # 3. Policy Head (策略头: 输出概率)
        self.act_conv1 = nn.Conv2d(64, 4, kernel_size=1)  # 降维
        self.act_bn1 = nn.BatchNorm2d(4)
        self.act_fc1 = nn.Linear(
            4 * board_width * board_height, board_width * board_height
        )

        # 4. Value Head (价值头: 输出胜率)
        self.val_conv1 = nn.Conv2d(64, 2, kernel_size=1)
        self.val_bn1 = nn.BatchNorm2d(2)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # 公共部分
        x = self.conv1(state_input)
        x = self.bn1(x)
        x = F.relu(x)

        for block in self.res_blocks:
            x = block(x)

        # Policy 计算
        x_act = self.act_conv1(x)
        x_act = self.act_bn1(x_act)
        x_act = F.relu(x_act)
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        # Log Softmax 方便训练，但在推理时需要 exp 变回概率
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # Value 计算
        x_val = self.val_conv1(x)
        x_val = self.val_bn1(x_val)
        x_val = F.relu(x_val)
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))  # 输出区间 [-1, 1]

        return x_act, x_val


# ==========================================
# Part 3: MCTS 树搜索 (Simulation)
# ==========================================


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """树节点"""

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # 儿子节点 map
        self._n_visits = 0  # 访问次数
        self._Q = 0  # 平均价值
        self._u = 0  # PUCT 探索值
        self._P = prior_p  # 神经网络输出的先验概率

    def expand(self, action_priors):
        """扩展：把网络给出的概率挂载到子节点上"""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """选择：寻找得分(Q+U)最高的子节点"""
        return max(
            self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct)
        )

    def get_value(self, c_puct):
        """PUCT 算法公式"""
        self._u = (
            c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        )
        return self._Q + self._u

    def update(self, leaf_value):
        """回溯更新：更新当前节点的访问次数和平均价值"""
        self._n_visits += 1
        # Q值更新公式：累积移动平均
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

        if self._parent:
            # 这里的 -leaf_value 很关键：
            # 如果这步我是正分，那对父节点（对手）来说就是负分
            self._parent.update(-leaf_value)

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=400):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn  # 这是一个函数，指向外部的网络
        self._c_puct = c_puct
        self._n_playout = n_playout

    def get_move(self, board):
        """执行搜索，返回下一步的落子策略"""
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(board)
            node = self._root

            # 1. Select
            while not node.is_leaf():
                action, node = node.select(self._c_puct)
                state_copy.do_move(action)

            # 2. Evaluate & Expand
            # 检查游戏是否结束
            end, winner = state_copy.game_end()
            if not end:
                # 没结束，问神经网络
                action_probs, leaf_value = self._policy(state_copy)
                node.expand(action_probs)
            else:
                # 结束了，直接算分
                if winner == -1:  # 平局
                    leaf_value = 0.0
                else:
                    # leaf_value 的语义：从“当前轮到走棋的一方(state_copy.current_player)”视角评估局面
                    # 终局时，若赢家就是当前要走棋的一方 => +1，否则 => -1
                    leaf_value = 1.0 if winner == state_copy.current_player else -1.0

            # 3. Backup
            node.update(leaf_value)

        # 统计根节点的访问量
        return [(act, node._n_visits) for act, node in self._root._children.items()]

    def update_with_move(self, last_move):
        """在实际落子后，保留这棵树的对应分支（重用树），类似于缓存"""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


# ==========================================
# Part 4: 胶水层 (AlphaZero Player)
# ==========================================


class AlphaZeroPlayer(object):
    def __init__(self, model_file=None, use_gpu=True):
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.policy_value_net = PolicyValueNet().to(self.device)

        # 如果有训练好的模型，加载它
        if model_file:
            # map_location='cpu' 确保在无显卡机器上也能跑
            self.policy_value_net.load_state_dict(
                torch.load(model_file, map_location=self.device)
            )

        self.policy_value_net.eval()  # 开启评估模式

        # 初始化 MCTS，传入 "胶水函数"
        self.mcts = MCTS(self.policy_value_fn, c_puct=5, n_playout=400)

    def policy_value_fn(self, board):
        """
        胶水函数：
        Board对象 -> 神经网络输入 -> 获取概率和胜率 -> 过滤非法落子 -> 返回给MCTS
        """
        legal_positions = board.availables

        # 1. 提取 Board 特征并转为 Tensor
        current_state = np.ascontiguousarray(
            board.current_state().reshape(-1, N_FEATURE, BOARD_WIDTH, BOARD_HEIGHT)
        )

        input_tensor = torch.from_numpy(current_state).float().to(self.device)

        # 2. 推理
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(input_tensor)
            # log_softmax -> exp -> probabilities
            act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
            value = value.item()

        # 3. 只保留合法落子点的概率
        act_probs_zip = zip(legal_positions, act_probs[legal_positions])
        return act_probs_zip, value

    def get_action(self, board, return_prob=False):
        """
        对外接口：输入当前棋盘，输出 AI 决定下哪里
        """
        # 检查有没有剩余空间
        if len(board.availables) <= 0:
            print("棋盘已满")
            return -1

        # 运行 MCTS
        # 如果是第一次运行，这里会比较慢（需要初始化）
        move_visits = self.mcts.get_move(board)

        acts, visits = zip(*move_visits)

        # 根据访问次数计算最终概率 (Softmax)
        # 实际比赛中(Temperature -> 0)，直接选 visits 最大的即可
        act_probs = softmax(1.0 / 1.0 * np.log(np.array(visits) + 1e-10))

        # 随机选择（训练时）或者 选最大的（比赛时）
        # 这里演示选最大的
        move = acts[np.argmax(act_probs)]

        # 更新树的根节点，为下一步做准备
        self.mcts.update_with_move(move)

        if return_prob:
            return move, move_visits
        return move

    def reset_player(self):
        self.mcts.update_with_move(-1)


# ==========================================
# Main: 测试运行
# ==========================================
if __name__ == "__main__":
    print("正在初始化 AlphaZero 五子棋 AI...")

    # 1. 创建棋盘
    board = Board()
    board.init_board()

    # 2. 创建 AI 玩家
    # 注意：这里我们没有加载模型文件，所以它是在用“随机初始化的参数”乱下
    # 但它的逻辑是通的，只是棋力为0
    ai_player = AlphaZeroPlayer(use_gpu=False)

    print("AI 加载完成，开始自我演示一步...")

    # 假设现在棋盘中间有个黑子 (7,7) -> index 112
    center_move = 7 * 15 + 7
    board.do_move(center_move)
    ai_player.mcts.update_with_move(center_move)  # 告诉AI刚才走了这一步

    print(f"当前局面: 黑棋落在中心 (index {center_move})")
    print("AI (白棋) 正在思考 (模拟400次)...")

    # 3. AI 思考
    action = ai_player.get_action(board)

    h, w = board.move_to_location(action)
    print(f"AI 决定下在: 行{h}, 列{w} (index {action})")

    print("代码运行成功！")
