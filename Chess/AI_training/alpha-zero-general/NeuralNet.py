import os
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


N_FEATURE = 3

log = logging.getLogger(__name__)


class ResBlock(nn.Module):
    def __init__(self, inplanes: int = 64, planes: int = 64):
        super().__init__()
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
        out = out + residual
        out = F.relu(out)
        return out


class GomokuNNet(nn.Module):
    """对齐到 AI_training/model.py 的 PolicyValueNet 结构（便于 C++/ONNX/TorchScript 对齐）。

    输入: (B,3,n,n)
        - C0: 当前玩家棋子 (board > 0)
        - C1: 对手棋子 (board < 0)
        - C2: 当前回合平面（canonical 视角下恒为 1）

    结构:
        - conv1: 3x3, 3->64
        - res tower: 5x ResBlock(64)
        - policy head: 1x1, 64->4, BN, FC -> action_size
        - value head: 1x1, 64->2, BN, FC64 -> FC1 -> tanh
    """

    def __init__(self, game, n_res_blocks: int = 5):
        super().__init__()

        self.n, _ = game.getBoardSize()
        self.action_size = game.getActionSize()

        # 1) 公共部分
        self.conv1 = nn.Conv2d(N_FEATURE, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.res_blocks = nn.ModuleList(
            [ResBlock(64, 64) for _ in range(int(n_res_blocks))]
        )

        # 2) Policy head
        self.act_conv1 = nn.Conv2d(64, 4, kernel_size=1, bias=True)
        self.act_bn1 = nn.BatchNorm2d(4)
        self.act_fc1 = nn.Linear(4 * self.n * self.n, self.action_size)

        # 3) Value head
        self.val_conv1 = nn.Conv2d(64, 2, kernel_size=1, bias=True)
        self.val_bn1 = nn.BatchNorm2d(2)
        self.val_fc1 = nn.Linear(2 * self.n * self.n, 64)
        self.val_fc2 = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        # 更稳定的初始化（AlphaZero/ResNet 常用）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 公共部分
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        for block in self.res_blocks:
            x = block(x)

        # Policy
        x_act = self.act_conv1(x)
        x_act = self.act_bn1(x_act)
        x_act = F.relu(x_act)
        x_act = x_act.reshape(x_act.size(0), -1)
        x_act = self.act_fc1(x_act)
        log_pi = F.log_softmax(x_act, dim=1)

        # Value
        x_val = self.val_conv1(x)
        x_val = self.val_bn1(x_val)
        x_val = F.relu(x_val)
        x_val = x_val.reshape(x_val.size(0), -1)
        x_val = F.relu(self.val_fc1(x_val))
        v = torch.tanh(self.val_fc2(x_val))

        return log_pi, v


class NeuralNet:
    """AlphaZeroGeneral 期望的神经网络封装：train / predict / save / load。"""

    def __init__(
        self,
        game,
        lr=1e-3,
        epochs=20,
        batch_size=64,
        weight_decay=1e-4,
        grad_clip_norm=5.0,
        log_losses=True,
    ):
        self.game = game
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.grad_clip_norm = (
            float(grad_clip_norm) if grad_clip_norm is not None else None
        )
        self.log_losses = bool(log_losses)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nnet = GomokuNNet(game).to(self.device)
        self.optimizer = optim.Adam(
            self.nnet.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def _loss_pi(self, targets, outputs_log_probs):
        # targets, outputs: (batch, action_size)
        return -torch.mean(torch.sum(targets * outputs_log_probs, dim=1))

    def _loss_v(self, targets, outputs):
        # targets: (batch,), outputs: (batch,1)
        return F.mse_loss(outputs.view(-1), targets)

    def train(self, examples):
        """examples: list of (board, pi, v)，board 为 canonical board。"""

        if not examples:
            return

        self.nnet.train()

        boards, pis, vs = list(zip(*examples))
        boards = np.array(boards)
        pis = np.array(pis, dtype=np.float32)
        vs = np.array(vs, dtype=np.float32)

        n_samples = boards.shape[0]
        indices = np.arange(n_samples)

        for ep in range(self.epochs):
            np.random.shuffle(indices)

            sum_pi = 0.0
            sum_v = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_boards = self._boards_to_tensor(boards[batch_idx])
                batch_pis = torch.tensor(
                    pis[batch_idx], dtype=torch.float32, device=self.device
                )
                batch_vs = torch.tensor(
                    vs[batch_idx], dtype=torch.float32, device=self.device
                )

                out_log_pi, out_v = self.nnet(batch_boards)

                l_pi = self._loss_pi(batch_pis, out_log_pi)
                l_v = self._loss_v(batch_vs, out_v)
                loss = l_pi + l_v

                self.optimizer.zero_grad()
                loss.backward()

                if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.nnet.parameters(), max_norm=self.grad_clip_norm
                    )
                self.optimizer.step()

                sum_pi += float(l_pi.item())
                sum_v += float(l_v.item())
                n_batches += 1

            if self.log_losses and n_batches > 0:
                log.info(
                    "[NeuralNet] epoch %d/%d pi_loss=%.4f v_loss=%.4f",
                    ep + 1,
                    self.epochs,
                    sum_pi / n_batches,
                    sum_v / n_batches,
                )

    def predict(self, board):
        """board: canonical board, shape (H, W)。返回 (pi, v)。"""

        self.nnet.eval()
        with torch.no_grad():
            b = self._boards_to_tensor(np.expand_dims(np.asarray(board), axis=0))
            b = b.contiguous()
            log_pi, v = self.nnet(b)

        pi = torch.exp(log_pi).cpu().numpy()[0]  # (action_size,)
        v = v.item()
        return pi, v

    def predict_batch(self, boards):
        """批量预测：boards 为形状 (B, H, W) 的 numpy 数组或可迭代对象，返回 (pis, vs)。"""

        boards = np.array(boards)

        self.nnet.eval()
        with torch.no_grad():
            b = self._boards_to_tensor(boards)  # (B,3,H,W)
            b = b.contiguous()
            log_pi, v = self.nnet(b)

        pis = torch.exp(log_pi).cpu().numpy()  # (B, action_size)
        vs = v.view(-1).cpu().numpy()  # (B,)
        return pis, vs

    def _boards_to_tensor(self, boards_np: np.ndarray) -> torch.Tensor:
        """把 (B,H,W) 的 canonical board 转成 (B,3,H,W) float32 张量。"""

        b = np.asarray(boards_np)
        if b.ndim != 3:
            raise ValueError(f"boards must have shape (B,H,W), got {b.shape}")

        own = (b > 0).astype(np.float32)
        opp = (b < 0).astype(np.float32)
        turn = np.ones_like(own, dtype=np.float32)

        stacked = np.ascontiguousarray(np.stack([own, opp, turn], axis=1))  # (B,3,H,W)
        t = torch.from_numpy(stacked)
        return t.to(self.device, dtype=torch.float32)

    def save_checkpoint(self, folder, filename):
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        torch.save(self.nnet.state_dict(), filepath)

    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        map_location = self.device
        state_dict = torch.load(filepath, map_location=map_location)

        try:
            self.nnet.load_state_dict(state_dict)
            return
        except RuntimeError:
            pass

        new_sd = self.nnet.state_dict()

        def _copy_if_match(dst_key, src_tensor):
            if dst_key not in new_sd:
                return
            if new_sd[dst_key].shape != src_tensor.shape:
                return
            new_sd[dst_key] = src_tensor

        # 迁移来源 1：上一版 3 通道 backbone.*（ConvBlock/ResidualBlock 版本）
        if any(k.startswith("backbone.") for k in state_dict.keys()):
            # conv1 <- backbone.0
            w = state_dict.get("backbone.0.conv.weight", None)
            if w is not None and w.shape == new_sd["conv1.weight"].shape:
                new_sd["conv1.weight"] = w
            _copy_if_match(
                "bn1.weight",
                state_dict.get("backbone.0.bn.weight", new_sd["bn1.weight"]),
            )
            _copy_if_match(
                "bn1.bias", state_dict.get("backbone.0.bn.bias", new_sd["bn1.bias"])
            )
            _copy_if_match(
                "bn1.running_mean",
                state_dict.get(
                    "backbone.0.bn.running_mean", new_sd["bn1.running_mean"]
                ),
            )
            _copy_if_match(
                "bn1.running_var",
                state_dict.get("backbone.0.bn.running_var", new_sd["bn1.running_var"]),
            )

            # res blocks：取 backbone 的 3 个 ResidualBlock 映射到前 3 个
            mapping = {
                0: 2,
                1: 5,
                2: 8,
            }
            for dst_i, src_idx in mapping.items():
                _copy_if_match(
                    f"res_blocks.{dst_i}.conv1.weight",
                    state_dict.get(
                        f"backbone.{src_idx}.conv1.weight",
                        new_sd[f"res_blocks.{dst_i}.conv1.weight"],
                    ),
                )
                _copy_if_match(
                    f"res_blocks.{dst_i}.bn1.weight",
                    state_dict.get(
                        f"backbone.{src_idx}.bn1.weight",
                        new_sd[f"res_blocks.{dst_i}.bn1.weight"],
                    ),
                )
                _copy_if_match(
                    f"res_blocks.{dst_i}.bn1.bias",
                    state_dict.get(
                        f"backbone.{src_idx}.bn1.bias",
                        new_sd[f"res_blocks.{dst_i}.bn1.bias"],
                    ),
                )
                _copy_if_match(
                    f"res_blocks.{dst_i}.bn1.running_mean",
                    state_dict.get(
                        f"backbone.{src_idx}.bn1.running_mean",
                        new_sd[f"res_blocks.{dst_i}.bn1.running_mean"],
                    ),
                )
                _copy_if_match(
                    f"res_blocks.{dst_i}.bn1.running_var",
                    state_dict.get(
                        f"backbone.{src_idx}.bn1.running_var",
                        new_sd[f"res_blocks.{dst_i}.bn1.running_var"],
                    ),
                )

                _copy_if_match(
                    f"res_blocks.{dst_i}.conv2.weight",
                    state_dict.get(
                        f"backbone.{src_idx}.conv2.weight",
                        new_sd[f"res_blocks.{dst_i}.conv2.weight"],
                    ),
                )
                _copy_if_match(
                    f"res_blocks.{dst_i}.bn2.weight",
                    state_dict.get(
                        f"backbone.{src_idx}.bn2.weight",
                        new_sd[f"res_blocks.{dst_i}.bn2.weight"],
                    ),
                )
                _copy_if_match(
                    f"res_blocks.{dst_i}.bn2.bias",
                    state_dict.get(
                        f"backbone.{src_idx}.bn2.bias",
                        new_sd[f"res_blocks.{dst_i}.bn2.bias"],
                    ),
                )
                _copy_if_match(
                    f"res_blocks.{dst_i}.bn2.running_mean",
                    state_dict.get(
                        f"backbone.{src_idx}.bn2.running_mean",
                        new_sd[f"res_blocks.{dst_i}.bn2.running_mean"],
                    ),
                )
                _copy_if_match(
                    f"res_blocks.{dst_i}.bn2.running_var",
                    state_dict.get(
                        f"backbone.{src_idx}.bn2.running_var",
                        new_sd[f"res_blocks.{dst_i}.bn2.running_var"],
                    ),
                )

        # 迁移来源 2：更旧版 1 通道 conv_layers.*（至少迁移首层卷积到 conv1）
        if any(k.startswith("conv_layers.") for k in state_dict.keys()):
            w = state_dict.get("conv_layers.0.weight", None)
            if w is not None and w.ndim == 4 and w.shape[1] == 1:
                # 扩到 3 通道，均分复制
                w3 = w.repeat(1, 3, 1, 1) / 3.0
                if w3.shape == new_sd["conv1.weight"].shape:
                    new_sd["conv1.weight"] = w3

        missing, unexpected = self.nnet.load_state_dict(new_sd, strict=False)
        print(
            f"[load_checkpoint] Loaded checkpoint partially into new PolicyValueNet. "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
