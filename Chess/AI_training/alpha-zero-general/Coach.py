import logging
import os
import shutil
import _pickle
import time
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
import torch
from tqdm import tqdm
from typing import Any

from Arena import Arena
from MCTS import MCTS
from multiprocessing.pool import ThreadPool, Pool as ProcessPool

log = logging.getLogger(__name__)


_SP_GAME: Any = None
_SP_NNET: Any = None
_SP_ARGS: Any = None


def _selfplay_process_init(
    game, nnet_class, args, checkpoint_folder, checkpoint_filename, use_gpu
):
    """子进程初始化：构造 game/nnet 并加载 checkpoint。"""
    global _SP_GAME, _SP_NNET, _SP_ARGS

    _SP_GAME = game
    _SP_ARGS = args

    # 重要：子进程 self-play 默认不使用 GPU。
    # NeuralNet.__init__ 里会根据 torch.cuda.is_available() 自动选择 cuda，
    # 若这里不提前禁用，可能在子进程触发 CUDA lazy init 报错。
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        orig_is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False  # type: ignore[assignment]
        try:
            nnet = nnet_class(_SP_GAME)
        finally:
            torch.cuda.is_available = orig_is_available  # type: ignore[assignment]

        try:
            nnet.device = torch.device("cpu")
            if hasattr(nnet, "nnet"):
                nnet.nnet = nnet.nnet.to(nnet.device)
        except Exception:
            pass
    else:
        nnet = nnet_class(_SP_GAME)

    try:
        nnet.load_checkpoint(checkpoint_folder, checkpoint_filename)
    except Exception as e:
        log.warning(
            "Self-play worker failed to load checkpoint %s/%s: %s",
            checkpoint_folder,
            checkpoint_filename,
            e,
        )

    _SP_NNET = nnet


def _selfplay_process_run(_):
    """子进程执行单局 self-play，返回该局的训练样本列表。"""
    if _SP_GAME is None or _SP_NNET is None or _SP_ARGS is None:
        raise RuntimeError("Self-play process not initialized")

    game = _SP_GAME
    nnet = _SP_NNET
    args = _SP_ARGS

    mcts = MCTS(game, nnet, args)

    train_examples = []
    board = game.getInitBoard()
    cur_player = 1
    episode_step = 0

    while True:
        episode_step += 1
        canonical_board = game.getCanonicalForm(board, cur_player)
        temp = int(episode_step < args.tempThreshold)

        pi = mcts.getActionProb(canonical_board, temp=temp)
        sym = game.getSymmetries(canonical_board, pi)
        for b, p in sym:
            train_examples.append([b, cur_player, p, None])

        action = np.random.choice(len(pi), p=pi)
        board, cur_player = game.getNextState(board, cur_player, action)

        r = game.getGameEnded(board, cur_player)
        if r != 0:
            return [
                (x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in train_examples
            ]


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        # 可选的 self-play 并行 worker 数，缺省为 1（串行）
        self.numSelfPlayWorkers = self.args.get("numSelfPlayWorkers", 1)
        # self-play 并行后端：thread(默认) / process(多进程，Windows 下更能吃满 CPU)
        self.selfPlayBackend = str(self.args.get("selfPlayBackend", "thread")).lower()
        # 多进程 self-play 是否使用 GPU（默认 False，避免每个进程各占一份显存/上下文）
        self.selfPlayUseGPU = bool(self.args.get("selfPlayUseGPU", False))
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        # 每个 episode 使用独立的 MCTS 实例，避免线程之间共享树
        mcts = MCTS(self.game, self.nnet, self.args)

        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(
                board, self.curPlayer, action
            )

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [
                    (x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer)))
                    for x in trainExamples
                ]

    def _executeEpisode_worker(self, _):
        """ThreadPool 调用的包装函数，忽略传入参数，仅返回 executeEpisode 的结果。"""
        return self.executeEpisode()

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            log.info(f"Starting Iter #{i} ...")

            iterationTrainExamples = None
            try:
                # examples of the iteration
                if not self.skipFirstSelfPlay or i > 1:
                    iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                    if self.numSelfPlayWorkers <= 1:
                        for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                            iterationTrainExamples += self.executeEpisode()
                    else:
                        if self.selfPlayBackend == "process":
                            # 多进程：先把当前网络权重落盘一次，让子进程加载同一份 checkpoint
                            sp_ckpt = "selfplay_current.pth.tar"
                            self.nnet.save_checkpoint(
                                folder=self.args.checkpoint, filename=sp_ckpt
                            )

                            with ProcessPool(
                                processes=self.numSelfPlayWorkers,
                                initializer=_selfplay_process_init,
                                initargs=(
                                    self.game,
                                    self.nnet.__class__,
                                    self.args,
                                    self.args.checkpoint,
                                    sp_ckpt,
                                    self.selfPlayUseGPU,
                                ),
                            ) as pool:
                                for ex in tqdm(
                                    pool.imap_unordered(
                                        _selfplay_process_run, range(self.args.numEps)
                                    ),
                                    total=self.args.numEps,
                                    desc="Self Play",
                                ):
                                    iterationTrainExamples += ex
                        else:
                            with ThreadPool(self.numSelfPlayWorkers) as pool:
                                for ex in tqdm(
                                    pool.imap_unordered(
                                        self._executeEpisode_worker,
                                        range(self.args.numEps),
                                    ),
                                    total=self.args.numEps,
                                    desc="Self Play",
                                ):
                                    iterationTrainExamples += ex

                    self.trainExamplesHistory.append(iterationTrainExamples)
            except KeyboardInterrupt:
                log.warning(
                    "KeyboardInterrupt: saving trainExamples snapshot and stopping..."
                )
                # 若本轮有部分数据，也加入历史并落盘，方便下次继续训
                if iterationTrainExamples is not None and (
                    not self.trainExamplesHistory
                    or self.trainExamplesHistory[-1] is not iterationTrainExamples
                ):
                    self.trainExamplesHistory.append(iterationTrainExamples)

                # 落盘快照：latest.examples
                self.saveTrainExamplesSnapshot(tag="latest")
                # 同时保存当前权重，避免 best/temp 不一致
                try:
                    self.nnet.save_checkpoint(
                        folder=self.args.checkpoint, filename="interrupt.pth.tar"
                    )
                except Exception:
                    pass
                return

            if (
                len(self.trainExamplesHistory)
                > self.args.numItersForTrainExamplesHistory
            ):
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}"
                )
                # 兼容 list / deque / dict（某些旧 examples 可能用 torch.save 保存为 dict）
                if isinstance(self.trainExamplesHistory, dict):
                    # 尝试按最小 key 删除（常见是迭代号/时间戳）
                    try:
                        k0 = sorted(self.trainExamplesHistory.keys())[0]
                        if k0 in self.trainExamplesHistory:
                            del self.trainExamplesHistory[k0]
                    except Exception:
                        # 兜底：清空，避免死循环
                        self.trainExamplesHistory = []
                elif isinstance(self.trainExamplesHistory, deque):
                    self.trainExamplesHistory.popleft()
                else:
                    self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)
            # 额外保存一个稳定路径的快照，便于中途查看/恢复
            self.saveTrainExamplesSnapshot(tag="latest")

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                # 容错：历史里可能混入非迭代容器（例如误把权重文件当 examples 加载）
                if isinstance(e, (list, tuple, deque)):
                    trainExamples.extend(e)
                else:
                    log.warning(
                        "Skipping non-iterable entry in trainExamplesHistory: type=%s",
                        type(e),
                    )
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            self.pnet.load_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info("PITTING AGAINST PREVIOUS VERSION")
            arena = Arena(
                lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                self.game,
            )
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (nwins, pwins, draws))
            if (
                pwins + nwins == 0
                or float(nwins) / (pwins + nwins) < self.args.updateThreshold
            ):
                log.info("REJECTING NEW MODEL")
                self.nnet.load_checkpoint(
                    folder=self.args.checkpoint, filename="temp.pth.tar"
                )
            else:
                log.info("ACCEPTING NEW MODEL")
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename=self.getCheckpointFile(i)
                )
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename="best.pth.tar"
                )

                # 同步保存 best.pth.tar.examples，保证下次重启能续跑 trainExamplesHistory。
                # 注意：examples 是在本轮训练前基于 (i-1) 迭代收集/保存的（见 saveTrainExamples(i-1)）。
                src_examples = os.path.join(
                    self.args.checkpoint, self.getCheckpointFile(i - 1) + ".examples"
                )
                dst_examples = os.path.join(
                    self.args.checkpoint, "best.pth.tar.examples"
                )
                if os.path.isfile(src_examples):
                    shutil.copyfile(src_examples, dst_examples)

    def getCheckpointFile(self, iteration):
        return "checkpoint_" + str(iteration) + ".pth.tar"

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")

        payload = {
            "trainExamplesHistory": self.trainExamplesHistory,
            "saved_time": time.time(),
            "iteration": int(iteration),
        }

        tmpname = filename + ".tmp"
        with open(tmpname, "wb+") as f:
            Pickler(f, protocol=4).dump(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmpname, filename)

    def saveTrainExamplesSnapshot(self, tag: str = "latest"):
        """保存一个固定文件名的样本快照，便于中断恢复。"""
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, f"{tag}.examples")

        payload = {
            "trainExamplesHistory": self.trainExamplesHistory,
            "saved_time": time.time(),
            "tag": str(tag),
        }

        tmpname = filename + ".tmp"
        with open(tmpname, "wb+") as f:
            Pickler(f, protocol=4).dump(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmpname, filename)

    def loadTrainExamples(self):
        modelFile = os.path.join(
            self.args.load_folder_file[0], self.args.load_folder_file[1]
        )
        examplesFile = modelFile + ".examples"

        # 若指定的 examples 不存在，尝试回退到固定快照 latest.examples（支持中断恢复）
        if not os.path.isfile(examplesFile):
            fallback = os.path.join(self.args.load_folder_file[0], "latest.examples")
            if os.path.isfile(fallback):
                log.warning(
                    'File "%s" not found; using fallback "%s".',
                    examplesFile,
                    fallback,
                )
                examplesFile = fallback
            else:
                log.warning(
                    'File "%s" with trainExamples not found! Starting fresh.',
                    examplesFile,
                )
                self.trainExamplesHistory = []
                self.skipFirstSelfPlay = False
                return

        log.info("File with trainExamples found. Loading it...")

        def _looks_like_state_dict(obj):
            """粗略判断 obj 是否像 PyTorch state_dict（OrderedDict[str, Tensor]）。"""
            if not isinstance(obj, dict) or not obj:
                return False
            keys = list(obj.keys())
            if not all(isinstance(k, str) for k in keys[:50]):
                return False
            sample = keys[:50]
            score = sum(
                (
                    "." in k
                    or k.endswith(".weight")
                    or k.endswith(".bias")
                    or "running_mean" in k
                    or "running_var" in k
                    or "num_batches_tracked" in k
                )
                for k in sample
            ) / max(1, len(sample))
            return score >= 0.5

        # 有些历史 examples 可能是用 torch.save 保存的（zip 容器，文件头为 b'PK'）。
        # 原版 alpha-zero-general 是纯 pickle；这里根据文件头自动选择加载方式。
        def _normalize_history(obj):
            """把从磁盘加载的对象规范化成 list[deque/list]。"""
            if obj is None:
                return []

            # 有些实现可能保存为 dict
            if isinstance(obj, dict):
                # 误把权重 state_dict 当作 examples：直接拒绝
                if _looks_like_state_dict(obj):
                    return []

                for key in ("trainExamplesHistory", "examples", "history", "data"):
                    if key in obj:
                        obj = obj[key]
                        break
                else:
                    # 若是 {iter: examples} 这种，按 key 排序转成 list（仅接受整数 key）
                    try:
                        keys = list(obj.keys())
                        if keys and all(isinstance(k, (int, np.integer)) for k in keys):
                            obj = [obj[k] for k in sorted(keys)]
                        else:
                            return []
                    except Exception:
                        return []

            # deque/tuple 等转 list
            if isinstance(obj, (tuple, set)):
                obj = list(obj)
            elif (
                hasattr(obj, "__iter__")
                and not isinstance(obj, list)
                and not isinstance(obj, (str, bytes))
            ):
                # deque 等
                try:
                    obj = list(obj)
                except Exception:
                    pass

            if not isinstance(obj, list):
                return []
            return obj

        try:
            with open(examplesFile, "rb") as f:
                head2 = f.read(2)

            if head2 == b"PK":
                loaded = torch.load(
                    examplesFile, map_location="cpu", weights_only=False
                )
            else:
                with open(examplesFile, "rb") as f:
                    loaded = Unpickler(f).load()

            # 若加载出来其实是权重 state_dict，说明 examples 文件被误命名/误保存，直接忽略
            if _looks_like_state_dict(loaded):
                log.warning(
                    "trainExamples file %s looks like a PyTorch state_dict; ignoring it.",
                    examplesFile,
                )
                self.trainExamplesHistory = []
                self.skipFirstSelfPlay = False
                return

            self.trainExamplesHistory = _normalize_history(loaded)

            # 启动时如果历史过长（例如以前保存了很多轮），直接裁剪到最近 N 轮
            max_hist = int(getattr(self.args, "numItersForTrainExamplesHistory", 20))
            if (
                isinstance(self.trainExamplesHistory, list)
                and len(self.trainExamplesHistory) > max_hist
            ):
                self.trainExamplesHistory = self.trainExamplesHistory[-max_hist:]

            log.info("Loading done!")
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

        except (_pickle.UnpicklingError, Exception) as e:
            log.warning(
                "Failed to load trainExamples from %s: %s. Will start fresh examples.",
                examplesFile,
                e,
            )
            self.trainExamplesHistory = []
            self.skipFirstSelfPlay = False
