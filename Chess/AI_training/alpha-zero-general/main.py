import logging
import os

import coloredlogs

from Coach import Coach
from Game import GomokuGame as Game
from NeuralNet import NeuralNet as nn
from utils import dotdict

log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")  # 改成 DEBUG 可以看到更多信息

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 统一把模型/样本目录固定到项目根目录下的 temp，避免不同工作目录导致相对路径混乱
CHESS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
TEMP_DIR = os.path.join(CHESS_DIR, "temp")

args = dotdict(
    {
        "numIters": 500,  # 训练总迭代轮数（缩小到个人可跑完的规模）
        "numEps": 50,  # 每轮自博弈局数
        "tempThreshold": 15,  # 前多少步使用高温度（更探索）
        "updateThreshold": 0.6,  # 新网络在对战中胜率超过该阈值才被接受
        "maxlenOfQueue": 200000,  # 训练样本队列最大长度
        "numMCTSSims": 80,  # 每一步 MCTS 模拟次数
        "arenaCompare": 40,  # 新旧网络对战局数
        "cpuct": 1,
        "mcts_batch_size": 64,  # 每批次并行评估的叶子数量
        "numSelfPlayWorkers": 8,  # self-play 并行线程数（1 表示串行）
        # self-play 并行后端：thread(默认) / process(多进程更能吃满 CPU，但有额外开销)
        "selfPlayBackend": "thread",
        # 多进程 self-play 是否使用 GPU：默认 False，避免每个进程都占用一份显存
        "selfPlayUseGPU": False,
        "checkpoint": TEMP_DIR,  # 模型与样本保存目录（用绝对路径避免工作目录变化）
        "load_model": False,  # 是否从已有模型继续训练
        "load_folder_file": (TEMP_DIR, "best.pth.tar"),
        "numItersForTrainExamplesHistory": 20,
    }
)


def main():
    log.info("Loading %s...", Game.__name__)
    g = Game(15)  # 15x15 五子棋

    log.info("Loading %s...", nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info(
            'Loading checkpoint "%s/%s"...',
            args.load_folder_file[0],
            args.load_folder_file[1],
        )
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning("Not loading a checkpoint!")

    log.info("Loading the Coach...")
    c = Coach(g, nnet, args)

    if args.load_model:
        examples_path = os.path.join(
            args.load_folder_file[0], args.load_folder_file[1] + ".examples"
        )
        if os.path.isfile(examples_path):
            log.info("Loading 'trainExamples' from file...")
            c.loadTrainExamples()
        else:
            log.warning(
                "trainExamples not found at %s; will start fresh examples.",
                examples_path,
            )

    log.info("Starting the learning process 🎉")
    c.learn()


if __name__ == "__main__":
    main()
