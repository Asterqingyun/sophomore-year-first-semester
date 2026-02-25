## 基本wheel和规则实现
### 25/12/25
圣诞节 写活三 if-else嵌套中 分类讨论好 希望后面不会有问题
### 25/12/27
发现有假装落子函数影响全局变量，重新定义结构体等，但是逻辑不变。（因为周五才学了结构体www)
完成禁手和判断逻辑，判断了棋子整个过程就不难办了
### 25/12/28
配了windows的C变量，主要觉得虚拟机和windows有两套环境比较安全
注意设置：$env:Path = "F:\C_environment\msys2\ucrt64\bin;" + $env:Path
基本完成人人对战的流程（包括错误回退），唯一缺陷是禁手双三和双四不行，写了测试文件发现判断是对的
设置：$env:Path = "F:\C_environment\msys2\ucrt64\bin;" + $env:Path
## 人机对战
### 26/1/1
我想设置一个C语言的递归树，就是我下了a位置之后，有一个point()函数（已经写好，可以判断在当前棋谱的自己的分数和对方的分数），然后再在每个位置扫描一遍，算出对方下出来的最高分数，两个以一个权值相减，得到本局的分数。然后这个分数和一个固定的和递归层数有关的f(n)相比，比这个大的继续递归（假设自己下了这个地方后的其他位置再扫描，然后算出下了各个位置的当前得分和对方可能拿的最高分，相减，得出每个地方可能的分值，和f(n)比较，然后加在原始位置的分上，然后继续递归），递归到最后，得出的最终分数最高的就是我想下的决策。

写了一个框架
写C侧的逻辑：随机生成参数，进行战斗，得到的再进行战斗。
gcc -std=c11 -O2 -Isrc \
  AI_training/param_worker.c \
  AI_training/play.c \
  AI_training/winning_algorithm.c \
  AI_training/random_arguments.c \
  inc/human_to_human/ban.c \
  inc/human_to_human/bound_checking_and_is_luozi.c \
  inc/human_to_human/different_types_of_chess.c \
  inc/human_to_human/look_at_surrounding.c \
  inc/human_to_human/place_chess.c \
  inc/main/show_board.c \
  -lm -o ai_worker
而后python3 train_params_parallel.py
### 26/1/3
随机生成进行训练未果，寻病终。准备一面在服务器上继续跑，一面自己先调着参

然后发现没有悔棋的话，按错了很烦，添加一个历史记录查询栈

### 26/1/4
漫漫调参路。。。。为什么会出现明明有4看不到的情况？应该是我的冲四给的太小了？对面的冲四应该大大大，我的冲四应该小一些？

对手的和我的参数同样棋型的意义不同，参数大小也应该不同
开头的好慢；次方让一些废棋容易下

继续调节参数，对手的棋很好的时候就会思考很快，因为分很低，很多地方都被剪掉了，感觉思考的不够多。
似乎在所有已有棋子的3*3/5*5范围内来进行判断会更好，这样会更快，要不也没用。
下出来了禁手，活三漏掉了情况，已修复

第一次下赢了朋友：Black(ai) wins the game!
the steps of the chess:
(7, 7) (6, 6) (6, 7) (5, 7) (7, 5) (7, 6) (5, 6) (4, 5) (6, 8) (8, 6) (8, 7) (7, 8) (9, 7) (10, 7) (10, 6) (8, 8) (9, 8) (9, 6) (8, 5) (6, 5) (6, 3) (9, 5) (5, 3) (4, 3) (4, 2) (3, 1) (5, 2) (6, 2) (5, 4) (5, 5) (4, 1) (7, 4) (5, 1) (5, 0) (3, 3) (2, 4) (4, 6) (4, 0) (3, 0) (2, 2) (2, 5) (4, 14) (3, 4) (5, 13) (0, 4) (6, 12) (3, 5) (3, 6) (9, 9) (4, 4) (9, 10) (9, 11) (8, 11) (6, 1) (10, 9) (7, 12) (11, 9) (8, 9) (8, 12) (7, 2) (11, 8) (12, 7) (11, 10) (12, 11) (11, 7) (11, 1) (11, 6)
PS F:\project\Chess>
而后输掉了
White(human) wins the game!
the steps of the chess:
(7, 7) (6, 6) (6, 7) (5, 7) (7, 5) (7, 6) (5, 6) (4, 5) (6, 8) (8, 6) (8, 7) (7, 8) (9, 7) (10, 7) (10, 6) (8, 8) (9, 8) (9, 6) (8, 5) (6, 5) (6, 3) (9, 5) (5, 3) (4, 3) (4, 2) (3, 1) (5, 2) (6, 2) (5, 4) (5, 5) (4, 1) (7, 4) (5, 1) (5, 0) (3, 3) (2, 4) (4, 6) (2, 2) (2, 5) (1, 3) (0, 4) (3, 6) (1, 4) (2, 3) (2, 0) (3, 4) (0, 1) (3, 5) (0, 2) (0, 3) (3, 7) (2, 7) (2, 8) (4, 8) (1, 9) (0, 10) (1, 8) (1, 7) (0, 8) (2, 9) (3, 11) (1, 10) (4, 10) (5, 9) (2, 10) (4, 12) (3, 8) (3, 9) (6, 9) (5, 8) (5, 10) (6, 10) (5, 11) (4, 11) (6, 12) (5, 12) (6, 13) (6, 11) (4, 13) (5, 13) (7, 13) (8, 14) (7, 11) (7, 12) (3, 12) (3, 13) (0, 7) (0, 6) (8, 12) (8, 13) (9, 11) (10, 10) (9, 10) (9, 9) (10, 9) (8, 11) (11, 10) (12, 11) (11, 11) (2, 12) (11, 9) (11, 8) (11, 12) (11, 13) (10, 11) (12, 13) (12, 9) (13, 8) (13, 9) (14, 9) (10, 12) (12, 10) (12, 12) (9, 12) (10, 13) (2, 11) (13, 12) (14, 12) (7, 9) (8, 10) (12, 8) (7, 2) (11, 7) (14, 10) (14, 11) (7, 3) (7, 0) (6, 1) (9, 4) (10, 4) (11, 3) (10, 3) (10, 2) (9, 1) (6, 4) (9, 2) (11, 4) (9, 3) (11, 6) (11, 5) (8, 2) (10, 5) (12, 6) (10, 1) (8, 3) (8, 1) (7, 1) (11, 2) (13, 4) (12, 3) (13, 6) (14, 6) (12, 7) (12, 5) (12, 4) (13, 5)
PS F:\project\Chess>

### 26/1/4-1/5(continue)
生气了，随着步数增加，搜索的递归越来越多！（因为棋子越多，棋子周围的空位越来越多）所以准备将每一次最优的就那么10来个进行递归！
gcc -std=c11 -O2 -Isrc `
  inc\ai\count_point_and_give_an_answer.c `
  inc\human_to_human\ban.c `
  inc\human_to_human\bound_checking_and_is_luozi.c `
  inc\human_to_human\different_types_of_chess.c `
  inc\human_to_human\look_at_surrounding.c `
  inc\human_to_human\place_chess.c `
  inc\main\main_for_ai_to_human.c `
  inc\main\show_board.c `
  inc\ai\kill_threat_proof.c`
  inc\ai\kill_threat_proof.h`
  -lm -o ai_vs_human

26/1/15
F:\C_environment\msys2\ucrt64\bin\gcc.exe -shared -O2 -Isrc `
  src\gomoku_forbidden_bridge.c `
  inc\human_to_human\ban.c `
  inc\human_to_human\bound_checking_and_is_luozi.c `
  inc\human_to_human\different_types_of_chess.c `
  inc\human_to_human\look_at_surrounding.c `
  inc\main\show_board.c `
  -o AI_training\alpha-zero-general\gomoku_rules.dll

26/1/18
主要是把 self-play 的瓶颈（CPU 上的 MCTS） 从“重复做无效工作”变成“少做 + 做得更合并”，所以吞吐就上来了。核心改动点在 MCTS.py + Coach.py 两处。

1) MCTS：把神经网络调用从“碎片化”变成“批量化”
在 MCTS.py 里做了：

getActionProb() 里用 pending 收集多个叶子（('nn', s, board)）
达到 mcts_batch_size 或者 sim 结束时，统一进 _evaluate_and_backup(pending)
_evaluate_and_backup() 内对 nn_boards 一次性调用 nnet.predict_batch(nn_boards)
结果：一次 forward 吃多个局面，减少 Python↔Torch 调用次数和 GPU/CPU 启动开销，吞吐更高。

2) MCTS：只遍历“合法动作列表”，不再每次扫全动作空间
仍在 MCTS.py：

扩展叶子时缓存 Vs_actions[s] = np.flatnonzero(valids).tolist()
_select() 时直接 for a in actions:，不再 for a in range(action_size) 然后再 if valids[a]
结果：大量减少 Python 循环次数和字典访问次数（这块通常是 self-play 最重的 CPU 开销）。

3) self-play 并行：从线程（GIL）切到多进程（真多核）
在 Coach.py：

增加 selfPlayBackend="process" 分支，用 multiprocessing.pool.Pool
每个进程跑一局 self-play，互不共享 MCTS 树（每局仍单线程 MCTS）
通过 initializer 让子进程加载同一份 checkpoint，避免主进程传大对象
结果：在你本机多核环境下，self-play 样本吞吐可以接近按核数提升（直到 CPU/内存带宽成为新瓶颈）。

4) 避免多进程触发 CUDA 初始化崩溃（稳定性）
还是在 Coach.py 的 _selfplay_process_init()：

当 selfPlayUseGPU=False 时，子进程禁用 CUDA，强制 self-play 走 CPU
避免 fork 子进程里 torch.cuda._lazy_init 报错
这一步主要是让“多进程 self-play”可用且稳定。

如果你想要一个可量化的解释，我也可以加一个极轻量的计时统计，把 self-play 时间拆成：_select / predict_batch / getValidMoves&nextState 三块，直接看到加速来自哪部分。

gcc -std=c11 -O2 -Isrc inc\ai\count_point_and_give_an_answer.c inc\ai\kill_threat_proof.c inc\human_to_human\ban.c inc\human_to_human\bound_checking_and_is_luozi.c inc\human_to_human\different_types_of_chess.c inc\human_to_human\look_at_surrounding.c inc\human_to_human\place_chess.c inc\main\main_combined.c inc\main\show_board.c -lm -o chess_all.exe


gcc -std=c11 -O2 -Isrc src/shared_variable_function.c inc\ai\count_point_and_give_an_answer.c inc\ai\kill_threat_proof.c inc\human_to_human\ban.c inc\human_to_human\bound_checking_and_is_luozi.c inc\human_to_human\different_types_of_chess.c inc\human_to_human\look_at_surrounding.c inc\human_to_human\place_chess.c inc\main\main_combined.c inc\main\show_board.c -lm -o chess_all.exe