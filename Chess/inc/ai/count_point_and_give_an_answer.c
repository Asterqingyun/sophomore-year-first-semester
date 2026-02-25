#include "kill_threat_proof.h"
//----1.一些辅助函数（因为和其他的一些宏定义有关，所以放在开头，避免编译错误）
// --- 模拟落子检查五连珠 ---
int check_5_if_place(int board[SIZE][SIZE], int x, int y, int player)
{
    int old = board[x][y];
    board[x][y] = player;
    int r = check_5(x, y, player);
    board[x][y] = old;
    return r;
}
// --- 最终合法性判断函数 ---
static inline int is_final_legal_move(
    int board[SIZE][SIZE],
    int x, int y,
    int player,
    int first_player)
{
    if (return_what_is_at(x, y) != EMPTY)
        return 0;
    return give_point(board, x, y, player, first_player) != NEGATIVE_INFINITY;
}
//----2.include and define（广义define,包括根据current player选择常量值的函数)
#include "shared_variable_function.h"

#include <stdint.h>
#define MAX_LAYER 3
// 当检测到对手威胁很高时，额外加深搜索层数
#define EXTRA_DEPTH_ON_THREAT 2
// 加深后的最大层数上限（防止分支爆炸）
#define MAX_LAYER_CAP (MAX_LAYER + EXTRA_DEPTH_ON_THREAT)
// 有一些必须要停下来的情况
// 固定的棋型权重（与早期版本一致）

// 返回当前玩家的对手颜色
static int opposite_player(int current_player)
{
    return (current_player == BLACK) ? WHITE : BLACK;
}
long long int POINT_SLEEP_FOUR(int who)
{
    if (who == BLACK)
    {
        return 10000;
    }
    else
    {
        return 30000;
    }
}
#define POINT_OPEN_FOUR 1000000

long long int POINT_OPEN_THREE(int who)
{
    if (who == BLACK)
    {
        return 10000;
    }
    else
    {
        return 30000;
    }
}
#define POINT_SLEEP_THREE 300
#define POINT_OPEN_TWO 80
#define POINT_SLEEP_TWO 10
#define BASE -10000
// 简单的衰减因子：第一层不衰减，之后按 1/layer 衰减
#define DEFENSE_TRIGGER_THREAT (POINT_OPEN_THREE(WHITE))

// 防守奖励权重：bonus = (opp_now - opp_after) * NUM / DEN
#define DEFENSE_BONUS_NUM 3
#define DEFENSE_BONUS_DEN 2

// “必须堵”的惩罚：对手已接近活四时，你走完仍不降下来 -> 极大扣分
#define MUST_BLOCK_PENALTY 600000

// “必须堵双三”的惩罚：对手已具备双活三级别威胁，但你走完仍然保持该级别 -> 强行逼你优先去堵
#define MUST_BLOCK_DOUBLE_THREE_THREAT (POINT_OPEN_THREE(WHITE) * 2)
#define MUST_BLOCK_DOUBLE_THREE_PENALTY (POINT_OPEN_THREE(WHITE) * 10)
// 衰减因子函数
static double Decay_Factor(int layer)
{ /*
  if (layer <= 1)
  {
      return 1.0;
  }
  return 1.0 / (double)layer;
  */
    return 1 / (double)pow(layer, 0.5);
}
// 阈值函数
static long long Threshold(int layer)
{
    int base = BASE;
    if (step < 3)
    {
        return 0;
    }
    base = layer * 1000;

    return base;
}

// danger 模式：对手威胁越大，阈值越低（更负）=> 越少剪枝、愿意更深入
// 你可以按实际分数尺度调整这两个量
#define DANGER_THRESHOLD_DELTA_FOUR POINT_OPEN_FOUR
#define DANGER_THRESHOLD_DELTA_FIVE 30000

// 相对阈值容差：给 shallow 估值留误差空间（越大越不容易剪掉“唯一出路”的防守棋）
#define THRESH_TOLERANCE 5000
// ================================
// AI 搜索统计（用于显示递归层数）
// ================================
int ai_last_max_depth_reached = 0;
long long ai_last_nodes_visited = 0;
long long ai_last_nodes_by_layer[AI_STATS_LAYER_CAP] = {0};
int ai_last_danger_triggered = 0;
long long ai_last_root_opp_threat = 0;
int ai_last_effective_max_layer = 0;
long long ai_last_cache_lookups = 0;
long long ai_last_cache_hits = 0;

// ================================
// -------2.置换表（缓存）：Zobrist Hash + 固定大小哈希表：当算到相同的情况棋局的时候直接使用缓存
// ================================

static uint64_t zobrist[SIZE][SIZE][3];
static int zobrist_inited = 0;

// 2.1根局面 hash（在 give_an_answer 中计算一次，避免每个候选点都全盘扫描）
static uint64_t ai_root_board_hash = 0;

static uint64_t xorshift64(uint64_t *state)

{
    // 用于生成 Zobrist 哈希表里的随机数
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ULL;
}

static void init_zobrist_if_needed(void)
//  2.2初始化 Zobrist 哈希表
{
    if (zobrist_inited)
    {
        return;
    }
    uint64_t seed = 0xC0FFEE1234ABCDEFULL;
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                zobrist[i][j][k] = xorshift64(&seed);
            }
        }
    }
    zobrist_inited = 1;
}

static uint64_t compute_board_hash(int board[SIZE][SIZE])
//  2.3计算当前棋盘的 Zobrist 哈希值
{
    init_zobrist_if_needed();
    uint64_t h = 0;
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            int p = board[i][j];
            if (p != EMPTY)
            {
                h ^= zobrist[i][j][p];
            }
        }
    }
    return h;
}
/*2.4置换表（Transposition Table, TT）

用途：

记忆已经计算过的棋盘状态及其评估值，避免重复搜索

提升 AI 速度

*/
typedef struct
{
    uint64_t key;
    long long value;
} TTEntry;

// 2^19 = 524288 entries (~8MB for key+value)
#define TT_SIZE (1u << 19)
#define TT_MASK (TT_SIZE - 1u)
static TTEntry tt[TT_SIZE];

static uint64_t make_tt_key(uint64_t board_hash, int player, int layer, int max_layer, int first_player)
// 生成置换表的 key
{
    // 注意：Threshold/ThresholdDyn 依赖全局 step，所以把 step 也纳入 key，避免跨回合误命中
    uint64_t k = board_hash;
    k ^= (uint64_t)(player & 3) << 62;
    k ^= (uint64_t)(layer & 63) << 56;
    k ^= (uint64_t)(max_layer & 63) << 50;
    k ^= (uint64_t)(first_player & 3) << 48;
    k ^= (uint64_t)(step & 0x3FF) << 32;
    // 再做一次轻混合
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    return k;
}

static int tt_get(uint64_t key, long long *out)
// 查找置换表
{
    ai_last_cache_lookups++;
    TTEntry *e = &tt[(unsigned)(key & TT_MASK)];
    if (e->key == key)
    {
        ai_last_cache_hits++;
        *out = e->value;
        return 1;
    }
    return 0;
}

static void tt_put(uint64_t key, long long value)
// 存入置换表
{
    TTEntry *e = &tt[(unsigned)(key & TT_MASK)];
    e->key = key;
    e->value = value;
}

//-----3.AI 统计相关函数
static void ai_stats_reset(void)
// 3.1重置 AI 统计数据
{
    ai_last_max_depth_reached = 0;
    ai_last_nodes_visited = 0;
    for (int i = 0; i < AI_STATS_LAYER_CAP; i++)
    {
        ai_last_nodes_by_layer[i] = 0;
    }
    ai_last_danger_triggered = 0;
    ai_last_root_opp_threat = 0;
    ai_last_effective_max_layer = 0;
    ai_last_cache_lookups = 0;
    ai_last_cache_hits = 0;
}

static void ai_stats_visit(int layer)
// 3.2记录访问的节点数和最大深度
{
    ai_last_nodes_visited++;
    if (layer > ai_last_max_depth_reached)
    {
        ai_last_max_depth_reached = layer;
    }
    if (layer >= 0 && layer < AI_STATS_LAYER_CAP)
    {
        ai_last_nodes_by_layer[layer]++;
    }
}
//---4.剪枝的方法：阈值+TOP-K
static long long ThresholdDyn(int layer, long long opp_threat, int first_player)
// 4.1动态阈值函数：结合绝对阈值和相对阈值
{
    // 1) 平稳局面：沿用基础阈值（绝对阈值）
    long long base_threshold = Threshold(layer);

    // 2) 危险局面：相对阈值（跟随 opp_threat）：防止因为阈值减掉了唯一防守点
    //    “不防守/不处理威胁”的基准大约是 -(opp_threat * OPP_FACTOR)
    //    只有当某步比这个基准更烂很多(超过容差)才剪掉。
    long long relative_threshold;
    if (opp_threat >= POSITIVE_INFINITY / 8)
    {
        relative_threshold = NEGATIVE_INFINITY / 4;
    }
    else
    {
        relative_threshold = -((long long)(opp_threat * OPP_FACTOR(first_player))) - THRESH_TOLERANCE;
    }

    // 取更“宽容”的门槛（更小/更负）
    return (relative_threshold < base_threshold) ? relative_threshold : base_threshold;
}

// ================================
// 威胁感知防守加成：危险局面更偏向“堵对面”
// ================================

// 触发“更偏防守”的威胁阈值（更低=更爱防守）

static long long clamp_ll(long long v, long long lo, long long hi)
{
    if (v < lo)
        return lo;
    if (v > hi)
        return hi;
    return v;
}

// 4.2当前层要保留的候选步数（Top-K）。暂时固定为 15，之后可以按 layer 动态调整。
static int amount(int layer)
{
    (void)layer;
    int k;
    if (layer == 1)
    {
        k = 19;
    }
    else if (layer == 2)
    {
        k = 18;
    }
    else if (layer == 3)
    {
        k = 16;
    }
    else if (layer == 4)
    {
        k = 15;
    }
    else
    {
        k = 10;
    }
    return k;
}

int compare_dian_and_fenshu(const void *a, const void *b)
// 将 dian_and_fenshu 结构体数组按 fenshu 字段（分数）从高到低排序
{
    dian_and_fenshu *dian_a = (dian_and_fenshu *)a;
    dian_and_fenshu *dian_b = (dian_and_fenshu *)b;
    if (dian_a->fenshu < dian_b->fenshu)
        return 1;
    else if (dian_a->fenshu > dian_b->fenshu)
        return -1;
    else
        return 0;
}
//----5.提高搜索速度：只搜索旁边3*3or5*5范围内的点
int is_neighbored_3_3(int arrayForInnerBoardLayout[SIZE][SIZE], int x, int y)
{
    if (in_bound(x - 1, y) && arrayForInnerBoardLayout[x - 1][y] != EMPTY)
    {
        return 1;
    }
    else if (in_bound(x + 1, y) && arrayForInnerBoardLayout[x + 1][y] != EMPTY)
    {
        return 1;
    }
    else if (in_bound(x, y - 1) && arrayForInnerBoardLayout[x][y - 1] != EMPTY)
    {
        return 1;
    }
    else if (in_bound(x, y + 1) && arrayForInnerBoardLayout[x][y + 1] != EMPTY)
    {
        return 1;
    }
    else if (in_bound(x - 1, y - 1) && arrayForInnerBoardLayout[x - 1][y - 1] != EMPTY)
    {
        return 1;
    }
    else if (in_bound(x - 1, y + 1) && arrayForInnerBoardLayout[x - 1][y + 1] != EMPTY)
    {
        return 1;
    }
    else if (in_bound(x + 1, y - 1) && arrayForInnerBoardLayout[x + 1][y - 1] != EMPTY)
    {
        return 1;
    }
    else if (in_bound(x + 1, y + 1) && arrayForInnerBoardLayout[x + 1][y + 1] != EMPTY)
    {
        return 1;
    }
    return 0;
}
int is_neighbored_5_5(int arrayForInnerBoardLayout[SIZE][SIZE], int x, int y)
{
    for (int i = x - 2; i <= x + 2; i++)
    {
        for (int j = y - 2; j <= y + 2; j++)
        {
            if (in_bound(i, j) && arrayForInnerBoardLayout[i][j] != EMPTY)
            {
                return 1;
            }
        }
    }
    return 0;
}
//----6.核心评估函数：6.1一步落子分
long long int give_point(int arrayForInnerBoardLayout[SIZE][SIZE], int x, int y, int current_player, int first_player)
{
    // 1. 模拟落子
    int original_val = arrayForInnerBoardLayout[x][y];
    arrayForInnerBoardLayout[x][y] = current_player;

    // 2. 规则判定 (长连禁手 > 五连胜 > 33/44禁手)
    if (current_player == first_player)
    {
        if (is_banned_by_long_connected(x, y, current_player))
        {
            arrayForInnerBoardLayout[x][y] = original_val;
            return NEGATIVE_INFINITY;
        }
    }
    // 检查五连
    if (check_5(x, y, current_player))
    {
        arrayForInnerBoardLayout[x][y] = original_val;
        return POSITIVE_INFINITY;
    }
    // 33/44禁手
    if (current_player == first_player)
    {
        if (is_banned_after(x, y, current_player))
        {
            arrayForInnerBoardLayout[x][y] = original_val;
            return NEGATIVE_INFINITY;
        }
    }

    // 3. 还原棋盘
    arrayForInnerBoardLayout[x][y] = original_val;

    // 4. 计算形状分
    return POINT_OPEN_FOUR * num_of_huo_si(x, y, current_player) +
           POINT_SLEEP_FOUR(current_player) * num_of_chong_si(x, y, current_player) +
           POINT_OPEN_THREE(current_player) * num_of_huo_san(x, y, current_player) +
           POINT_SLEEP_THREE * num_of_mian_san(x, y, current_player) +
           POINT_OPEN_TWO * num_of_huo_er(x, y, current_player) +
           POINT_SLEEP_TWO * num_of_mian_er(x, y, current_player);
}
int is_in_head(int x, int y, dian_and_fenshu point[SIZE][SIZE], int top_k)
/*
判断指定坐标 (x, y) 是否在分数最高的前 top_k 个候选点之中。

具体流程如下：

遍历整个 point[SIZE][SIZE] 数组，把分数不是 NEGATIVE_INFINITY 的点，按分数高低维护一个 top_k 大小的“高分点”数组 top。
如果还没满 top_k 个，就直接加入 top；如果已满，则只替换掉 top 里分数最低的那个（如果当前点分数更高）。
最后检查 (x, y) 是否在 top_k 个高分点里，如果在，返回 1，否则返回 0。*/
{
    if (top_k <= 0)
    {
        return 0;
    }

    dian_and_fenshu top[top_k];
    int count = 0;

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            // 跳过还没赋有效分数的位置
            if (point[i][j].fenshu == NEGATIVE_INFINITY)
            {
                continue;
            }

            if (count < top_k)
            {
                top[count++] = point[i][j];
            }
            else
            {
                int min_index = 0;
                long long int min_score = top[0].fenshu;
                for (int k = 1; k < top_k; k++)
                {
                    if (top[k].fenshu < min_score)
                    {
                        min_score = top[k].fenshu;
                        min_index = k;
                    }
                }
                if (point[i][j].fenshu > min_score)
                {
                    top[min_index] = point[i][j];
                }
            }
        }
    }

    // 判断 (x, y) 是否在 Top-K 之中
    for (int i = 0; i < count; i++)
    {
        if (top[i].dian.x == x && top[i].dian.y == y)
        {
            return 1;
        }
    }
    return 0;
}
// -----6.核心评分函数 6.2计算对手在当前局面下的最优一步及其分数（用于威胁判断/防守）
dian_and_fenshu give_point_for_the_opposite(int arrayForInnerBoardLayout[SIZE][SIZE], int x, int y, int current_player, int first_player)
{
    int opposite_player;
    long long int max_point = NEGATIVE_INFINITY;
    long long int temp_point;
    Point best_answer = {-1, -1};
    dian_and_fenshu best_move_with_point;
    if (current_player == BLACK)
    {
        opposite_player = WHITE;
    }
    else
    {
        opposite_player = BLACK;
    }
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            if ((return_what_is_at(i, j) == EMPTY) && is_neighbored_5_5(arrayForInnerBoardLayout, i, j))
            {
                arrayForInnerBoardLayout[i][j] = opposite_player;
                temp_point = give_point(arrayForInnerBoardLayout, i, j, opposite_player, first_player);
                if (temp_point > max_point)
                {
                    max_point = temp_point;
                    best_answer.x = i;
                    best_answer.y = j;
                }
                arrayForInnerBoardLayout[i][j] = EMPTY;
            }
        }
    }
    // 如果没有找到合法落点，则随机从所有空位中选一个，避免 (-1, -1) 越界
    if (best_answer.x == -1)
    {
        Point candidates[SIZE * SIZE];
        int cand_count = 0;
        for (int i = 0; i < SIZE; i++)
        {
            for (int j = 0; j < SIZE; j++)
            {
                if (return_what_is_at(i, j) == EMPTY)
                {
                    candidates[cand_count].x = i;
                    candidates[cand_count].y = j;
                    cand_count++;
                }
            }
        }
        if (cand_count > 0)
        {
            int idx = rand() % cand_count;
            best_answer = candidates[idx];
            arrayForInnerBoardLayout[best_answer.x][best_answer.y] = opposite_player;
            max_point = give_point(arrayForInnerBoardLayout, best_answer.x, best_answer.y, opposite_player, first_player);
        }
    }

    if (best_answer.x != -1)
    {
        arrayForInnerBoardLayout[best_answer.x][best_answer.y] = opposite_player;
    }
    best_move_with_point.dian = best_answer;
    best_move_with_point.fenshu = max_point;
    return best_move_with_point;
}

// 7.评估当前局势，选择攻击强弱（递归深度等）基于候选掩码，计算当前玩家一步能获得的最大分数（不排序，仅取最大值，常用于威胁检测/浅层估计）
static long long best_immediate_from_mask(
    int arrayForInnerBoardLayout[SIZE][SIZE],
    int player,
    int layer,
    int first_player,
    int who_to_count[SIZE][SIZE])
{
    // 这里只需要“一步最强”的分数，不需要 Top-K / 排序。
    // 用扫描取 max 可以避免每个节点都 qsort，能显著提速。
    long long best = NEGATIVE_INFINITY;
    int (*neighbor_fn)(int[SIZE][SIZE], int, int) = (layer <= 5) ? is_neighbored_5_5 : is_neighbored_3_3;

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            if (!who_to_count[i][j])
            {
                continue;
            }
            if (return_what_is_at(i, j) != EMPTY)
            {
                continue;
            }
            if (!neighbor_fn(arrayForInnerBoardLayout, i, j))
            {
                continue;
            }

            long long s = give_point(arrayForInnerBoardLayout, i, j, player, first_player);
            if (s == NEGATIVE_INFINITY)
            {
                continue;
            }
            if (s > best)
            {
                best = s;
            }
        }
    }

    if (best == NEGATIVE_INFINITY)
    {
        return 0;
    }
    return best;
}

// 7.1生成下一层候选掩码，基于父层掩码并扩展当前落子点周围区域
static void build_next_mask_from_base(
    int base[SIZE][SIZE],
    int out[SIZE][SIZE],
    int cx, int cy,
    int next_layer)
{
    // 复制父层掩码（通常比全 1 更稀疏）
    memcpy(out, base, sizeof(int) * SIZE * SIZE);

    // 额外把“当前落子点”附近加进去，避免掩码过窄漏掉关键应手
    int r = (next_layer <= 5) ? 2 : 1;
    for (int i = cx - r; i <= cx + r; i++)
    {
        for (int j = cy - r; j <= cy + r; j++)
        {
            if (in_bound(i, j))
            {
                out[i][j] = 1;
            }
        }
    }
}

// 7.2扫描当前局面下 player 的“最强一步静态得分”（用于威胁检测/浅层估计）
static long long max_immediate_threat(int arrayForInnerBoardLayout[SIZE][SIZE], int player, int first_player)
{
    long long best = NEGATIVE_INFINITY;
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            if (return_what_is_at(i, j) != EMPTY)
            {
                continue;
            }
            if (!is_neighbored_5_5(arrayForInnerBoardLayout, i, j))
            {
                continue;
            }
            long long s = give_point(arrayForInnerBoardLayout, i, j, player, first_player);
            if (s > best)
            {
                best = s;
            }
        }
    }

    // 没有候选点时返回 0，避免 NEGATIVE_INFINITY 参与计算
    if (best == NEGATIVE_INFINITY)
    {
        return 0;
    }
    return best;
}

// 7.3生成根节点候选掩码（只保留空位且有邻居的点，开局无邻居时放开中心点）
static void build_root_mask(int arrayForInnerBoardLayout[SIZE][SIZE], int out[SIZE][SIZE])
{
    int any = 0;
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            if (return_what_is_at(i, j) != EMPTY)
            {
                out[i][j] = 0;
                continue;
            }
            // 根节点用 5x5 邻域过滤即可，避免每层全盘 225 扫描
            out[i][j] = is_neighbored_5_5(arrayForInnerBoardLayout, i, j) ? 1 : 0;
            if (out[i][j])
            {
                any = 1;
            }
        }
    }

    // 开局没邻居时，至少放开中心点，避免无候选
    if (!any)
    {
        out[SIZE / 2][SIZE / 2] = 1;
    }
}

// 8.补丁函数：强行判断 (x, y) 是否为合法落子点（分数不为负无穷），再一次一个强力的防护
static int is_legal_by_give_point(int arrayForInnerBoardLayout[SIZE][SIZE], int x, int y, int player, int first_player)
{
    if (return_what_is_at(x, y) != EMPTY)
    {
        return 0;
    }
    long long s = give_point(arrayForInnerBoardLayout, x, y, player, first_player);
    return (s != NEGATIVE_INFINITY);
}

// 10.规则裁决：判断当前局面是否存在立即胜负或必须应手的点，若有则直接返回该点
static int find_rule_adjudication_move(
    int arrayForInnerBoardLayout[SIZE][SIZE],
    int current_player,
    int first_player,
    int root_mask[SIZE][SIZE],
    Point *out_best)
{
    int opp = opposite_player(current_player);

    // 1) 我方成五
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            if (!root_mask[i][j])
                continue;
            if (return_what_is_at(i, j) != EMPTY)
                continue;
            if (!is_final_legal_move(arrayForInnerBoardLayout, i, j, current_player, first_player))
                continue;
            if (check_5_if_place(arrayForInnerBoardLayout, i, j, current_player))
            {
                out_best->x = i;
                out_best->y = j;
                return 1;
            }
        }
    }

    // 2) 挡对手成五（若多点可成五，选“我下在那”的自评最高的那个）
    long long best_block_score = NEGATIVE_INFINITY;
    int found_block = 0;
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            if (!root_mask[i][j])
                continue;
            if (return_what_is_at(i, j) != EMPTY)
                continue;
            if (!is_final_legal_move(arrayForInnerBoardLayout, i, j, opp, first_player))
                continue;
            if (!check_5_if_place(arrayForInnerBoardLayout, i, j, opp))
                continue;
            if (!is_final_legal_move(arrayForInnerBoardLayout, i, j, current_player, first_player))
                continue;
            long long s = give_point(arrayForInnerBoardLayout, i, j, current_player, first_player);
            if (!found_block || s > best_block_score)
            {
                best_block_score = s;
                out_best->x = i;
                out_best->y = j;
                found_block = 1;
            }
        }
    }
    if (found_block)
    {
        return 1;
    }

    // 3) 活四/冲四/双活三（我方优先做；否则挡对手“下这里就形成”）
    // 我方：活四
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            if (!root_mask[i][j])
                continue;
            if (return_what_is_at(i, j) != EMPTY)
                continue;
            if (!is_legal_by_give_point(arrayForInnerBoardLayout, i, j, current_player, first_player))
                continue;
            if (num_of_huo_si(i, j, current_player) > 0)
            {
                out_best->x = i;
                out_best->y = j;
                return 1;
            }
        }
    }
    // 挡对手：活四
    long long best_block_threat = NEGATIVE_INFINITY;
    int found = 0;
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            if (!root_mask[i][j])
                continue;
            if (return_what_is_at(i, j) != EMPTY)
                continue;
            if (!is_legal_by_give_point(arrayForInnerBoardLayout, i, j, opp, first_player))
                continue;
            if (num_of_huo_si(i, j, opp) <= 0)
                continue;
            if (!is_legal_by_give_point(arrayForInnerBoardLayout, i, j, current_player, first_player))
                continue;

            long long k = rule_key_for_opp_threat(arrayForInnerBoardLayout, i, j, opp, first_player);
            if (!found || k > best_block_threat)
            {
                best_block_threat = k;
                out_best->x = i;
                out_best->y = j;
                found = 1;
            }
        }
    }
    if (found)
    {
        return 1;
    }

    // 我方：双活三（注意：黑方禁手会被 give_point 拦截）
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            if (!root_mask[i][j])
                continue;
            if (return_what_is_at(i, j) != EMPTY)
                continue;
            if (!is_legal_by_give_point(arrayForInnerBoardLayout, i, j, current_player, first_player))
                continue;
            if (num_of_huo_san(i, j, current_player) >= 2)
            {
                out_best->x = i;
                out_best->y = j;
                return 1;
            }
        }
    }

    // 挡对手：双活三
    best_block_threat = NEGATIVE_INFINITY;
    found = 0;
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            if (!root_mask[i][j])
                continue;
            if (return_what_is_at(i, j) != EMPTY)
                continue;
            if (!is_legal_by_give_point(arrayForInnerBoardLayout, i, j, opp, first_player))
                continue;
            if (num_of_huo_san(i, j, opp) < 2)
                continue;
            if (!is_legal_by_give_point(arrayForInnerBoardLayout, i, j, current_player, first_player))
                continue;
            long long k = rule_key_for_opp_threat(arrayForInnerBoardLayout, i, j, opp, first_player);
            if (!found || k > best_block_threat)
            {
                best_block_threat = k;
                out_best->x = i;
                out_best->y = j;
                found = 1;
            }
        }
    }
    if (found)
    {
        return 1;
    }

    return 0;
}

// ================================
// 11.威胁搜索（受限、无 Top-K）：证明某些点必须提前占住（防止有些点不占住导致的后果）
// 思路：枚举对手“强威胁点”P，假设对手下 P，再枚举我方少量防守点 D；
// 若所有 D 都挡不住对手下一手再形成成五/活四/冲四/双活三，则 P 是 must-block-now。
// ================================

static int collect_local_defenses(
    int arrayForInnerBoardLayout[SIZE][SIZE],
    dian_and_fenshu *out, int out_cap,
    int me, int first_player,
    int cx, int cy)
{
    int n = 0;
    int r = 2;
    for (int i = cx - r; i <= cx + r; i++)
    {
        for (int j = cy - r; j <= cy + r; j++)
        {
            if (!in_bound(i, j))
                continue;
            if (return_what_is_at(i, j) != EMPTY)
                continue;
            if (!is_legal_by_give_point(arrayForInnerBoardLayout, i, j, me, first_player))
                continue;
            long long s = give_point(arrayForInnerBoardLayout, i, j, me, first_player);
            if (n < out_cap)
            {
                out[n].dian.x = i;
                out[n].dian.y = j;
                out[n].fenshu = s;
                n++;
            }
        }
    }
    if (n > 1)
    {
        qsort(out, (size_t)n, sizeof(dian_and_fenshu), compare_dian_and_fenshu);
    }
    return n;
}

// 判断对手在 (cx,cy) 附近是否有直接杀棋（成五/活四/冲四/双活三）
static int opp_has_kill_followup_local(
    int arrayForInnerBoardLayout[SIZE][SIZE],
    int opp, int first_player,
    int cx, int cy)
{
    int r = 2;
    for (int i = cx - r; i <= cx + r; i++)
    {
        for (int j = cy - r; j <= cy + r; j++)
        {
            if (!in_bound(i, j))
                continue;
            if (return_what_is_at(i, j) != EMPTY)
                continue;
            if (!is_legal_by_give_point(arrayForInnerBoardLayout, i, j, opp, first_player))
                continue;
            if (check_5_if_place(arrayForInnerBoardLayout, i, j, opp))
                return 1;
            if (num_of_huo_si(i, j, opp) > 0)
                return 1;
            if (num_of_chong_si(i, j, opp) > 0)
                return 1;
            if (num_of_huo_san(i, j, opp) >= 2)
                return 1;
        }
    }
    return 0;
}

// 收集所有必须立即防守的点（must-block），用于威胁证明
static int collect_forced_block_points(
    int arrayForInnerBoardLayout[SIZE][SIZE],
    Point *out, int out_cap,
    int me, int first_player,
    int root_mask[SIZE][SIZE])
{
    int opp = opposite_player(me);
    // 仅枚举“强威胁点”，数量很少
    RuleMove threats[64];
    int tn = 0;

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            if (!root_mask[i][j])
                continue;
            if (return_what_is_at(i, j) != EMPTY)
                continue;
            if (!is_final_legal_move(arrayForInnerBoardLayout, i, j, opp, first_player))
                continue;

            // 只关心：成五/活四/双活三（移除对手冲四）
            if (!check_5(i, j, opp) && num_of_huo_si(i, j, opp) <= 0 && num_of_huo_san(i, j, opp) < 2)
            {
                continue;
            }

            long long k = rule_key_for_opp_threat(arrayForInnerBoardLayout, i, j, opp, first_player);
            if (tn < (int)(sizeof(threats) / sizeof(threats[0])))
            {
                threats[tn].p.x = i;
                threats[tn].p.y = j;
                threats[tn].key = k;
                tn++;
            }
        }
    }

    // 简单选择前若干强威胁点（n 很小，这里用 O(n^2) 选择即可，避免再写 compare）
    int take = tn;
    if (take > 8)
        take = 8;
    for (int a = 0; a < take; a++)
    {
        int best = a;
        for (int b = a + 1; b < tn; b++)
        {
            if (threats[b].key > threats[best].key)
                best = b;
        }
        if (best != a)
        {
            RuleMove tmp = threats[a];
            threats[a] = threats[best];
            threats[best] = tmp;
        }
    }

    int out_n = 0;
    for (int t = 0; t < take && out_n < out_cap; t++)
    {
        int px = threats[t].p.x;
        int py = threats[t].p.y;

        // 若我们现在占住 (px,py) 本身就是非法（禁手），那没法 must-block-now，只能交给后续常规搜索
        if (!is_legal_by_give_point(arrayForInnerBoardLayout, px, py, me, first_player))
        {
            continue;
        }

        // 假设对手下在 P
        arrayForInnerBoardLayout[px][py] = opp;

        // 对手若立即成五，则 P 必须现在堵
        if (check_5(px, py, opp))
        {
            arrayForInnerBoardLayout[px][py] = EMPTY;
            out[out_n].x = px;
            out[out_n].y = py;
            out_n++;
            continue;
        }

        // 我方防守枚举（局部少量点，不做 Top-K 全局裁剪）
        dian_and_fenshu defs[24];
        int dn = collect_local_defenses(arrayForInnerBoardLayout, defs, 24, me, first_player, px, py);

        int has_defense = 0;
        for (int di = 0; di < dn; di++)
        {
            int dx = defs[di].dian.x;
            int dy = defs[di].dian.y;
            arrayForInnerBoardLayout[dx][dy] = me;

            // 反击：如果我方此时能直接成五，则说明对手这步不是“强制杀”
            if (check_5(dx, dy, me))
            {
                has_defense = 1;
                arrayForInnerBoardLayout[dx][dy] = EMPTY;
                break;
            }

            // 只检查局部跟进杀（极快）；若没有局部杀，则认为此防守有效
            if (!opp_has_kill_followup_local(arrayForInnerBoardLayout, opp, first_player, px, py))
            {
                has_defense = 1;
                arrayForInnerBoardLayout[dx][dy] = EMPTY;
                break;
            }

            arrayForInnerBoardLayout[dx][dy] = EMPTY;
        }

        arrayForInnerBoardLayout[px][py] = EMPTY;

        if (!has_defense)
        {
            out[out_n].x = px;
            out[out_n].y = py;
            out_n++;
        }
    }

    return out_n;
}

// 12.生成候选点列表（邻域过滤 + who_to_count 掩码），并按静态分数降序排序
static int collect_candidates(
    int arrayForInnerBoardLayout[SIZE][SIZE],
    dian_and_fenshu *out, int out_cap,
    int player, int layer, int first_player,
    int who_to_count[SIZE][SIZE])
{
    int n = 0;
    int (*neighbor_fn)(int[SIZE][SIZE], int, int) = (layer <= 5) ? is_neighbored_5_5 : is_neighbored_3_3;

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            if (!who_to_count[i][j])
            {
                continue;
            }
            if (return_what_is_at(i, j) != EMPTY)
            {
                continue;
            }
            if (!neighbor_fn(arrayForInnerBoardLayout, i, j))
            {
                continue;
            }

            long long s = give_point(arrayForInnerBoardLayout, i, j, player, first_player);
            if (s == NEGATIVE_INFINITY)
            {
                continue;
            }

            if (n < out_cap)
            {
                out[n].dian.x = i;
                out[n].dian.y = j;
                out[n].fenshu = s;
                n++;
            }
        }
    }

    if (n > 1)
    {
        qsort(out, (size_t)n, sizeof(dian_and_fenshu), compare_dian_and_fenshu);
    }
    return n;
}

// 判断 arr 数组中是否包含 (x, y)
static int contains_xy(const dian_and_fenshu *arr, int n, int x, int y)
{
    for (int i = 0; i < n; i++)
    {
        if (arr[i].dian.x == x && arr[i].dian.y == y)
        {
            return 1;
        }
    }
    return 0;
}

// 收集所有对手威胁分高于阈值的点，作为必须优先防守的候选
static int collect_must_defense_candidates(
    int arrayForInnerBoardLayout[SIZE][SIZE],
    dian_and_fenshu *out, int out_cap,
    int opp_player, int layer, int first_player,
    int who_to_count[SIZE][SIZE])
{
    int n = 0;
    int (*neighbor_fn)(int[SIZE][SIZE], int, int) = (layer <= 5) ? is_neighbored_5_5 : is_neighbored_3_3;
    long long thr = POINT_OPEN_THREE(opp_player);

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            if (!who_to_count[i][j])
            {
                continue;
            }
            if (return_what_is_at(i, j) != EMPTY)
            {
                continue;
            }
            if (!neighbor_fn(arrayForInnerBoardLayout, i, j))
            {
                continue;
            }

            long long s_opp = give_point(arrayForInnerBoardLayout, i, j, opp_player, first_player);
            if (s_opp == NEGATIVE_INFINITY)
            {
                continue;
            }
            if (s_opp < thr)
            {
                continue;
            }

            if (n < out_cap)
            {
                out[n].dian.x = i;
                out[n].dian.y = j;
                out[n].fenshu = s_opp; // 这里暂存“对手威胁分”，用于排序挑选
                n++;
            }
        }
    }

    if (n > 1)
    {
        qsort(out, (size_t)n, sizeof(dian_and_fenshu), compare_dian_and_fenshu);
    }
    return n;
}

// 13.对称递归搜索主函数：双方都用 Top-K+阈值，遇到强威胁时加深搜索
static long long best_value_symmetric(
    int arrayForInnerBoardLayout[SIZE][SIZE],
    int player,
    int layer,
    int max_layer,
    int first_player,
    int who_to_count[SIZE][SIZE],
    uint64_t board_hash)
{
    ai_stats_visit(layer);
    if (layer > max_layer)
    {
        return 0;
    }

    // 13.1威胁触发：如果对手一手就有“活四级别”威胁，则允许额外加深
    long long opp_now = best_immediate_from_mask(arrayForInnerBoardLayout, opposite_player(player), layer, first_player, who_to_count);
    int local_max = max_layer;
    if (opp_now >= POINT_OPEN_FOUR && local_max < MAX_LAYER_CAP)
    {
        local_max = MAX_LAYER_CAP;
        ai_last_danger_triggered = 1;
    }
    if (local_max > ai_last_effective_max_layer)
    {
        ai_last_effective_max_layer = local_max;
    }
    if (layer > local_max)
    {
        return 0;
    }

    // 13.2置换表命中：直接返回缓存值（使用置换表缓存法）
    long long cached;
    uint64_t key = make_tt_key(board_hash, player, layer, local_max, first_player);
    if (tt_get(key, &cached))
    {
        return cached;
    }

    dian_and_fenshu cands[SIZE * SIZE];
    int n = collect_candidates(arrayForInnerBoardLayout, cands, SIZE * SIZE, player, layer, first_player, who_to_count);
    // 只保留最终合法点
    int legal_n = 0;
    for (int i = 0; i < n; ++i)
    {
        int x = cands[i].dian.x, y = cands[i].dian.y;
        if (is_final_legal_move(arrayForInnerBoardLayout, x, y, player, first_player))
        {
            cands[legal_n++] = cands[i];
        }
    }
    if (legal_n <= 0)
    {
        return 0;
    }
    n = legal_n;

    int k = amount(layer);
    if (k > n)
    {
        k = n;
    }

    // 13.3Top-K 只在“安全/价值线”上用；危险局面下额外保送防守点，避免静态排序阶段被淘汰。
    dian_and_fenshu evals[SIZE * SIZE];
    int eval_n = 0;

    for (int idx = 0; idx < k; idx++)
    {
        evals[eval_n++] = cands[idx];
    }

    int opp_player = opposite_player(player);
    if (opp_now >= POINT_OPEN_THREE(opp_player))
    {
        dian_and_fenshu musts[32];
        int must_n = collect_must_defense_candidates(arrayForInnerBoardLayout, musts, 32, opp_player, layer, first_player, who_to_count);
        for (int i = 0; i < must_n; i++)
        {
            int x = musts[i].dian.x;
            int y = musts[i].dian.y;
            if (contains_xy(evals, eval_n, x, y))
            {
                continue;
            }
            long long self_s = give_point(arrayForInnerBoardLayout, x, y, player, first_player);
            if (self_s == NEGATIVE_INFINITY)
            {
                continue;
            }
            evals[eval_n].dian.x = x;
            evals[eval_n].dian.y = y;
            evals[eval_n].fenshu = self_s;
            eval_n++;
        }
    }

    long long best = NEGATIVE_INFINITY;
    for (int idx = 0; idx < eval_n; idx++)
    {
        int x = evals[idx].dian.x;
        int y = evals[idx].dian.y;
        long long move_score = evals[idx].fenshu;

        arrayForInnerBoardLayout[x][y] = player;
        uint64_t h2 = board_hash ^ zobrist[x][y][player];

        // 下一层掩码：继承父层掩码，并把当前落子点周围纳入
        int who_next[SIZE][SIZE];
        build_next_mask_from_base(who_to_count, who_next, x, y, layer + 1);

        // 13.4浅层估计：对手一步最强（我走之后的对手威胁）
        long long opp_best_1 = best_immediate_from_mask(arrayForInnerBoardLayout, opposite_player(player), layer + 1, first_player, who_next);
        long long shallow = move_score - (long long)(opp_best_1 * OPP_FACTOR(first_player));

        // 13.5防守奖励：危险局面优先降低对手威胁
        long long bonus = 0;
        if (opp_now >= DEFENSE_TRIGGER_THREAT)
        {
            long long gain = opp_now - opp_best_1;
            if (gain > 0)
            {
                bonus = (gain * DEFENSE_BONUS_NUM) / DEFENSE_BONUS_DEN;
                bonus = clamp_ll(bonus, 0, MUST_BLOCK_PENALTY);
            }

            // 对手已是活四级别威胁，但你走完仍然活四级 => 强惩罚逼你去堵
            if (opp_now >= POINT_OPEN_FOUR && opp_best_1 >= POINT_OPEN_FOUR)
            {
                shallow -= MUST_BLOCK_PENALTY;
            }

            // 13.6对手已是“双三”级别威胁，但你走完仍然“双三” => 强惩罚逼你优先去堵
            // 如果你这步本身就制造了活四（强进攻），允许“以攻代守”不强制堵。
            if (move_score < POINT_OPEN_FOUR && opp_now >= MUST_BLOCK_DOUBLE_THREE_THREAT && opp_best_1 >= MUST_BLOCK_DOUBLE_THREE_THREAT)
            {
                shallow -= MUST_BLOCK_DOUBLE_THREE_PENALTY;
            }
        }

        // 13.7 bonus 参与剪枝：避免“防守步能降低威胁但 shallow 偏低被剪掉”
        long long prune_score = shallow + bonus;

        long long total;
        if (layer >= local_max || prune_score < ThresholdDyn(layer, opp_best_1, first_player))
        {
            total = prune_score;
        }
        else
        {
            long long opp_deep = best_value_symmetric(arrayForInnerBoardLayout, opposite_player(player), layer + 1, max_layer, first_player, who_next, h2);
            total = move_score - (long long)(opp_deep * OPP_FACTOR(first_player)) + bonus;
        }

        arrayForInnerBoardLayout[x][y] = EMPTY;

        if (total > best)
        {
            best = total;
        }
    }

    if (best == NEGATIVE_INFINITY)
    {
        tt_put(key, 0);
        return 0;
    }
    tt_put(key, best);
    return best;
}

// 14.计算所有分数的函数（每个邻域调用一遍）
void count_all_points(int arrayForInnerBoardLayout[SIZE][SIZE], dian_and_fenshu point[SIZE][SIZE], int current_player, int layer, int first_player, int who_to_count[SIZE][SIZE])
{

    if (layer <= 5)
    {
        for (int i = 0; i < SIZE; i++)
        {
            for (int j = 0; j < SIZE; j++)
            {
                if (return_what_is_at(i, j) == EMPTY && is_neighbored_5_5(arrayForInnerBoardLayout, i, j) && who_to_count[i][j])
                {
                    point[i][j].fenshu = 0;
                    return_recursive_points_at_position(arrayForInnerBoardLayout, i, j, point, current_player, layer, first_player);
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < SIZE; i++)
        {
            for (int j = 0; j < SIZE; j++)
            {
                if (return_what_is_at(i, j) == EMPTY && is_neighbored_3_3(arrayForInnerBoardLayout, i, j) && who_to_count[i][j])
                {
                    point[i][j].fenshu = 0;
                    return_recursive_points_at_position(arrayForInnerBoardLayout, i, j, point, current_player, layer, first_player);
                }
            }
        }
    }
}

// 15.每个地方判分的最终的函数
void return_recursive_points_at_position(int arrayForInnerBoardLayout[SIZE][SIZE], int x, int y, dian_and_fenshu point[SIZE][SIZE], int current_player, int layer, int first_player)
{
    ai_stats_visit(layer);
    if (return_what_is_at(x, y) != EMPTY)
    {
        return;
    }

    point[x][y].dian.x = x;
    point[x][y].dian.y = y;

    long long move_score = give_point(arrayForInnerBoardLayout, x, y, current_player, first_player);
    if (move_score == NEGATIVE_INFINITY)
    {
        point[x][y].fenshu = NEGATIVE_INFINITY;
        return;
    }

    // root 层掩码：根节点默认全 1（不限制），并用于对手“一步最强/递归”
    int who_all[SIZE][SIZE];
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            who_all[i][j] = 1;
        }
    }

    // 如果当前局面里对手威胁很高（活四级别），则允许整棵树加深
    int max_layer = MAX_LAYER;
    long long opp_now = best_immediate_from_mask(arrayForInnerBoardLayout, opposite_player(current_player), layer, first_player, who_all);
    ai_last_root_opp_threat = opp_now;
    if (opp_now >= POINT_OPEN_FOUR)
    {
        max_layer = MAX_LAYER_CAP;
        ai_last_danger_triggered = 1;
    }
    if (max_layer > ai_last_effective_max_layer)
    {
        ai_last_effective_max_layer = max_layer;
    }

    // 先落子
    arrayForInnerBoardLayout[x][y] = current_player;
    uint64_t root_hash = ai_root_board_hash ^ zobrist[x][y][current_player];

    // 浅层估计：对手一步最强（低于阈值就不深入，但保留该估计分数）
    long long opp_best_1 = best_immediate_from_mask(arrayForInnerBoardLayout, opposite_player(current_player), layer + 1, first_player, who_all);
    long long shallow = move_score - (long long)(opp_best_1 * OPP_FACTOR(first_player));

    // 根节点同样加防守奖励/必须堵惩罚
    long long bonus = 0;
    if (opp_now >= DEFENSE_TRIGGER_THREAT)
    {
        long long gain = opp_now - opp_best_1;
        if (gain > 0)
        {
            bonus = (gain * DEFENSE_BONUS_NUM) / DEFENSE_BONUS_DEN;
            bonus = clamp_ll(bonus, 0, MUST_BLOCK_PENALTY);
        }
        if (opp_now >= POINT_OPEN_FOUR && opp_best_1 >= POINT_OPEN_FOUR)
        {
            shallow -= MUST_BLOCK_PENALTY;
        }

        if (move_score < POINT_OPEN_FOUR && opp_now >= MUST_BLOCK_DOUBLE_THREE_THREAT && opp_best_1 >= MUST_BLOCK_DOUBLE_THREE_THREAT)
        {
            shallow -= MUST_BLOCK_DOUBLE_THREE_PENALTY;
        }
    }

    // bonus 参与剪枝
    long long prune_score = shallow + bonus;

    if (layer >= max_layer || prune_score < ThresholdDyn(layer, opp_best_1, first_player))
    {
        point[x][y].fenshu = prune_score;
        arrayForInnerBoardLayout[x][y] = EMPTY;
        return;
    }

    long long opp_deep = best_value_symmetric(arrayForInnerBoardLayout, opposite_player(current_player), layer + 1, max_layer, first_player, who_all, root_hash);
    point[x][y].fenshu = move_score - (long long)(opp_deep * OPP_FACTOR(first_player)) + bonus;

    // 回滚
    arrayForInnerBoardLayout[x][y] = EMPTY;
}

// 16.主接口：给定当前局面，返回最佳落子点及其分数（AI主入口）
dian_and_fenshu give_an_answer(int arrayForInnerBoardLayout[SIZE][SIZE], int current_player, int first_player)
{ // 计算所有空位的分数，选择分数最高的作为落子点；所有空位的分数是用递归方式算的
    ai_stats_reset();

    // 当前局面只算一次 hash；后续根节点“试落子”用 XOR 增量更新
    ai_root_board_hash = compute_board_hash(arrayForInnerBoardLayout);

    int layer = 1;
    // 默认给一个安全值，避免极端情况下未初始化
    Point best_move = {SIZE / 2, SIZE / 2};
    long long best_score = NEGATIVE_INFINITY;

    // 根节点候选掩码：大幅减少全盘扫描
    int root_mask[SIZE][SIZE];
    build_root_mask(arrayForInnerBoardLayout, root_mask);

    // 根层威胁（用于侧边栏显示 + danger 加深判定）
    long long opp_now = best_immediate_from_mask(arrayForInnerBoardLayout, opposite_player(current_player), layer, first_player, root_mask);
    ai_last_root_opp_threat = opp_now;
    int max_layer = MAX_LAYER;
    if (opp_now >= POINT_OPEN_FOUR)
    {
        max_layer = MAX_LAYER_CAP;
        ai_last_danger_triggered = 1;
    }
    if (max_layer > ai_last_effective_max_layer)
    {
        ai_last_effective_max_layer = max_layer;
    }

    // 1) 规则裁决（立即胜负/必须应手）：极快，命中就直接返回
    Point rule_move;
    if (find_rule_adjudication_move(arrayForInnerBoardLayout, current_player, first_player, root_mask, &rule_move))
    {
        dian_and_fenshu ans;
        ans.dian = rule_move;
        ans.fenshu = POSITIVE_INFINITY / 2;
        return ans;
    }

    // 2) 威胁强制搜索：无 Top-K，只对“强威胁点”做 1.5-ply 证明
    Point forced_blocks[16];
    int forced_n = collect_forced_block_points(arrayForInnerBoardLayout, forced_blocks, 16, current_player, first_player, root_mask);
    if (forced_n > 0)
    {
        // 在 must-block 集合里选一个：优先把对手下一手最强威胁降得最低
        long long best_def = POSITIVE_INFINITY;
        for (int i = 0; i < forced_n; i++)
        {
            int x = forced_blocks[i].x;
            int y = forced_blocks[i].y;
            if (!is_legal_by_give_point(arrayForInnerBoardLayout, x, y, current_player, first_player))
                continue;

            arrayForInnerBoardLayout[x][y] = current_player;
            long long opp_best = max_immediate_threat(arrayForInnerBoardLayout, opposite_player(current_player), first_player);
            arrayForInnerBoardLayout[x][y] = EMPTY;

            if (opp_best < best_def)
            {
                best_def = opp_best;
                best_move.x = x;
                best_move.y = y;
            }
        }

        dian_and_fenshu ans;
        ans.dian = best_move;
        ans.fenshu = (best_def == POSITIVE_INFINITY) ? NEGATIVE_INFINITY / 2 : -best_def;
        return ans;
    }

    // 3) 常规启发式搜索：Top-K + 权重 + 对称递归
    // 只评估少量候选点，避免旧逻辑“全盘 225 点递归算分”
    dian_and_fenshu cands[SIZE * SIZE];
    int n = collect_candidates(arrayForInnerBoardLayout, cands, SIZE * SIZE, current_player, layer, first_player, root_mask);
    // 只保留最终合法点
    int legal_n = 0;
    for (int i = 0; i < n; ++i)
    {
        int x = cands[i].dian.x, y = cands[i].dian.y;
        if (is_final_legal_move(arrayForInnerBoardLayout, x, y, current_player, first_player))
        {
            cands[legal_n++] = cands[i];
        }
    }
    if (legal_n <= 0)
    {
        // 明确返回“无合法走法 → 规则失败”
        dian_and_fenshu ans;
        ans.dian.x = -1;
        ans.dian.y = -1;
        ans.fenshu = NEGATIVE_INFINITY;
        return ans;
    }
    n = legal_n;

    int k = amount(layer);
    if (k > n)
        k = n;

    for (int idx = 0; idx < k; idx++)
    {
        int x = cands[idx].dian.x;
        int y = cands[idx].dian.y;
        if (!is_legal_by_give_point(arrayForInnerBoardLayout, x, y, current_player, first_player))
        {
            continue;
        }

        long long move_score = cands[idx].fenshu;
        arrayForInnerBoardLayout[x][y] = current_player;
        uint64_t root_hash = ai_root_board_hash ^ zobrist[x][y][current_player];

        // 下一层掩码：继承根 mask，并把当前落子点附近补进来
        int who_next[SIZE][SIZE];
        build_next_mask_from_base(root_mask, who_next, x, y, layer + 1);

        // 浅层估计
        long long opp_best_1 = best_immediate_from_mask(arrayForInnerBoardLayout, opposite_player(current_player), layer + 1, first_player, who_next);
        long long shallow = move_score - (long long)(opp_best_1 * OPP_FACTOR(first_player));

        // 根节点同样加防守奖励/必须堵惩罚
        long long bonus = 0;
        if (opp_now >= DEFENSE_TRIGGER_THREAT)
        {
            long long gain = opp_now - opp_best_1;
            if (gain > 0)
            {
                bonus = (gain * DEFENSE_BONUS_NUM) / DEFENSE_BONUS_DEN;
                bonus = clamp_ll(bonus, 0, MUST_BLOCK_PENALTY);
            }
            if (opp_now >= POINT_OPEN_FOUR && opp_best_1 >= POINT_OPEN_FOUR)
            {
                shallow -= MUST_BLOCK_PENALTY;
            }

            if (move_score < POINT_OPEN_FOUR && opp_now >= MUST_BLOCK_DOUBLE_THREE_THREAT && opp_best_1 >= MUST_BLOCK_DOUBLE_THREE_THREAT)
            {
                shallow -= MUST_BLOCK_DOUBLE_THREE_PENALTY;
            }
        }

        long long prune_score = shallow + bonus;
        long long total;
        if (layer >= max_layer || prune_score < ThresholdDyn(layer, opp_best_1, first_player))
        {
            total = prune_score;
        }
        else
        {
            long long opp_deep = best_value_symmetric(arrayForInnerBoardLayout, opposite_player(current_player), layer + 1, max_layer, first_player, who_next, root_hash);
            total = move_score - (long long)(opp_deep * OPP_FACTOR(first_player)) + bonus;
        }

        arrayForInnerBoardLayout[x][y] = EMPTY;

        if (best_score == NEGATIVE_INFINITY || total > best_score)
        {
            best_score = total;
            best_move.x = x;
            best_move.y = y;
        }
    }

    if (best_score == NEGATIVE_INFINITY)
    {
        // 兜底：找任意空位
        for (int i = 0; i < SIZE; i++)
        {
            for (int j = 0; j < SIZE; j++)
            {
                if (return_what_is_at(i, j) == EMPTY)
                {
                    best_move.x = i;
                    best_move.y = j;
                    best_score = 0;
                    i = SIZE;
                    break;
                }
            }
        }
    }

    dian_and_fenshu answer;
    answer.dian = best_move;
    answer.fenshu = best_score;
    return answer;
}