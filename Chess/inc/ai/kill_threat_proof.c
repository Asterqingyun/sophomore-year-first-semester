// kill_threat_proof.c
// 局部杀棋证明模块实现
#include "shared_variable_function.h"
#include "kill_threat_proof.h"

// =====================
// 阶段 1.1: 收集攻击方强威胁点
// =====================
int collect_attacker_threats(int board[SIZE][SIZE], int attacker, Move threats[], int max_n)
{
    int n = 0;
    for (int x = 0; x < SIZE; ++x)
    {
        for (int y = 0; y < SIZE; ++y)
        {
            if (board[x][y] != EMPTY)
                continue;
            // 成五
            if (check_5_if_place(board, x, y, attacker))
            {
                threats[n++] = (Move){x, y};
                if (n >= max_n)
                    return n;
                continue;
            }
            // 活四/冲四
            if (num_of_huo_si(x, y, attacker) > 0 || num_of_chong_si(x, y, attacker) > 0)
            {
                threats[n++] = (Move){x, y};
                if (n >= max_n)
                    return n;
                continue;
            }
            // 双活三
            if (num_of_huo_san(x, y, attacker) >= 2)
            {
                threats[n++] = (Move){x, y};
                if (n >= max_n)
                    return n;
            }
        }
    }
    return n;
}

// =====================
// 阶段 1.2: 收集防守方唯一防点
// =====================
int collect_defender_blocks(int board[SIZE][SIZE], int attacker, Move blocks[], int max_n)
{
    int n = 0;
    // 只收集能消灭当前威胁的点（即所有强威胁点）
    n = collect_attacker_threats(board, attacker, blocks, max_n);
    return n;
}

// =====================
// 阶段 2.1: 判断是否有直接成五
// =====================
int exists_immediate_five(int board[SIZE][SIZE], int player)
{
    for (int x = 0; x < SIZE; ++x)
    {
        for (int y = 0; y < SIZE; ++y)
        {
            if (board[x][y] != EMPTY)
                continue;
            if (check_5_if_place(board, x, y, player))
                return 1;
        }
    }
    return 0;
}

// =====================
// 阶段 2.2: 威胁序列递归搜索
// =====================
int threat_kill_search(int board[SIZE][SIZE], int attacker, int depth)
{
    if (depth <= 0)
        return 0;
    if (exists_immediate_five(board, attacker))
        return 1;
    Move threats[MAX_THREATS];
    int n_threats = collect_attacker_threats(board, attacker, threats, MAX_THREATS);
    for (int i = 0; i < n_threats; ++i)
    {
        Move m = threats[i];
        board[m.x][m.y] = attacker;
        Move blocks[MAX_BLOCKS];
        int n_blocks = collect_defender_blocks(board, attacker, blocks, MAX_BLOCKS);
        if (n_blocks == 0)
        {
            board[m.x][m.y] = EMPTY;
            return 1;
        }
        int all_fail = 1;
        for (int j = 0; j < n_blocks; ++j)
        {
            Move d = blocks[j];
            board[d.x][d.y] = (attacker == BLACK ? WHITE : BLACK);
            if (!threat_kill_search(board, attacker, depth - 1))
            {
                all_fail = 0;
                board[d.x][d.y] = EMPTY;
                break;
            }
            board[d.x][d.y] = EMPTY;
        }
        board[m.x][m.y] = EMPTY;
        if (all_fail)
            return 1;
    }
    return 0;
}
