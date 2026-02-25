// kill_threat_proof.h
// 局部杀棋证明模块接口
// threat_kill_search:
// - 不算分
// - 不做 alpha-beta
// - 不用 Top-K
// - 不看对称
// - 只看“强制威胁”
// 这是一个规则级战术工具，不是主搜索的一部分。

#ifndef KILL_THREAT_PROOF_H
#define KILL_THREAT_PROOF_H

#include "shared_variable_function.h"

#define MAX_THREATS 8
#define MAX_BLOCKS 8

typedef struct
{
    int x, y;
} Move;

int collect_attacker_threats(int board[SIZE][SIZE], int attacker, Move threats[], int max_n);
int collect_defender_blocks(int board[SIZE][SIZE], int attacker, Move blocks[], int max_n);
int exists_immediate_five(int board[SIZE][SIZE], int player);
int threat_kill_search(int board[SIZE][SIZE], int attacker, int depth);

#endif
