
#ifndef GAME_H
#define GAME_H

// 1. 基本常量与头文件
#define SIZE 15
#define CHARSIZE 1
#define DISPLAY_COLS (2 + 1 + SIZE * 2 + 1) // 行号2+空格1+15*2+\0
#define EMPTY 0
#define BLACK 1
#define WHITE 2
#define REGRET_CODE -1

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>

// 2. 结构体定义
typedef struct
{
    int x;
    int y;
} Point;
typedef struct
{
    int length;
    int is_open;
    Point gap_coord;
} SegmentInfo;
typedef struct
{
    int total_length;
    SegmentInfo seg1;
    SegmentInfo seg2;
} AxisInfo;
typedef struct
{
    Point dian;
    long long int fenshu;
} dian_and_fenshu;

extern char arrayForDisplayBoard[SIZE][DISPLAY_COLS];
extern int arrayForInnerBoardLayout[SIZE][SIZE];
extern char play1Pic[];
extern char play1CurrentPic[];
extern char play2Pic[];
extern char play2CurrentPic[];
extern int winner;
extern int step;
extern int top;

// 3 历史记录栈相关声明
void push(Point p);
Point pop(void);

// 4. 棋盘与显示相关函数声明
extern void initRecordBorard(void);
extern void innerLayoutToDisplayArray(void);
extern void displayBoard(void);

// 5. 棋子操作与胜负判定
void placeChess(int x, int y, int currentplayer);
int check_5(int x, int y, int current_player);
int in_bound(int x, int y);
int return_what_is_at(int x, int y);

// 6. 方向与轴相关函数
SegmentInfo get_segment_left(int x, int y, int color);
SegmentInfo get_segment_right(int x, int y, int color);
SegmentInfo get_segment_up(int x, int y, int color);
SegmentInfo get_segment_down(int x, int y, int color);
SegmentInfo get_segment_leftup(int x, int y, int color);
SegmentInfo get_segment_rightdown(int x, int y, int color);
SegmentInfo get_segment_rightup(int x, int y, int color);
SegmentInfo get_segment_leftdown(int x, int y, int color);
AxisInfo get_horizontal_axis_info(int x, int y, int color);
AxisInfo get_vertical_axis_info(int x, int y, int color);
AxisInfo get_diag1_axis_info(int x, int y, int color);
AxisInfo get_diag2_axis_info(int x, int y, int color);
int return_line_self(int x, int y, int color);
int return_column_self(int x, int y, int color);
int return_leftup_rightdown_self(int x, int y, int color);
int return_rightup_leftdown_self(int x, int y, int color);

// 7. 规则判定相关
int is_long_connected(int x, int y, int current_player);
int check_five_in_a_row(int x, int y, int current_player);
int num_of_huo_san(int x, int y, int current_player);
int if_put_is_huo_si(int x, int y, int current_player);
int num_of_huo_si(int x, int y, int current_player);
int num_of_chong_si(int x, int y, int current_player);
int num_of_mian_san(int x, int y, int current_player);
int num_of_huo_er(int x, int y, int current_player);
int num_of_mian_er(int x, int y, int current_player);
int is_double_huo_san(int x, int y, int current_player);
int double_si(int x, int y, int current_player);
int is_banned_after(int x, int y, int current_player);
int is_banned_by_long_connected(int x, int y, int current_player);

// 8. AI相关接口与统计（给分or调试使用)
long long int give_point(int arrayForInnerBoardLayout[SIZE][SIZE], int x, int y, int current_player, int first_player);
dian_and_fenshu give_point_for_the_opposite(int arrayForInnerBoardLayout[SIZE][SIZE], int x, int y, int current_player, int first_player);
#define AI_STATS_LAYER_CAP 32
extern int ai_last_max_depth_reached;
extern long long ai_last_nodes_visited;
extern long long ai_last_nodes_by_layer[AI_STATS_LAYER_CAP];
extern int ai_last_danger_triggered;
extern long long ai_last_root_opp_threat;
extern int ai_last_effective_max_layer;
extern long long ai_last_cache_lookups;
extern long long ai_last_cache_hits;
void count_all_points(int arrayForInnerBoardLayout[SIZE][SIZE], dian_and_fenshu point[SIZE][SIZE], int current_player, int layer, int first_player, int who_to_count[SIZE][SIZE]);
void return_recursive_points_at_position(int arrayForInnerBoardLayout[SIZE][SIZE], int x, int y, dian_and_fenshu point[SIZE][SIZE], int current_player, int layer, int first_player);
dian_and_fenshu give_an_answer(int arrayForInnerBoardLayout[SIZE][SIZE], int current_player, int first_player);
/*
#define POINT_FIVE_IN_A_ROW 10000000
#define POINT_OPEN_FOUR 1000000
#define POINT_SLEEP_FOUR 100000
#define POINT_OPEN_THREE 10000
#define POINT_SLEEP_THREE 1000
#define POINT_OPEN_TWO 100
#define POINT_SLEEP_TWO 10
*/
double pow(double x, double y);
#define NEGATIVE_INFINITY -100000000000
#define POSITIVE_INFINITY 100000000000
int check_5_if_place(int board[SIZE][SIZE], int x, int y, int player);
static inline double OPP_FACTOR(int first_player) { return (first_player == BLACK) ? 1.2 : 1.5; }

// 9. 用户输入/输出转换相关
void user_to_internal(const char *col_str, int row_input, int *x, int *y);
void internal_to_user(int x, int y, char *col_str, int *row_output);

// 10.模式定义
#define MODE_HUMAN_VS_HUMAN 1
#define MODE_HUMAN_VS_AI 2
#endif // GAME_H
