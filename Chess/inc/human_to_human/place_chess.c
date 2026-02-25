#include "shared_variable_function.h"

int winner;
void placeChess(int x, int y, int currentplayer) // 1.放置棋子用的
{
    if (currentplayer == BLACK)
    {
        arrayForInnerBoardLayout[x][y] = BLACK;
    }
    else if (currentplayer == WHITE)
    {
        arrayForInnerBoardLayout[x][y] = WHITE;
    }
}

int check_5(int x, int y, int current_player)
//2.看看有无连成5的
{
    // 使用轴向总长度，保持原逻辑：任一方向 total_length>=4 即成五
    if (get_horizontal_axis_info(x, y, current_player).total_length >= 4)
        return 1;
    if (get_vertical_axis_info(x, y, current_player).total_length >= 4)
        return 1;
    if (get_diag1_axis_info(x, y, current_player).total_length >= 4)
        return 1;
    if (get_diag2_axis_info(x, y, current_player).total_length >= 4)
        return 1;
    return 0; // 没有达成5
}
