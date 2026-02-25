#include "shared_variable_function.h"
// 基本的一些工具函数实现
int in_bound(int x, int y)
// 1.边界检查函数
{
    if (x >= 0 && x < SIZE && y >= 0 && y < SIZE)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}
int return_what_is_at(int x, int y)
// 2.查看内容函数
{
    return arrayForInnerBoardLayout[x][y];
}
// 3.用户转换输入和输出函数
//  用户输入 (如 H 8) 转内部坐标 (如 7,7)
//  行自下到上为1-15，列自左到右为A-O
void user_to_internal(const char *col_str, int row_input, int *x, int *y)
{
    if (col_str && col_str[0] != '\0')
    {
        char col = toupper(col_str[0]);
        *y = col - 'A';
    }
    else
    {
        *y = -1;
    }
    // x = SIZE - row_input
    *x = SIZE - row_input;
}

// 内部坐标 (如 7,7) 转用户输出 (如 H 8)
void internal_to_user(int x, int y, char *col_str, int *row_output)
{
    if (col_str)
    {
        col_str[0] = 'A' + y;
        col_str[1] = '\0';
    }
    if (row_output)
    {
        // row_output = SIZE - x
        *row_output = SIZE - x;
    }
}
