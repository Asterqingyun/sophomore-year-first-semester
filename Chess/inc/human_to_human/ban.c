#include "shared_variable_function.h"
// 判断禁手的文件(借助底层直接实现)
int is_double_huo_san(int x, int y, int current_player)
// 判断在(x,y)落子后是否能形成双活三
{
    if (num_of_huo_san(x, y, current_player) >= 2)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}
int double_si(int x, int y, int current_player)
// 判断在(x,y)落子后是否能形成双四
{
    if (num_of_huo_si(x, y, current_player) + num_of_chong_si(x, y, current_player) >= 2)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}
int is_banned_by_long_connected(int x, int y, int current_player)
// 判断在(x,y)落子后是否能形成长连(6连及以上)
{
    if (is_long_connected(x, y, current_player) >= 1)
    {
        return 1;
    }
    return 0;
}
int is_banned_after(int x, int y, int current_player)
// 判断在(x,y)落子后是否触发33/44禁手
{

    if (double_si(x, y, current_player) >= 1)
    {
        return 1;
    }
    if (is_double_huo_san(x, y, current_player) >= 1)
    {
        return 1;
    }
    return 0;
}