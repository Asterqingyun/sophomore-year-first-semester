// 判断周围自己人和别人
#include "shared_variable_function.h"

// 1.单方向：从 (x,y) 开始，沿指定方向遍历连续同色棋子，并记录尽头的空位
// 分别实现八个方向的函数
SegmentInfo get_segment_left(int x, int y, int color)
{
    SegmentInfo info;
    info.length = 0;
    info.is_open = 0;
    info.gap_coord.x = -1;
    info.gap_coord.y = -1;

    int cy = y;
    while (in_bound(x, cy - 1) && arrayForInnerBoardLayout[x][cy - 1] == color)
    {
        info.length++; // 如果左边空并且是同色，那么继续向左扫描
        cy--;
    }

    if (in_bound(x, cy - 1) && arrayForInnerBoardLayout[x][cy - 1] == EMPTY)
    {
        info.is_open = 1;
        info.gap_coord.x = x; // 如果左边是空位，记录该空位坐标
        info.gap_coord.y = cy - 1;
    }

    return info;
}
// 其余同理
SegmentInfo get_segment_right(int x, int y, int color)
{
    SegmentInfo info;
    info.length = 0;
    info.is_open = 0;
    info.gap_coord.x = -1;
    info.gap_coord.y = -1;

    int cy = y;
    while (in_bound(x, cy + 1) && arrayForInnerBoardLayout[x][cy + 1] == color)
    {
        info.length++; // 如果右边空并且是同色，那么继续向右扫描
        cy++;
    }

    if (in_bound(x, cy + 1) && arrayForInnerBoardLayout[x][cy + 1] == EMPTY)
    {
        info.is_open = 1;
        info.gap_coord.x = x;
        info.gap_coord.y = cy + 1;
    }

    return info;
}

SegmentInfo get_segment_up(int x, int y, int color)
{
    SegmentInfo info;
    info.length = 0;
    info.is_open = 0;
    info.gap_coord.x = -1;
    info.gap_coord.y = -1;

    int cx = x;
    while (in_bound(cx - 1, y) && arrayForInnerBoardLayout[cx - 1][y] == color)
    {
        info.length++;
        cx--;
    }

    if (in_bound(cx - 1, y) && arrayForInnerBoardLayout[cx - 1][y] == EMPTY)
    {
        info.is_open = 1;
        info.gap_coord.x = cx - 1;
        info.gap_coord.y = y;
    }

    return info;
}

SegmentInfo get_segment_down(int x, int y, int color)
{
    SegmentInfo info;
    info.length = 0;
    info.is_open = 0;
    info.gap_coord.x = -1;
    info.gap_coord.y = -1;

    int cx = x;
    while (in_bound(cx + 1, y) && arrayForInnerBoardLayout[cx + 1][y] == color)
    {
        info.length++;
        cx++;
    }

    if (in_bound(cx + 1, y) && arrayForInnerBoardLayout[cx + 1][y] == EMPTY)
    {
        info.is_open = 1;
        info.gap_coord.x = cx + 1;
        info.gap_coord.y = y;
    }

    return info;
}

SegmentInfo get_segment_leftup(int x, int y, int color)
{
    SegmentInfo info;
    info.length = 0;
    info.is_open = 0;
    info.gap_coord.x = -1;
    info.gap_coord.y = -1;

    int cx = x;
    int cy = y;
    while (in_bound(cx - 1, cy - 1) && arrayForInnerBoardLayout[cx - 1][cy - 1] == color)
    {
        info.length++;
        cx--;
        cy--;
    }

    if (in_bound(cx - 1, cy - 1) && arrayForInnerBoardLayout[cx - 1][cy - 1] == EMPTY)
    {
        info.is_open = 1;
        info.gap_coord.x = cx - 1;
        info.gap_coord.y = cy - 1;
    }

    return info;
}

SegmentInfo get_segment_rightdown(int x, int y, int color)
{
    SegmentInfo info;
    info.length = 0;
    info.is_open = 0;
    info.gap_coord.x = -1;
    info.gap_coord.y = -1;

    int cx = x;
    int cy = y;
    while (in_bound(cx + 1, cy + 1) && arrayForInnerBoardLayout[cx + 1][cy + 1] == color)
    {
        info.length++;
        cx++;
        cy++;
    }

    if (in_bound(cx + 1, cy + 1) && arrayForInnerBoardLayout[cx + 1][cy + 1] == EMPTY)
    {
        info.is_open = 1;
        info.gap_coord.x = cx + 1;
        info.gap_coord.y = cy + 1;
    }

    return info;
}

SegmentInfo get_segment_rightup(int x, int y, int color)
{
    SegmentInfo info;
    info.length = 0;
    info.is_open = 0;
    info.gap_coord.x = -1;
    info.gap_coord.y = -1;

    int cx = x;
    int cy = y;
    while (in_bound(cx - 1, cy + 1) && arrayForInnerBoardLayout[cx - 1][cy + 1] == color)
    {
        info.length++;
        cx--;
        cy++;
    }

    if (in_bound(cx - 1, cy + 1) && arrayForInnerBoardLayout[cx - 1][cy + 1] == EMPTY)
    {
        info.is_open = 1;
        info.gap_coord.x = cx - 1;
        info.gap_coord.y = cy + 1;
    }

    return info;
}

SegmentInfo get_segment_leftdown(int x, int y, int color)
{
    SegmentInfo info;
    info.length = 0;
    info.is_open = 0;
    info.gap_coord.x = -1;
    info.gap_coord.y = -1;

    int cx = x;
    int cy = y;
    while (in_bound(cx + 1, cy - 1) && arrayForInnerBoardLayout[cx + 1][cy - 1] == color)
    {
        info.length++;
        cx++;
        cy--;
    }

    if (in_bound(cx + 1, cy - 1) && arrayForInnerBoardLayout[cx + 1][cy - 1] == EMPTY)
    {
        info.is_open = 1;
        info.gap_coord.x = cx + 1;
        info.gap_coord.y = cy - 1;
    }

    return info;
}

// 2.轴向信息，综合左右得到水平，总共四个方向

AxisInfo get_horizontal_axis_info(int x, int y, int color)
{
    AxisInfo axis;
    axis.seg1 = get_segment_left(x, y, color);
    axis.seg2 = get_segment_right(x, y, color);
    axis.total_length = axis.seg1.length + axis.seg2.length;
    return axis;
}

AxisInfo get_vertical_axis_info(int x, int y, int color)
{
    AxisInfo axis;
    axis.seg1 = get_segment_up(x, y, color);
    axis.seg2 = get_segment_down(x, y, color);
    axis.total_length = axis.seg1.length + axis.seg2.length;
    return axis;
}

AxisInfo get_diag1_axis_info(int x, int y, int color)
{
    // 左上-右下
    AxisInfo axis;
    axis.seg1 = get_segment_leftup(x, y, color);
    axis.seg2 = get_segment_rightdown(x, y, color);
    axis.total_length = axis.seg1.length + axis.seg2.length;
    return axis;
}

AxisInfo get_diag2_axis_info(int x, int y, int color)
{
    // 右上-左下
    AxisInfo axis;
    axis.seg1 = get_segment_rightup(x, y, color);
    axis.seg2 = get_segment_leftdown(x, y, color);
    axis.total_length = axis.seg1.length + axis.seg2.length;
    return axis;
}

// 3.兼容旧接口：只返回总长度，不再有任何全局副作用（基本废弃）

int return_line_self(int x, int y, int color)
{
    AxisInfo axis = get_horizontal_axis_info(x, y, color);
    return axis.total_length;
}

int return_column_self(int x, int y, int color)
{
    AxisInfo axis = get_vertical_axis_info(x, y, color);
    return axis.total_length;
}

int return_leftup_rightdown_self(int x, int y, int color)
{
    AxisInfo axis = get_diag1_axis_info(x, y, color);
    return axis.total_length;
}

int return_rightup_leftdown_self(int x, int y, int color)
{
    AxisInfo axis = get_diag2_axis_info(x, y, color);
    return axis.total_length;
}
