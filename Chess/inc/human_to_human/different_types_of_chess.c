#include "shared_variable_function.h"
// 用于判断各个棋型的文件
//  1. 判断是否形成长连（6连及以上，禁手相关）
int is_long_connected(int x, int y, int current_player)
{
    // 直接使用轴向总长度，逻辑不变：任一方向 >= 6 即为长连
    if (get_horizontal_axis_info(x, y, current_player).total_length >= 5)
        return 1;
    if (get_vertical_axis_info(x, y, current_player).total_length >= 5)
        return 1;
    if (get_diag1_axis_info(x, y, current_player).total_length >= 5)
        return 1;
    if (get_diag2_axis_info(x, y, current_player).total_length >= 5)
        return 1;
    return 0; // 没有达成6
}

// 2. 判断是否形成五连珠（胜负判定）
int check_five_in_a_row(int x, int y, int current_player)
{
    // 检查四个方向是否有五连珠
    AxisInfo hor = get_horizontal_axis_info(x, y, current_player);
    if (hor.total_length >= 4)
        return 1;

    AxisInfo ver = get_vertical_axis_info(x, y, current_player);
    if (ver.total_length >= 4)
        return 1;

    AxisInfo d1 = get_diag1_axis_info(x, y, current_player);
    if (d1.total_length >= 4)
        return 1;

    AxisInfo d2 = get_diag2_axis_info(x, y, current_player);
    if (d2.total_length >= 4)
        return 1;

    return 0; // 没有五连珠
}

// 3. 统计当前位置形成的活三数量
int num_of_huo_san(int x, int y, int current_player)
{
    int count = 0;

    AxisInfo hor = get_horizontal_axis_info(x, y, current_player);
    AxisInfo ver = get_vertical_axis_info(x, y, current_player);
    AxisInfo d1 = get_diag1_axis_info(x, y, current_player);
    AxisInfo d2 = get_diag2_axis_info(x, y, current_player);

    // ==================== 1. 连活三 (连三且两头空) ====================
    if (hor.total_length == 2 && hor.seg1.is_open && hor.seg2.is_open)
        count++;
    if (ver.total_length == 2 && ver.seg1.is_open && ver.seg2.is_open)
        count++;
    if (d1.total_length == 2 && d1.seg1.is_open && d1.seg2.is_open)
        count++;
    if (d2.total_length == 2 && d2.seg1.is_open && d2.seg2.is_open)
        count++;

    // ==================== 2. 跳活三（需要模拟落子） ====================
    // 横向跳活三支持 X . X Me / Me X . X / X Me . X / X . Me X 等形状。
    // ----------- A. 横向 -----------
    if (hor.total_length == 1)
    {
        // 左边连着一个子，形如 (X Me)
        if (hor.seg1.length == 1 && hor.seg1.is_open)
        {
            // 情况 1: 往左看远端 -> X . X Me
            if (in_bound(x, y - 3) && arrayForInnerBoardLayout[x][y - 3] == current_player)
            {
                if (if_put_is_huo_si(x, y - 2, current_player))
                    count++;
            }
            // 情况 1.5 (新增): 往右看 -> X Me . X (当前棋子夹在中间)
            // 此时 (x,y) 和左边组成 X Me，我们需要看右边隔一个位置(y+2)是不是也是 X
            if (in_bound(x, y + 2) && arrayForInnerBoardLayout[x][y + 2] == current_player)
            {
                // 如果在空位 (y+1) 落子能成活四，说明现在是跳活三
                if (if_put_is_huo_si(x, y + 1, current_player))
                    count++;
            }
        }

        // 右边连着一个子，形如 (Me X)
        if (hor.seg2.length == 1 && hor.seg2.is_open)
        {
            // 情况 2: 往右看远端 -> Me X . X
            if (in_bound(x, y + 3) && arrayForInnerBoardLayout[x][y + 3] == current_player)
            {
                if (if_put_is_huo_si(x, y + 2, current_player))
                    count++;
            }
            // 情况 2.5 (新增): 往左看 -> X . Me X (当前棋子夹在中间)
            // 此时 (x,y) 和右边组成 Me X，我们需要看左边隔一个位置(y-2)是不是也是 X
            if (in_bound(x, y - 2) && arrayForInnerBoardLayout[x][y - 2] == current_player)
            {
                // 如果在空位 (y-1) 落子能成活四，说明现在是跳活三
                if (if_put_is_huo_si(x, y - 1, current_player))
                    count++;
            }
        }
    }
    else if (hor.total_length == 0)
    {
        // 这一部分没有变，保持原样即可

        // 情况 3: 边缘跳三 X X . Me
        if (hor.seg1.is_open)
        {
            if (in_bound(x, y - 2) && arrayForInnerBoardLayout[x][y - 2] == current_player &&
                in_bound(x, y - 3) && arrayForInnerBoardLayout[x][y - 3] == current_player)
            {
                if (if_put_is_huo_si(x, y - 1, current_player))
                    count++;
            }
        }
        // 情况 4: 边缘跳三 Me . X X
        if (hor.seg2.is_open)
        {
            if (in_bound(x, y + 2) && arrayForInnerBoardLayout[x][y + 2] == current_player &&
                in_bound(x, y + 3) && arrayForInnerBoardLayout[x][y + 3] == current_player)
            {
                if (if_put_is_huo_si(x, y + 1, current_player))
                    count++;
            }
        }
        // 情况 5: 双跳三 X . Me . X
        if (hor.seg1.is_open && hor.seg2.is_open)
        {
            if (in_bound(x, y - 2) && arrayForInnerBoardLayout[x][y - 2] == current_player &&
                in_bound(x, y + 2) && arrayForInnerBoardLayout[x][y + 2] == current_player)
            {
                if (if_put_is_huo_si(x, y - 1, current_player))
                    count++;
            }
        }
    }
    // ----------- B. 纵向 -----------
    if (ver.total_length == 1)
    {
        // 上方跳三 X . X Me （从上到下）
        if (ver.seg1.length == 1 && ver.seg1.is_open)
        {
            if (in_bound(x - 3, y) && arrayForInnerBoardLayout[x - 3][y] == current_player)
            {
                if (if_put_is_huo_si(x - 2, y, current_player))
                    count++;
            }
            if (in_bound(x + 2, y) && arrayForInnerBoardLayout[x + 2][y] == current_player)
            {
                if (if_put_is_huo_si(x + 1, y, current_player))
                    count++;
            }
        }
        // 下方跳三 Me X . X
        if (ver.seg2.length == 1 && ver.seg2.is_open)
        {
            if (in_bound(x + 3, y) && arrayForInnerBoardLayout[x + 3][y] == current_player)
            {
                if (if_put_is_huo_si(x + 2, y, current_player))
                    count++;
            }
            if (in_bound(x - 2, y) && arrayForInnerBoardLayout[x - 2][y] == current_player)
            {
                if (if_put_is_huo_si(x - 1, y, current_player))
                    count++;
            }
        }
    }
    else if (ver.total_length == 0)
    {
        // 上方边缘跳三 X X . Me
        if (ver.seg1.is_open)
        {
            if (in_bound(x - 2, y) && arrayForInnerBoardLayout[x - 2][y] == current_player &&
                in_bound(x - 3, y) && arrayForInnerBoardLayout[x - 3][y] == current_player)
            {
                if (if_put_is_huo_si(x - 1, y, current_player))
                    count++;
            }
        }
        // 下方边缘跳三 Me . X X
        if (ver.seg2.is_open)
        {
            if (in_bound(x + 2, y) && arrayForInnerBoardLayout[x + 2][y] == current_player &&
                in_bound(x + 3, y) && arrayForInnerBoardLayout[x + 3][y] == current_player)
            {
                if (if_put_is_huo_si(x + 1, y, current_player))
                    count++;
            }
        }
        // 垂直双跳三 X . Me . X
        if (ver.seg1.is_open && ver.seg2.is_open)
        {
            if (in_bound(x - 2, y) && arrayForInnerBoardLayout[x - 2][y] == current_player &&
                in_bound(x + 2, y) && arrayForInnerBoardLayout[x + 2][y] == current_player)
            {
                if (if_put_is_huo_si(x - 1, y, current_player))
                    count++;
            }
        }
    }

    // ----------- C. 左上-右下方向 (\\ 方向) -----------
    if (d1.total_length == 1)
    {
        // 左上跳三: (x-3,y-3) X, (x-2,y-2) ., (x-1,y-1) X, (x,y) Me
        if (d1.seg1.length == 1 && d1.seg1.is_open)
        {
            if (in_bound(x - 3, y - 3) && arrayForInnerBoardLayout[x - 3][y - 3] == current_player)
            {
                if (if_put_is_huo_si(x - 2, y - 2, current_player))
                    count++;
            }
            if (in_bound(x + 2, y + 2) && arrayForInnerBoardLayout[x + 2][y + 2] == current_player)
            {
                if (if_put_is_huo_si(x + 1, y + 1, current_player))
                    count++;
            }
        }
        // 右下跳三: (x,y) Me, (x+1,y+1) X, (x+2,y+2) ., (x+3,y+3) X
        if (d1.seg2.length == 1 && d1.seg2.is_open)
        {
            if (in_bound(x + 3, y + 3) && arrayForInnerBoardLayout[x + 3][y + 3] == current_player)
            {
                if (if_put_is_huo_si(x + 2, y + 2, current_player))
                    count++;
            }
            if (in_bound(x - 2, y - 2) && arrayForInnerBoardLayout[x - 2][y - 2] == current_player)
            {
                if (if_put_is_huo_si(x - 1, y - 1, current_player))
                    count++;
            }
        }
    }
    else if (d1.total_length == 0)
    {
        // 左上边缘跳三: X X . Me 沿左上方向展开
        if (d1.seg1.is_open)
        {
            if (in_bound(x - 2, y - 2) && arrayForInnerBoardLayout[x - 2][y - 2] == current_player &&
                in_bound(x - 3, y - 3) && arrayForInnerBoardLayout[x - 3][y - 3] == current_player)
            {
                if (if_put_is_huo_si(x - 1, y - 1, current_player))
                    count++;
            }
        }
        // 右下边缘跳三: Me . X X
        if (d1.seg2.is_open)
        {
            if (in_bound(x + 2, y + 2) && arrayForInnerBoardLayout[x + 2][y + 2] == current_player &&
                in_bound(x + 3, y + 3) && arrayForInnerBoardLayout[x + 3][y + 3] == current_player)
            {
                if (if_put_is_huo_si(x + 1, y + 1, current_player))
                    count++;
            }
        }
        // 斜向双跳三: X . Me . X
        if (d1.seg1.is_open && d1.seg2.is_open)
        {
            if (in_bound(x - 2, y - 2) && arrayForInnerBoardLayout[x - 2][y - 2] == current_player &&
                in_bound(x + 2, y + 2) && arrayForInnerBoardLayout[x + 2][y + 2] == current_player)
            {
                if (if_put_is_huo_si(x - 1, y - 1, current_player))
                    count++;
            }
        }
    }

    // ----------- D. 右上-左下方向 (/ 方向) -----------
    if (d2.total_length == 1)
    {
        // 右上跳三: (x-3,y+3) X, (x-2,y+2) ., (x-1,y+1) X, (x,y) Me
        if (d2.seg1.length == 1 && d2.seg1.is_open)
        {
            if (in_bound(x - 3, y + 3) && arrayForInnerBoardLayout[x - 3][y + 3] == current_player)
            {
                if (if_put_is_huo_si(x - 2, y + 2, current_player))
                    count++;
            }
            if (in_bound(x + 2, y - 2) && arrayForInnerBoardLayout[x + 2][y - 2] == current_player)
            {
                if (if_put_is_huo_si(x + 1, y - 1, current_player))
                    count++;
            }
        }
        // 左下跳三: (x,y) Me, (x+1,y-1) X, (x+2,y-2) ., (x+3,y-3) X
        if (d2.seg2.length == 1 && d2.seg2.is_open)
        {
            if (in_bound(x + 3, y - 3) && arrayForInnerBoardLayout[x + 3][y - 3] == current_player)
            {
                if (if_put_is_huo_si(x + 2, y - 2, current_player))
                    count++;
            }
            if (in_bound(x - 2, y + 2) && arrayForInnerBoardLayout[x - 2][y + 2] == current_player)
            {
                if (if_put_is_huo_si(x - 1, y + 1, current_player))
                    count++;
            }
        }
    }
    else if (d2.total_length == 0)
    {
        // 右上边缘跳三: X X . Me
        if (d2.seg1.is_open)
        {
            if (in_bound(x - 2, y + 2) && arrayForInnerBoardLayout[x - 2][y + 2] == current_player &&
                in_bound(x - 3, y + 3) && arrayForInnerBoardLayout[x - 3][y + 3] == current_player)
            {
                if (if_put_is_huo_si(x - 1, y + 1, current_player))
                    count++;
            }
        }
        // 左下边缘跳三: Me . X X
        if (d2.seg2.is_open)
        {
            if (in_bound(x + 2, y - 2) && arrayForInnerBoardLayout[x + 2][y - 2] == current_player &&
                in_bound(x + 3, y - 3) && arrayForInnerBoardLayout[x + 3][y - 3] == current_player)
            {
                if (if_put_is_huo_si(x + 1, y - 1, current_player))
                    count++;
            }
        }
        // 斜向双跳三: X . Me . X
        if (d2.seg1.is_open && d2.seg2.is_open)
        {
            if (in_bound(x - 2, y + 2) && arrayForInnerBoardLayout[x - 2][y + 2] == current_player &&
                in_bound(x + 2, y - 2) && arrayForInnerBoardLayout[x + 2][y - 2] == current_player)
            {
                if (if_put_is_huo_si(x - 1, y + 1, current_player))
                    count++;
            }
        }
    }

    return count;
}
// 旧版 num_of_huo_si 已废弃，这里仅保留使用 AxisInfo 的新版实现
// 4. 判断在(x, y)落子后是否能形成活四（用于模拟落子判断禁手/冲四等）
int if_put_is_huo_si(int x, int y, int current_player)
{
    // 记录原来的棋子（有可能是 EMPTY / BLACK / WHITE）
    int old = arrayForInnerBoardLayout[x][y];

    // 临时放置棋子
    arrayForInnerBoardLayout[x][y] = current_player;

    // 计算活四数量
    int huo_si_count = num_of_huo_si(x, y, current_player);

    // 恢复原状（不论原来是什么都还回去，避免误删已有棋子）
    arrayForInnerBoardLayout[x][y] = old;

    // 如果活四数量大于等于1，则该位置为活四
    if (huo_si_count >= 1)
        return 1;
    else
        return 0;
}

// 5. 统计当前位置形成的活四数量
int num_of_huo_si(int x, int y, int current_player)
{
    int count = 0;

    AxisInfo hor = get_horizontal_axis_info(x, y, current_player);
    AxisInfo ver = get_vertical_axis_info(x, y, current_player);
    AxisInfo d1 = get_diag1_axis_info(x, y, current_player);
    AxisInfo d2 = get_diag2_axis_info(x, y, current_player);

    // 1. 横向
    if (hor.total_length == 3 && hor.seg1.is_open == 1 && hor.seg2.is_open == 1)
        count++;
    // 2. 纵向
    if (ver.total_length == 3 && ver.seg1.is_open == 1 && ver.seg2.is_open == 1)
        count++;
    // 3. 左上-右下方向
    if (d1.total_length == 3 && d1.seg1.is_open == 1 && d1.seg2.is_open == 1)
        count++;
    // 4. 右上-左下方向
    if (d2.total_length == 3 && d2.seg1.is_open == 1 && d2.seg2.is_open == 1)
        count++;

    return count;
}

// 辅助函数：判断在 (x, y) 落子后，是否构成了五连珠（纯计算，不修改任何全局状态）

// 6. 统计当前位置形成的冲四数量（冲四：一头空一头堵的连续四个子，禁手/胜负判定用）
int num_of_chong_si(int x, int y, int current_player)
{
    int count = 0;

    // 我们需要依次检查四个方向：横、纵、左上右下、右上左下
    // 为了代码清晰且防止全局变量冲突，我们把每个方向的逻辑分开写（或封装）

    // ==========================================
    // 1. 横向检测 (Line)
    // ==========================================
    AxisInfo hor = get_horizontal_axis_info(x, y, current_player);
    int len_line = hor.total_length;

    int left_valid = hor.seg1.is_open;    // 左边是否空
    int right_valid = hor.seg2.is_open;   // 右边是否空
    Point left_gap = hor.seg1.gap_coord;  // 左边空位坐标
    Point right_gap = hor.seg2.gap_coord; // 右边空位坐标

    // 逻辑 A：连冲 (如 X XXX 0)
    // 特征：连子数为3 (总4)，且一头空一头堵 (异或逻辑)
    if (len_line == 3 && (left_valid != right_valid))
    {
        count++;
    }
    // 逻辑 B：跳冲 (如 X X 0 XX)
    // 特征：连子数不够3，但填补某个空位后能成5
    else if (len_line < 3)
    {
        // 尝试填左边的坑
        if (left_valid)
        {
            arrayForInnerBoardLayout[left_gap.x][left_gap.y] = current_player; // 模拟落子
            if (check_five_in_a_row(left_gap.x, left_gap.y, current_player))
            { // 检查是否赢了
                count++;
            }
            arrayForInnerBoardLayout[left_gap.x][left_gap.y] = EMPTY; // 还原现场
        }
        // 尝试填右边的坑 (注意：如果左右是同一个空位需避免重复，但在五子棋逻辑中通常分开算威胁)
        // 加上 else if 可以避免同一条线重复计算双向威胁，视你的策略而定，通常分开if更好
        if (right_valid)
        {
            arrayForInnerBoardLayout[right_gap.x][right_gap.y] = current_player;
            if (check_five_in_a_row(right_gap.x, right_gap.y, current_player))
            {
                count++;
            }
            arrayForInnerBoardLayout[right_gap.x][right_gap.y] = EMPTY;
        }
    }

    // ==========================================
    // 2. 纵向检测 (Column)
    // ==========================================
    AxisInfo ver = get_vertical_axis_info(x, y, current_player);
    int len_col = ver.total_length;
    int up_valid = ver.seg1.is_open;
    int down_valid = ver.seg2.is_open;
    Point up_gap = ver.seg1.gap_coord;
    Point down_gap = ver.seg2.gap_coord;

    if (len_col == 3 && (up_valid != down_valid))
    {
        count++;
    }
    else if (len_col < 3)
    {
        if (up_valid)
        {
            arrayForInnerBoardLayout[up_gap.x][up_gap.y] = current_player;
            if (check_five_in_a_row(up_gap.x, up_gap.y, current_player))
                count++;
            arrayForInnerBoardLayout[up_gap.x][up_gap.y] = EMPTY;
        }
        if (down_valid)
        {
            arrayForInnerBoardLayout[down_gap.x][down_gap.y] = current_player;
            if (check_five_in_a_row(down_gap.x, down_gap.y, current_player))
                count++;
            arrayForInnerBoardLayout[down_gap.x][down_gap.y] = EMPTY;
        }
    }

    // ==========================================
    // 3. 左上-右下检测 (\)
    // ==========================================
    AxisInfo d1 = get_diag1_axis_info(x, y, current_player);
    int len_lr = d1.total_length;
    int lu_valid = d1.seg1.is_open;
    int rd_valid = d1.seg2.is_open;
    Point lu_gap = d1.seg1.gap_coord;
    Point rd_gap = d1.seg2.gap_coord;

    if (len_lr == 3 && (lu_valid != rd_valid))
    {
        count++;
    }
    else if (len_lr < 3)
    {
        if (lu_valid)
        {
            arrayForInnerBoardLayout[lu_gap.x][lu_gap.y] = current_player;
            if (check_five_in_a_row(lu_gap.x, lu_gap.y, current_player))
                count++;
            arrayForInnerBoardLayout[lu_gap.x][lu_gap.y] = EMPTY;
        }
        if (rd_valid)
        {
            arrayForInnerBoardLayout[rd_gap.x][rd_gap.y] = current_player;
            if (check_five_in_a_row(rd_gap.x, rd_gap.y, current_player))
                count++;
            arrayForInnerBoardLayout[rd_gap.x][rd_gap.y] = EMPTY;
        }
    }

    // ==========================================
    // 4. 右上-左下检测 (/)
    // ==========================================
    AxisInfo d2 = get_diag2_axis_info(x, y, current_player);
    int len_rl = d2.total_length;
    int ru_valid = d2.seg1.is_open;
    int ld_valid = d2.seg2.is_open;
    Point ru_gap = d2.seg1.gap_coord;
    Point ld_gap = d2.seg2.gap_coord;

    if (len_rl == 3 && (ru_valid != ld_valid))
    {
        count++;
    }
    else if (len_rl < 3)
    {
        if (ru_valid)
        {
            arrayForInnerBoardLayout[ru_gap.x][ru_gap.y] = current_player;
            if (check_five_in_a_row(ru_gap.x, ru_gap.y, current_player))
                count++;
            arrayForInnerBoardLayout[ru_gap.x][ru_gap.y] = EMPTY;
        }
        if (ld_valid)
        {
            arrayForInnerBoardLayout[ld_gap.x][ld_gap.y] = current_player;
            if (check_five_in_a_row(ld_gap.x, ld_gap.y, current_player))
                count++;
            arrayForInnerBoardLayout[ld_gap.x][ld_gap.y] = EMPTY;
        }
    }

    return count;
}

// 7. 统计当前位置形成的眠三数量（眠三：一头空一头堵的连续三个子，禁手/胜负判定用）
int num_of_mian_san(int x, int y, int current_player)
{
    int count = 0;

    // 我们需要依次检查四个方向：横、纵、左上右下、右上左下
    // 为了代码清晰且防止全局变量冲突，我们把每个方向的逻辑分开写（或封装）

    // ==========================================
    // 1. 横向检测 (Line)
    // ==========================================
    AxisInfo hor = get_horizontal_axis_info(x, y, current_player);
    int len_line = hor.total_length;

    int left_valid = hor.seg1.is_open;    // 左边是否空
    int right_valid = hor.seg2.is_open;   // 右边是否空
    Point left_gap = hor.seg1.gap_coord;  // 左边空位坐标
    Point right_gap = hor.seg2.gap_coord; // 右边空位坐标

    // 逻辑 A：连冲 (如 X XXX 0)
    // 特征：连子数为3 (总4)，且一头空一头堵 (异或逻辑)
    if (len_line == 2 && (left_valid != right_valid))
    {
        count++;
    }
    // 逻辑 B：跳冲 (如 X X 0 XX)
    // 特征：连子数不够3，但填补某个空位后能成5
    else if (len_line < 2)
    {
        // 尝试填左边的坑
        if (left_valid)
        {
            arrayForInnerBoardLayout[left_gap.x][left_gap.y] = current_player; // 模拟落子
            if (num_of_chong_si(left_gap.x, left_gap.y, current_player) > 0)
            { // 检查是否形成冲四
                count++;
            }
            arrayForInnerBoardLayout[left_gap.x][left_gap.y] = EMPTY; // 还原现场
        }
        // 尝试填右边的坑 (注意：如果左右是同一个空位需避免重复，但在五子棋逻辑中通常分开算威胁)
        // 加上 else if 可以避免同一条线重复计算双向威胁，视你的策略而定，通常分开if更好
        if (right_valid)
        {
            arrayForInnerBoardLayout[right_gap.x][right_gap.y] = current_player;
            if (num_of_chong_si(right_gap.x, right_gap.y, current_player) > 0)
            {
                count++;
            }
            arrayForInnerBoardLayout[right_gap.x][right_gap.y] = EMPTY;
        }
    }

    // ==========================================
    // 2. 纵向检测 (Column)
    // ==========================================
    AxisInfo ver = get_vertical_axis_info(x, y, current_player);
    int len_col = ver.total_length;
    int up_valid = ver.seg1.is_open;
    int down_valid = ver.seg2.is_open;
    Point up_gap = ver.seg1.gap_coord;
    Point down_gap = ver.seg2.gap_coord;

    if (len_col == 2 && (up_valid != down_valid))
    {
        count++;
    }
    else if (len_col < 2)
    {
        if (up_valid)
        {
            arrayForInnerBoardLayout[up_gap.x][up_gap.y] = current_player;
            if (num_of_chong_si(up_gap.x, up_gap.y, current_player) > 0)
                count++;
            arrayForInnerBoardLayout[up_gap.x][up_gap.y] = EMPTY;
        }
        if (down_valid)
        {
            arrayForInnerBoardLayout[down_gap.x][down_gap.y] = current_player;
            if (num_of_chong_si(down_gap.x, down_gap.y, current_player) > 0)
                count++;
            arrayForInnerBoardLayout[down_gap.x][down_gap.y] = EMPTY;
        }
    }

    // ==========================================
    // 3. 左上-右下检测 (\)
    // ==========================================
    AxisInfo d1 = get_diag1_axis_info(x, y, current_player);
    int len_lr = d1.total_length;
    int lu_valid = d1.seg1.is_open;
    int rd_valid = d1.seg2.is_open;
    Point lu_gap = d1.seg1.gap_coord;
    Point rd_gap = d1.seg2.gap_coord;

    if (len_lr == 2 && (lu_valid != rd_valid))
    {
        count++;
    }
    else if (len_lr < 2)
    {
        if (lu_valid)
        {
            arrayForInnerBoardLayout[lu_gap.x][lu_gap.y] = current_player;
            if (num_of_chong_si(lu_gap.x, lu_gap.y, current_player) > 0)
                count++;
            arrayForInnerBoardLayout[lu_gap.x][lu_gap.y] = EMPTY;
        }
        if (rd_valid)
        {
            arrayForInnerBoardLayout[rd_gap.x][rd_gap.y] = current_player;
            if (num_of_chong_si(rd_gap.x, rd_gap.y, current_player) > 0)
                count++;
            arrayForInnerBoardLayout[rd_gap.x][rd_gap.y] = EMPTY;
        }
    }

    // ==========================================
    // 4. 右上-左下检测 (/)
    // ==========================================
    AxisInfo d2 = get_diag2_axis_info(x, y, current_player);
    int len_rl = d2.total_length;
    int ru_valid = d2.seg1.is_open;
    int ld_valid = d2.seg2.is_open;
    Point ru_gap = d2.seg1.gap_coord;
    Point ld_gap = d2.seg2.gap_coord;

    if (len_rl == 2 && (ru_valid != ld_valid))
    {
        count++;
    }
    else if (len_rl < 2)
    {
        if (ru_valid)
        {
            arrayForInnerBoardLayout[ru_gap.x][ru_gap.y] = current_player;
            if (num_of_chong_si(ru_gap.x, ru_gap.y, current_player) > 0)
                count++;
            arrayForInnerBoardLayout[ru_gap.x][ru_gap.y] = EMPTY;
        }
        if (ld_valid)
        {
            arrayForInnerBoardLayout[ld_gap.x][ld_gap.y] = current_player;
            if (num_of_chong_si(ld_gap.x, ld_gap.y, current_player) > 0)
                count++;
            arrayForInnerBoardLayout[ld_gap.x][ld_gap.y] = EMPTY;
        }
    }

    return count;
}
// 8. 统计当前位置形成的活二数量（活二：两头空的连续两个子，主要用于AI评估）
int num_of_huo_er(int x, int y, int current_player)
{
    int count = 0;

    // 定义：活二 = 一条线上只有 1 颗己子（不含中心），且两端都空
    // 只统计简单连活二，和 num_of_mian_er 对称

    // 1. 横向
    AxisInfo hor2 = get_horizontal_axis_info(x, y, current_player);
    if (hor2.total_length == 1 && hor2.seg1.is_open && hor2.seg2.is_open)
    {
        count++;
    }

    // 2. 纵向
    AxisInfo ver2 = get_vertical_axis_info(x, y, current_player);
    if (ver2.total_length == 1 && ver2.seg1.is_open && ver2.seg2.is_open)
    {
        count++;
    }

    // 3. 左上-右下
    AxisInfo d12 = get_diag1_axis_info(x, y, current_player);
    if (d12.total_length == 1 && d12.seg1.is_open && d12.seg2.is_open)
    {
        count++;
    }

    // 4. 右上-左下
    AxisInfo d22 = get_diag2_axis_info(x, y, current_player);
    if (d22.total_length == 1 && d22.seg1.is_open && d22.seg2.is_open)
    {
        count++;
    }

    return count;
}
// 9. 统计当前位置形成的眠二数量（眠二：一头空一头堵的连续两个子，主要用于AI评估）
int num_of_mian_er(int x, int y, int current_player)
{
    int count = 0;

    // 简单的连眠二，到此结束吧

    // 1. 横向
    AxisInfo hor = get_horizontal_axis_info(x, y, current_player);
    // is_open 不相等，表示一个是1一个是0（即一头堵一头空）
    if (hor.total_length == 1 && (hor.seg1.is_open != hor.seg2.is_open))
    {
        count++;
    }

    // 2. 纵向
    AxisInfo ver = get_vertical_axis_info(x, y, current_player);
    if (ver.total_length == 1 && (ver.seg1.is_open != ver.seg2.is_open))
    {
        count++;
    }

    // 3. 左上-右下
    AxisInfo d1 = get_diag1_axis_info(x, y, current_player);
    if (d1.total_length == 1 && (d1.seg1.is_open != d1.seg2.is_open))
    {
        count++;
    }

    // 4. 右上-左下
    AxisInfo d2 = get_diag2_axis_info(x, y, current_player);
    if (d2.total_length == 1 && (d2.seg1.is_open != d2.seg2.is_open))
    {
        count++;
    }

    return count;
}
