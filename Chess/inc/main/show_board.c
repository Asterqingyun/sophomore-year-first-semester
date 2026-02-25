#include "shared_variable_function.h"
//输出棋盘的实现文件
// 使用纯 ASCII 棋盘，行首是行号，后面是由空格分隔的点阵

// 空棋盘模板：'.' 表示空格子
char arrayForEmptyBoard[SIZE][DISPLAY_COLS] = {
    "15 . . . . . . . . . . . . . . . ",
    "14 . . . . . . . . . . . . . . . ",
    "13 . . . . . . . . . . . . . . . ",
    "12 . . . . . . . . . . . . . . . ",
    "11 . . . . . . . . . . . . . . . ",
    "10 . . . . . . . . . . . . . . . ",
    "09 . . . . . . . . . . . . . . . ",
    "08 . . . . . . . . . . . . . . . ",
    "07 . . . . . . . . . . . . . . . ",
    "06 . . . . . . . . . . . . . . . ",
    "05 . . . . . . . . . . . . . . . ",
    "04 . . . . . . . . . . . . . . . ",
    "03 . . . . . . . . . . . . . . . ",
    "02 . . . . . . . . . . . . . . . ",
    "01 . . . . . . . . . . . . . . . ",
};
// 显示用棋盘
char arrayForDisplayBoard[SIZE][DISPLAY_COLS];

// 棋子符号（ASCII）
char play1Pic[] = "X"; // 黑棋
char play1CurrentPic[] = "X";

char play2Pic[] = "O"; // 白棋
char play2CurrentPic[] = "O";

// 记录内部布局
int arrayForInnerBoardLayout[SIZE][SIZE];

// 初始化内部棋盘
void initRecordBorard(void)
{
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            arrayForInnerBoardLayout[i][j] = EMPTY;
        }
    }
}

// 将内部布局转成可显示棋盘
void innerLayoutToDisplayArray(void)
{
    for (int i = 0; i < SIZE; i++)
    {
        strcpy(arrayForDisplayBoard[i], arrayForEmptyBoard[i]);
    }

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            // index 0-1: 行号, 2: 空格, 3: 第一个格子的符号, 4: 空格, 5: 第二个格子的符号...
            int col = 3 + j * 2;
            if (col < DISPLAY_COLS - 1)
            {
                switch (arrayForInnerBoardLayout[i][j])
                {
                case BLACK:
                    arrayForDisplayBoard[i][col] = 'X';
                    break;
                case WHITE:
                    arrayForDisplayBoard[i][col] = 'O';
                    break;
                default:
                    break;
                }
            }
        }
    }
}

// 显示棋盘
void displayBoard(void)
{
    // 清屏
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif

    // 输出棋盘
    for (int i = 0; i < SIZE; i++)
    {
        printf("%s", arrayForDisplayBoard[i]);

        // 右侧侧边栏：显示 AI 递归层数/节点数/危险模式（调试使用）
        if (i == 0)
            printf("   | AI depth(max): %d", ai_last_max_depth_reached);
        else if (i == 1)
            printf("   | AI nodes: %lld", ai_last_nodes_visited);
        else if (i == 2)
            printf("   | danger: %s", ai_last_danger_triggered ? "YES" : "NO");
        else if (i == 3)
            printf("   | oppThreat(root): %lld", ai_last_root_opp_threat);
        else if (i == 4)
            printf("   | maxLayer(eff): %d", ai_last_effective_max_layer);
        else if (i == 5)
            printf("   | cache: %lld/%lld", ai_last_cache_hits, ai_last_cache_lookups);
        else
        {
            int layer = i - 5;
            if (layer >= 1 && layer < AI_STATS_LAYER_CAP)
                printf("   | L%-2d: %lld", layer, ai_last_nodes_by_layer[layer]);
        }
        printf("\n");
    }

    // 输出列标 A..O，对齐棋盘
    printf("   "); // 行号2+空格1
    for (int j = 0; j < SIZE; j++)
        printf("%c ", 'A' + j);
    printf("\n");
}
