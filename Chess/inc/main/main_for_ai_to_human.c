#include "shared_variable_function.h"
// 引入ctype.h以使用toupper函数

// 只声明，不定义，避免重复定义
extern Point stack[SIZE * SIZE];
extern int top;
extern int step;

// 修改print_stack，确保输出符合(1-15, A-O)的格式
void print_stack()
{
    printf("the steps of the chess:\n");
    for (int i = 0; i < top; i++)
    {
        // 内部x(0-14) -> 显示x(1-15)
        // 内部y(0-14) -> 显示y(A-O)
        printf("(%d, %c) ", stack[i].x + 1, 'A' + stack[i].y);
    }
    printf("\n");
}

// 只声明，不实现，避免重复定义
void push(Point p);
Point pop(void);

void clear_input_buffer(void)
{
    int c;
    while ((c = getchar()) != '\n' && c != EOF)
        ;
}

int is_unapproved_move(int x, int y, int current_player, int result)
{
    if (result != 2)
    {
        // 这里的result是scanf的返回值
        return 1;
    }
    // 检查是否越界 (注意：这里的x, y已经是转换后的内部坐标 0-14)
    if (!in_bound(x, y))
    {
        printf("Move out of bounds (1-15, A-O), please re-enter!\n");
        return 1;
    }
    if (return_what_is_at(x, y) != EMPTY)
    {
        printf("This position is already occupied, please re-enter!\n");
        return 1;
    }
    return 0;
}

void play(int first_player)
{
    int is_empty = 1;
    int current_player = first_player; // 假设黑棋先手
    int x, y;                          // 这里存储的是内部数组坐标 (0-14)
    int result;

    while (1)
    {
        if (current_player == BLACK)
        {
            printf("Black(ai) is thinking...\n");
            if (is_empty)
            {
                x = SIZE / 2;
                y = SIZE / 2;
                is_empty = 0;
            }
            else
            {
                dian_and_fenshu move_black = give_an_answer(arrayForInnerBoardLayout, BLACK, first_player);
                x = move_black.dian.x;
                y = move_black.dian.y;
            }
        }
        else
        {
            // 人类玩家输入提示修改
            printf("Human please enter your move (Row 1-15, Col A-O, e.g., 8 H or H 8): ");
            int x_input = -1;
            int y_input_int = -1;
            char y_input[8] = {0};
            char input1[8] = {0}, input2[8] = {0};
            result = scanf("%s %s", input1, input2);
            is_empty = 0;

            // 支持两种输入格式：8 H 或 H 8
            if (result == 2)
            {
                if (isdigit(input1[0]))
                {
                    // 8 H 格式
                    x_input = atoi(input1);
                    strncpy(y_input, input2, 7);
                }
                else if (isalpha(input1[0]))
                {
                    // H 8 格式
                    strncpy(y_input, input1, 7);
                    x_input = atoi(input2);
                }
            }

            // 特殊功能：输入 -1 进行悔棋 (Regret)
            if (result == 2 && x_input == -1)
            {
                Point p1 = pop();
                Point p2 = pop();
                if (p1.x == -1 || p2.x == -1)
                {
                    printf("No moves to regret!\n");
                    continue;
                }
                else
                {
                    arrayForInnerBoardLayout[p2.x][p2.y] = EMPTY;
                    arrayForInnerBoardLayout[p1.x][p1.y] = EMPTY;
                    innerLayoutToDisplayArray();
                    displayBoard();
                    current_player = WHITE;
                    step -= 2;
                    if (step < 0)
                        step = 0;
                    printf("Regret successful, it's now Player %d's turn.\n", current_player);
                    continue;
                }
            }

            // 1. 输入格式校验
            if (result != 2 || strlen(y_input) != 1 || x_input < 1 || x_input > 15)
            {
                clear_input_buffer();
                printf("Invalid format! Please enter numeric row and letter column (e.g., 8 H or H 8).\n");
                continue;
            }

            // 2. 坐标转换逻辑
            user_to_internal(y_input, x_input, &x, &y);

            // 3. 逻辑校验
            if (!in_bound(x, y))
            {
                printf("Invalid range! Row must be 1-15, Col must be A-O.\n");
                clear_input_buffer();
                continue;
            }
            if (is_unapproved_move(x, y, current_player, result))
            {
                continue;
            }
        }

        // 落子逻辑
        placeChess(x, y, current_player);
        step++;
        innerLayoutToDisplayArray();
        displayBoard(); // 你的displayBoard应该不需要改，它负责画图
        push((Point){x, y});

        // 胜负判定逻辑
        if (current_player == first_player)
        {
            if (is_banned_by_long_connected(x, y, current_player))
            {
                winner = (current_player == BLACK) ? WHITE : BLACK;
                printf("BANNED FOUND, long connected! ");
                break;
            }
        }
        if (check_5(x, y, current_player))
        {
            winner = current_player;
            break;
        }
        if (current_player == first_player)
        {
            if (is_banned_after(x, y, current_player))
            {
                winner = (current_player == BLACK) ? WHITE : BLACK;
                printf("BANNED FOUND! ");
                break;
            }
        }

        // 修改落子后的提示信息，将内部坐标转换回人类坐标显示
        // x+1: 0->1, 7->8
        // 'A'+y: 0->A, 7->H
        printf("Player %d placed at (%d, %c)\n", current_player, x + 1, 'A' + y);

        current_player = (current_player == BLACK) ? WHITE : BLACK;
        printf("Next player's turn.\n"); // 切换玩家
    }
}

int main(int argc, char *argv[])
{
    int first_player;
    initRecordBorard();
    innerLayoutToDisplayArray();
    displayBoard();
    printf("We tend to regard Black as ai and White as human.\n");
    printf("Enter first player (1=Black(ai) 2=White(human)): ");

    // 增加一点容错，防止输入非数字导致无限循环
    if (scanf("%d", &first_player) != 1)
    {
        printf("Invalid input, defaulting to Black(ai).\n");
        first_player = BLACK;
    }

    play(first_player);

    if (winner == BLACK)
    {
        printf("Black(ai) wins the game!\n");
    }
    else if (winner == WHITE)
    {
        printf("White(human) wins the game!\n");
    }
    print_stack();
    return 0;
}