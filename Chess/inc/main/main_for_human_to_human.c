
#include "shared_variable_function.h"
void clear_input_buffer(void)
// 清空缓存
{
    int c;
    while ((c = getchar()) != '\n' && c != EOF)
        ;
}
int is_unapproved_move(int x, int y, int current_player, int result)
// 判断是否为不合法落子
{
    if (result != 2)
    {
        printf("Invalid input, please enter numeric coordinates!\n");
        return 1;
    }
    if (!in_bound(x, y))
    {
        printf("Move out of bounds, please re-enter!\n");
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
// 人类对人类对战函数
{
    int current_player = first_player; // 假设黑棋先手
    int x, y;
    int result;
    while (1)
    {
        printf("Enter your move (x y): ");
        result = scanf("%d %d", &x, &y);
        if (result != 2)
        {
            clear_input_buffer();
        }
        if (is_unapproved_move(x, y, current_player, result))
        {
            printf("This move is not allowed due to game rules, please re-enter!\n");
            continue;
        }
        placeChess(x, y, current_player);
        innerLayoutToDisplayArray();
        displayBoard();
        char col_char = 'A' + y; // 列：0->A, 1->B, ...
        int row_num = SIZE - x;  // 行：0->SIZE, 1->SIZE-1,... 例如 6->9
        printf("Player %d placed at (%d, %d) mapped to (%c, %d)\n",
               current_player, x, y, col_char, row_num);
        if (current_player == first_player)
        {
            if (is_banned_by_long_connected(x, y, current_player))
            {
                winner = (current_player == BLACK) ? WHITE : BLACK;
                printf("BANNED FOUND,long connected! ");
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
    printf("Enter first player (1=Black 2=White): ");
    scanf("%d", &first_player);
    play(first_player);
    if (winner == BLACK)
    {
        printf("Black wins the game!\n");
    }
    else if (winner == WHITE)
    {
        printf("White wins the game!\n");
    }
    return 0;
}