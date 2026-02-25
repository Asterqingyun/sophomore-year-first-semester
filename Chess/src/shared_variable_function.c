#include "shared_variable_function.h"
#include <stdio.h>

// 历史记录栈实现
Point stack[SIZE * SIZE];
int top = 0;

// 全局步数变量定义
int step = 0;

void push(Point p)
{
    if (top < SIZE * SIZE)
    {
        stack[top] = p;
        top++;
    }
    else
    {
        printf("The chess board is full, cannot push more points!\n");
    }
}

Point pop(void)
{
    if (top > 0)
    {
        top--;
        return stack[top];
    }
    else
    {
        printf("The stack is empty, cannot pop more points!\n");
        Point p;
        p.x = -1;
        p.y = -1;
        return p;
    }
}
