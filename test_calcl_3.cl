#include "commonTypeDefine.h"

__kernel void add_matrices(
    __global const float* A, 
    __global const float* B, 
    __global float* C )
{
    int i = get_global_id(0);
    C[i] = A[i] + B[i];
}

//typedef struct {
//    double* data[2];
//} double2;

//typedef struct {
//    double2* trigon[3];
//} Trigon2d;

//typedef struct {
//    double x;
//    double y;
//} Vector2d;

typedef struct {
    double2 trigon[3];
} Trigon2d;

__kernel void add_triangle(
    __global const Trigon2d* A,
    __global const Trigon2d* B,
    __global Trigon2d* C)
{
    int i = get_global_id(0);
    for (int j = 0; j < 3; j++)
    {
        C[i].trigon[j][0] = A[i].trigon[j][0] + B[i].trigon[j][0]; //x
        C[i].trigon[j][1] = A[i].trigon[j][1] + B[i].trigon[j][1]; //y
        //C[i].trigon[j].x = A[i].trigon[j].x + B[i].trigon[j].x; //x
        //C[i].trigon[j].y = A[i].trigon[j].y + B[i].trigon[j].y; //y
    }
}


/*

get_global_size(idx)： 获取第 idx 维度数据项的长度。

get_global_id(idx)： 获取第 idx 维度当前数据项索引。

*/