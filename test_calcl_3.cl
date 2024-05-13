#include "commonTypeDefine.h"

__kernel void add_matrices(
    __global const float* A, 
    __global const float* B, 
    __global float* C )
{
    int id = get_global_id(0);
    C[id] = A[id] + B[id];
}

//struct Trigon2d
//{
//    double2 m_triangle[3];
//};
typedef struct {
    double* data[2];
} double2;

typedef struct {
    double2* triangle[3];
} Trigon2d;

__kernel void add_triangle(
    __global const Trigon2d* A,
    __global const Trigon2d* B,
    __global Trigon2d* C)
{
    int id = get_global_id(0);
    for (int i = 0; i < 3; i++)
    {
        C[id].triangle[i] = A[id].triangle[i] + B[id].triangle[i];
    }
}

/*

get_global_size(idx)： 获取第 idx 维度数据项的长度。

get_global_id(idx)： 获取第 idx 维度当前数据项索引。

*/