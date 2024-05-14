//#pragma once

#ifndef COMMON_TYPE_DEFINE
#define COMMON_TYPE_DEFINE

__global__ void saxpy(int n, float a, float* x, float* y);

__global__ void kernel(float* a, int offset);

#endif