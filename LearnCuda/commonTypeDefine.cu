#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "commonTypeDefine.h"
using namespace std;

__global__ void saxpy(int n, float a, float* x, float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}


__global__ void kernel(float* a, int offset)
{
    int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
    float x = (float)i;
    float s = sinf(x);
    float c = cosf(x);
    a[i] = a[i] + sqrtf(s * s + c * c);
}