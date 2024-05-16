#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <math.h>

//凡是挂有“__global__”或者“__device__”前缀的函数，都是在GPU上运行的设备程序，不同的是__global__设备程序可被主机程序调用，而__device__设备程序则只能被设备程序调用。
//没有挂任何前缀的函数，都是主机程序。主机程序显示声明可以用__host__前缀

/*

cuda程序会使用gpu和cpu内存
2.cpu内存释放与分配是标准的
    1）栈，自动分配的
    2）堆：用户自己分配释放的，如new，delete，malloc，free等

3.gpu内存分配
    1) cudaMalloc(void**devPtr , size_t size)
    2) cudafree(void*devPtr)

4.gpu,cpu都可以访问的内存,统一内存
    1)  cudaMallocManaged(void**devPtr , size_t size)
    2)  cudafree(void*devPtr)

5.gpu内存拷贝
cudaMemcpy(depth_vec.data, depth_in, depthWidth * depthHeight * sizeof(float), cudaMemcpyDeviceToHost);

*/

 //Kernel function to add the elements of two arrays
__global__
void add(int n, float* x, float* y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        y[i] = x[i] + y[i];
        //printf("Thread %d in Block %d has index %d\n", threadIdx.x, blockIdx.x, index);
        if (i == 0) { // 仅在第一个线程中打印，避免重复输出
            printf("Block index: (%d, %d, %d), Thread index: (%d, %d, %d)\n",
                blockIdx.x, blockIdx.y, blockIdx.z,
                threadIdx.x, threadIdx.y, threadIdx.z);
            printf("Block dimensions: (%d, %d, %d), Grid dimensions: (%d, %d, %d)\n",
                blockDim.x, blockDim.y, blockDim.z,
                gridDim.x, gridDim.y, gridDim.z);
        }
    }
        
}

int main1(void)
{
    cudaError_t cudaStatus;
    int num = 0;
    cudaDeviceProp prop;
    cudaStatus = cudaGetDeviceCount(&num);
    for (int i = 0; i < num; i++)
    {
        cudaGetDeviceProperties(&prop, i);
    }

    int N = 10000;
    float* x, * y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add << <numBlocks, blockSize >> > (N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}

struct MyClass
{
    std::vector<int> data;
    //size_t size;
    //int* ptr;
};

static int main2(void)
{
    std::vector<int> h_b;
    int* d_a;
    cudaMalloc((void**)&d_a, sizeof(int) * h_b.size());
    cudaMemcpy(d_a, h_b.data(), sizeof(int) * h_b.size(), cudaMemcpyHostToDevice);
    return 0;
}

static int _enrol = []()
    {
        main1();
        return 0;
    }();