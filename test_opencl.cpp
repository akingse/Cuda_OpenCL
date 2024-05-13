
// https://github.com/KhronosGroup/OpenCL-SDK/releases
// https://developer.nvidia.com/cuda-toolkit

/*
��ַ�ռ����η�
OpenCL�Ĵ洢��ģ�ͷֱ�Ϊ��ȫ�ִ洢�����ֲ��洢���������洢����˽�д洢����
��Ӧ�ĵ�ַ�ռ����η�Ϊ��__global(��global)��__local(��local)��__constant(��constant)��__private(��private)��
__global���������ݽ���������ȫ���ڴ��С�
__constant���������ݽ��洢��ȫ��ֻ���ڴ��У���Ч����
__local���������ݽ��洢�ڱ����ڴ��С�
__private���������ݽ��洢��˽���ڴ��У�Ĭ�ϣ���

__kernel��kernel�����η�����һ������Ϊ�ں˺�������OpenCL�豸��ִ�С�
kernel���η����Ժ��������η�__attribute__���ʹ�ã���Ҫ��������Ϸ�ʽ��

//��ʾ�������ں����ڴ����������͵Ĵ�С
__kernel __attribute__((vec_type_hint(typen)))

//��ʾ��������ǰʹ�ù�����Ĵ�С�Ƕ���
__kernel __attribute__((work_group_size_hint(16, 16, 1)))

// ָ������ʹ�õĹ������С��local_work_size�Ĵ�С
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
*/

/*
CL����֮���ú���


*/
#include "pch.h"
// OpenCL includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/opencl.h>
#include <CL/cl.h>
#pragma comment (lib, "OpenCL.lib")

// OpenCL kernel. Each work item takes care of one element of c
const char* kernelSource1 = "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
"__kernel void vecAdd(  __global double *a,                       \n" \
"                       __global double *b,                       \n" \
"                       __global double *c,                       \n" \
"                       const unsigned int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n)                                                 \n" \
"        c[id] = a[id] + b[id];                                  \n" \
"}                                                               \n" \
"\n";

// OpenCL kernel. many workGroups compute n iterations
//...��ʾ�ɱ������ͨ�����滻����__VA_ARGS__�滻��...
#define KERNEL(...) #__VA_ARGS__
const char* kernelSource3 = KERNEL(
    __kernel void Pi(__global float* workGroupBuffer, // 0..NumWorkGroups-1 
        __local float* insideWorkGroup,  // 0..workGroupSize-1 
        const uint n,        // Total iterations 
        const uint chunk)        // Chunk size 
{
    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);

    const float step = (1.0 / (float)n);
    float partial_sum = 0.0;

    // Each work-item computes chunk iterations 
    for (uint i = gid * chunk; i < (gid * chunk) + chunk; i++) {
        float x = step * ((float)i - 0.5);
        partial_sum += 4.0 / (1.0 + x * x);
    }

    // Each work-item stores its partial sum in the workgroup array 
    insideWorkGroup[lid] = partial_sum;

    // Synchronize all threads within the workgroup 
    barrier(CLK_LOCAL_MEM_FENCE);

    float local_pi = 0;

    // Only work-item 0 of each workgroup perform the reduction 
    // of that workgroup 
    if (lid == 0) {
        const uint length = lid + get_local_size(0);
        for (uint i = lid; i < length; i++) {
            local_pi += insideWorkGroup[i];
        }
        // It store the workgroup sum 
        // Final reduction, between block, is done out by CPU 
        workGroupBuffer[get_group_id(0)] = local_pi;
    }
}
);

//typedef double* double2;
//typedef struct {
//    double2 m_triangle[3];
//} Trigon2d;
//
//void add_triangle(
//    const Trigon2d* A,
//    const Trigon2d* B,
//    Trigon2d* C)
//{
//    int id;
//    for (int i = 0; i < 3; i++)
//    {
//        C[id].m_triangle[i] = A[id].m_triangle[i] + B[id].m_triangle[i];
//    }
//}

typedef struct {
    Eigen::Vector2d m_triangle[3];
} Trigon2d;

const int N = 10; //1024 // �����С
const size_t size = N * N * sizeof(float);
int main1()
{
    // ��ʼ���������
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N];
    for (int i = 0; i < N * N; i++)
    {
        A[i] = 1.0f * i;
        B[i] = 1.0f * i;
    }
    Trigon2d* trisA = new Trigon2d[N * N];
    Trigon2d* trisB = new Trigon2d[N * N];
    Trigon2d* trisC = new Trigon2d[N * N];
    for (int i = 0; i < N * N; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            trisA[i].m_triangle[j] = Eigen::Vector2d{ i,i };
            trisB[i].m_triangle[j] = Eigen::Vector2d{ i,i };
        }
    }

    // ��ʼ��OpenCL����
    cl_platform_id platform;
    cl_int st_p = clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    cl_int st_d = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    //����openCL�����ģ�������������ִ��
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    //����ִ��������У����ڿ���ִ��
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    // ����OpenCL�������
    const char* source = cl_ReadString("test_calcl_3.cl");
    cl_int err;
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    //cl_kernel kernel = clCreateKernel(program, "add_matrices", &err);
    cl_kernel kernel = clCreateKernel(program, "add_triangle", &err);

    // ����OpenCL�ڴ滺����
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, NULL);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, NULL);
    // ���������ݴ��䵽OpenCL������
    clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, size, trisA, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, size, trisB, 0, NULL, NULL);
    // ����OpenCL�ں˲���
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

    // �����ں�
    size_t globalWorkSize[2] = { N * N, N * N };
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);

    // ��ȡ�������
    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, size, C, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, size, trisC, 0, NULL, NULL);

    // ����OpenCL��Դ
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // ��ӡ���
    std::vector<float> addResult(N * N);
    for (int i = 0; i < N * N; i++)
        addResult[i] = C[i];
    //std::cout << "Result: " << A[0] << std::endl;
    delete[] A;
    delete[] B;
    return 0;
}

int main2()
{
    int i = 0;
    size_t globalSize, localSize;
    cl_int err;
    double sum = 0;

    // Length of vectors
    int n = 100;// 000;
    // Host input vectors
    double* h_a;
    double* h_b;
    // Host output vector
    double* h_c;
    // Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_c;

    cl_platform_id platform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel

    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(double);
    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);

    // Initialize vectors on host
    for (i = 0; i < n; i++) 
    {
        h_a[i] = sinf(i) * sinf(i);
        h_b[i] = cosf(i) * cosf(i);
    }

     // Number of work items in each local work group
    localSize = 64;

    // Number of total work items - localSize must be devisor
    globalSize = ceil(n / (float)localSize) * localSize;

    // Bind to platform
    err = clGetPlatformIDs(1, &platform, NULL);
    // Get ID for the device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    // Create a context  
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    // Create a command queue 
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource1, NULL, &err);

    // Build the program executable 
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "vecAdd", &err);

    // Create the input and output arrays in device memory for our calculation
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, bytes, h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, bytes, h_b, 0, NULL, NULL);
    // Set the arguments to our compute kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);
    // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
    // Read the results from the device
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);

    //Sum up vector c and print result divided by n, this should equal 1 within error
    std::vector<float> addResult(n);
    for (int i = 0; i < n; i++)
        addResult[i] = h_c[i];

    // release OpenCL resources
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    //release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

//��ȡgaussian�ļ�
int main3()
{
    std::string source_name = "gaussian.cl";
    std::string program_source = ClReadString(source_name);
    char* cl_str = (char*)program_source.c_str();
    //program = clCreateProgramWithSource(context, 1, (const char**)&cl_str, NULL, NULL);

    int i = 0;
    float pi;
    float* pi_partial;
    size_t maxWorkGroupSize;
    cl_int err;
    cl_mem memObjects;
    int niter, chunks, workGroups;
    size_t globalWorkSize;
    size_t localWorkSize;

    cl_platform_id platform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel

    niter = 262144;
    chunks = 64;
    err = clGetPlatformIDs(1, &platform, NULL);
    // Get ID for the device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
    workGroups = ceil((float)(niter / maxWorkGroupSize / chunks));
    pi_partial = (float*)malloc(sizeof(float) * workGroups);

    // Create a context  
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    // Create a command queue 
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, &kernelSource3, NULL, &err);
    // Build the program executable 
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    localWorkSize = maxWorkGroupSize;
    globalWorkSize = niter / chunks;

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "Pi", &err);
    // Create the input and output arrays in device memory for our calculation
    memObjects = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * workGroups, NULL, &err);

    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects);
    err = clSetKernelArg(kernel, 1, sizeof(float) * maxWorkGroupSize, NULL);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &niter);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &chunks);
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    clFinish(queue);
    err = clEnqueueReadBuffer(queue, memObjects, CL_TRUE, 0, sizeof(float) * workGroups, pi_partial, 0, NULL, NULL);
    pi = 0;
    for (i = 0; i < workGroups; i++)
        pi += pi_partial[i];
    pi *= (1.0 / (float)niter);
    printf("final result: %f\n", pi);

    // release OpenCL resources
    clReleaseMemObject(memObjects);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    //release host memory
    free(pi_partial);
    return 0;
}

class OCL
{
public:
    cl_platform_id platform;          // OpenCL ƽ̨
    cl_device_id device_id;           // �豸��ID
    cl_context context;               // ������
    cl_command_queue queue;           // �������
    cl_program program;               // ����
    cl_kernel kernel;                 // �˺���
};

//��ȡ�豸��Ϣ
int main4()
{
    cl_uint numPlatforms = 0;
    cl_platform_id* platforms = nullptr;
    cl_uint numDevices = 0;
    cl_device_id* devices = nullptr;
    cl_int result; //status

    // ��ȡƽ̨����
    result = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (result != CL_SUCCESS) {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return -1;
    }

    // ��ȡƽ̨ID�б�
    platforms = new cl_platform_id[numPlatforms];
    result = clGetPlatformIDs(numPlatforms, platforms, nullptr);
    if (result != CL_SUCCESS) {
        std::cerr << "Failed to get platform IDs." << std::endl;
        delete[] platforms;
        return -1;
    }

    // ����ÿ��ƽ̨������֧�ֵ��豸
    for (cl_uint i = 0; i < numPlatforms; i++) //2(nvidia+intel)
    {
        // ��ȡ�豸����
        result = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        if (result != CL_SUCCESS) {
            std::cerr << "Failed to get device IDs for platform " << i << "." << std::endl;
            continue;
        }
        // ��ȡ�豸ID�б�
        devices = new cl_device_id[numDevices];
        result = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, numDevices, devices, nullptr);
        if (result != CL_SUCCESS || numDevices == 0)
            continue;
        std::cout << "Found " << numDevices << " GPU devices supporting OpenCL." << std::endl;
        // ������Ի�ȡ������豸��Ϣ���������ơ��ͺŵ�
        for (cl_uint j = 0; j < numDevices; j++)
        {
            // ��ȡ�豸�ͺţ�����еĻ�����ȡ�����豸��Ӧ���Ƿ��ṩ��
            size_t vendorStringSize = 0;
            result = clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 0, nullptr, &vendorStringSize);
            if (result != CL_SUCCESS) {
                std::cerr << "Failed to get vendor string size." << std::endl;
                continue;
            }
            char* vendorString = new char[vendorStringSize];
            result = clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, vendorStringSize, vendorString, nullptr);
            if (result == CL_SUCCESS)
                std::cout << "Device Vendor: " << vendorString << std::endl;
            char* value;
            size_t      valueSize;
            size_t      maxWorkItemPerGroup;
            cl_uint     maxComputeUnits = 0;
            cl_ulong    maxGlobalMemSize = 0;
            cl_ulong    maxConstantBufferSize = 0;
            cl_ulong    maxLocalMemSize = 0;

            ///print the device name
            clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[0], CL_DEVICE_NAME, valueSize, value, NULL);
            printf("Device Name: %s\n", value);
            free(value);

            /// print parallel compute units(CU)
            clGetDeviceInfo(devices[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            printf("Parallel compute units: %u\n", maxComputeUnits);
            ///maxWorkItemPerGroup
            clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkItemPerGroup), &maxWorkItemPerGroup, NULL);
            printf("maxWorkItemPerGroup: %zd\n", maxWorkItemPerGroup);
            /// print maxGlobalMemSize
            clGetDeviceInfo(devices[0], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(maxGlobalMemSize), &maxGlobalMemSize, NULL);
            printf("maxGlobalMemSize: %lu(MB)\n", maxGlobalMemSize / 1024 / 1024);
            /// print maxConstantBufferSize
            clGetDeviceInfo(devices[0], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(maxConstantBufferSize), &maxConstantBufferSize, NULL);
            printf("maxConstantBufferSize: %lu(KB)\n", maxConstantBufferSize / 1024);
            /// print maxLocalMemSize
            clGetDeviceInfo(devices[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(maxLocalMemSize), &maxLocalMemSize, NULL);
            printf("maxLocalMemSize: %lu(KB)\n", maxLocalMemSize / 1024);
            delete[] vendorString;
        }
        delete[] devices;
    }
    delete[] platforms;
    return 0;
}




static int _enrol = []()
    {
        main1();
        main2();
        //main4();
        return 0;
    }();