#include "pch.h"

class OCLPlatformInfo
{
    class OCLDeviceInfo
    {
    public:
        cl_uint numDevices = 0;
        cl_device_id* devices = nullptr;
    };

    cl_platform_id id;
    std::string profile;
    std::string version;
    std::string name;
    std::string vendor;
    std::string extentsion;

public:
    cl_platform_id platform;          // OpenCL平台
    cl_device_id device_id;           // 设备的ID
    cl_context context;               // 上下文
    cl_command_queue queue;           // 命令队列
    cl_program program;               // 程序
    cl_kernel kernel;                 // 核函数
};

//https://zhuanlan.zhihu.com/p/451101452

void PrintPlatformMsg(cl_platform_id* platform, cl_platform_info platform_info, const char* platform_msg)
{
    size_t size;
    int err_num;
    // 1. 第一步通过size获取打印字符串长度
    err_num = clGetPlatformInfo(*platform, platform_info, 0, NULL, &size);
    char* result_string = (char*)malloc(size);
    // 2. 第二步获取平台信息到result_string 
    err_num = clGetPlatformInfo(*platform, platform_info, size, result_string, NULL);
    printf("%s=%s\n", platform_msg, result_string);
    free(result_string);
    result_string = NULL;
}


void getPlatformInfo()
{
    //法获取平台及相关信息
    cl_int err_num;
    cl_uint num_platform;
    cl_platform_id* platform_list;
    // 1. 第一次调用获取平台数
    err_num = clGetPlatformIDs(0, NULL, &num_platform);
    printf("num_platform=%d\n", num_platform);
    platform_list = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platform);
    // 2. 第二次调用获取平台对象数组
    err_num = clGetPlatformIDs(num_platform, platform_list, NULL);
    printf("err_num = %d\n", err_num);
    // 打印平台信息
    PrintPlatformMsg(&platform_list[0], CL_PLATFORM_PROFILE, "Platform Profile");
    PrintPlatformMsg(&platform_list[0], CL_PLATFORM_VERSION, "Platform Version");
    PrintPlatformMsg(&platform_list[0], CL_PLATFORM_NAME, "Platform Name");
    PrintPlatformMsg(&platform_list[0], CL_PLATFORM_VENDOR, "Platform Vendor");

    //获取设备及设备参数
    cl_uint num_device;
    cl_device_id device;
    // 1. 获取平台GPU类型OpenCL设备的数量
    err_num = clGetDeviceIDs(platform_list[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_device);
    printf("GPU num_device=%d\n", num_device);
    // 2. 获取一个GPU类型的OpenCL设备
    err_num = clGetDeviceIDs(platform_list[0], CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // 对于cl_uint cl_ulong等返回类型参数只需要一步查询
    cl_uint max_compute_units;
    // 获取并打印OpenCL设备的并行计算单元数量
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
        &max_compute_units, NULL);
    printf("max_compute_units=%d\n", max_compute_units);

    cl_ulong global_mem_size;
    // 获取并打印OpenCL设备的全局内存大小
    err_num = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
        &global_mem_size, NULL);
    printf("global_mem_size=%ld\n", global_mem_size);

    size_t* p_max_work_item_sizes = NULL;
    size_t size;
    // CL_DEVICE_MAX_WORK_ITEM_SIZES表示work_group每个维度的最大工作项数目
    // 1. 返回类型是size_t[]，首先查询返回信息的大小
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL, &size);
    p_max_work_item_sizes = (size_t*)malloc(size);
    // 2. 申请空间后查询结果并打印
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, size, p_max_work_item_sizes, NULL);
    for (size_t i = 0; i < size / sizeof(size_t); i++)
    {
        printf("max_work_item_size_of_work_group_dim %zu=%zu\n", i, p_max_work_item_sizes[i]);
    }
}