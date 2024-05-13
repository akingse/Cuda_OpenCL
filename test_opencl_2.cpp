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



