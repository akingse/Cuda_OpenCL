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
    cl_platform_id platform;          // OpenCLƽ̨
    cl_device_id device_id;           // �豸��ID
    cl_context context;               // ������
    cl_command_queue queue;           // �������
    cl_program program;               // ����
    cl_kernel kernel;                 // �˺���
};



