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

//https://zhuanlan.zhihu.com/p/451101452

void PrintPlatformMsg(cl_platform_id* platform, cl_platform_info platform_info, const char* platform_msg)
{
    size_t size;
    int err_num;
    // 1. ��һ��ͨ��size��ȡ��ӡ�ַ�������
    err_num = clGetPlatformInfo(*platform, platform_info, 0, NULL, &size);
    char* result_string = (char*)malloc(size);
    // 2. �ڶ�����ȡƽ̨��Ϣ��result_string 
    err_num = clGetPlatformInfo(*platform, platform_info, size, result_string, NULL);
    printf("%s=%s\n", platform_msg, result_string);
    free(result_string);
    result_string = NULL;
}


void getPlatformInfo()
{
    //����ȡƽ̨�������Ϣ
    cl_int err_num;
    cl_uint num_platform;
    cl_platform_id* platform_list;
    // 1. ��һ�ε��û�ȡƽ̨��
    err_num = clGetPlatformIDs(0, NULL, &num_platform);
    printf("num_platform=%d\n", num_platform);
    platform_list = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platform);
    // 2. �ڶ��ε��û�ȡƽ̨��������
    err_num = clGetPlatformIDs(num_platform, platform_list, NULL);
    printf("err_num = %d\n", err_num);
    // ��ӡƽ̨��Ϣ
    PrintPlatformMsg(&platform_list[0], CL_PLATFORM_PROFILE, "Platform Profile");
    PrintPlatformMsg(&platform_list[0], CL_PLATFORM_VERSION, "Platform Version");
    PrintPlatformMsg(&platform_list[0], CL_PLATFORM_NAME, "Platform Name");
    PrintPlatformMsg(&platform_list[0], CL_PLATFORM_VENDOR, "Platform Vendor");

    //��ȡ�豸���豸����
    cl_uint num_device;
    cl_device_id device;
    // 1. ��ȡƽ̨GPU����OpenCL�豸������
    err_num = clGetDeviceIDs(platform_list[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_device);
    printf("GPU num_device=%d\n", num_device);
    // 2. ��ȡһ��GPU���͵�OpenCL�豸
    err_num = clGetDeviceIDs(platform_list[0], CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // ����cl_uint cl_ulong�ȷ������Ͳ���ֻ��Ҫһ����ѯ
    cl_uint max_compute_units;
    // ��ȡ����ӡOpenCL�豸�Ĳ��м��㵥Ԫ����
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
        &max_compute_units, NULL);
    printf("max_compute_units=%d\n", max_compute_units);

    cl_ulong global_mem_size;
    // ��ȡ����ӡOpenCL�豸��ȫ���ڴ��С
    err_num = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
        &global_mem_size, NULL);
    printf("global_mem_size=%ld\n", global_mem_size);

    size_t* p_max_work_item_sizes = NULL;
    size_t size;
    // CL_DEVICE_MAX_WORK_ITEM_SIZES��ʾwork_groupÿ��ά�ȵ����������Ŀ
    // 1. ����������size_t[]�����Ȳ�ѯ������Ϣ�Ĵ�С
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL, &size);
    p_max_work_item_sizes = (size_t*)malloc(size);
    // 2. ����ռ���ѯ�������ӡ
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, size, p_max_work_item_sizes, NULL);
    for (size_t i = 0; i < size / sizeof(size_t); i++)
    {
        printf("max_work_item_size_of_work_group_dim %zu=%zu\n", i, p_max_work_item_sizes[i]);
    }
}