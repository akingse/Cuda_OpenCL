#include"pch.h"
#include <CL/cl.h>

using namespace std;

//using OpenCL
int ocl::calculateFrontJudgeOfTrigon(std::vector<eigen::TrigonPart>& trigonVct, double toleDist, double toleAngle, double toleFixed)
{
    cl_uint numPlatforms = 0;
    cl_platform_id* platforms = nullptr;
    cl_uint numDevices = 0;
    cl_device_id* devices = nullptr;
    cl_int result = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (result != CL_SUCCESS) 
        return -1;
    platforms = new cl_platform_id[numPlatforms];
    result = clGetPlatformIDs(numPlatforms, platforms, nullptr);
    if (result != CL_SUCCESS) 
    {
        delete[] platforms;
        return -1;
    }




    return 0;
}
