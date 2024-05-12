#pragma once
#ifdef CAL_DLLEXPORT_DEFINE
#define DLLEXPORT_CAL __declspec(dllexport)
#else
#define DLLEXPORT_CAL __declspec(dllimport)
#endif

namespace ocl
{
    DLLEXPORT_CAL int calculateFrontJudgeOfTrigon(std::vector<eigen::TrigonPart>& trigonVct, double toleDist, double toleAngle, double toleFixed);
}
