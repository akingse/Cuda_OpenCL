#pragma once
#define CAL_DLLEXPORT_DEFINE
#ifdef CAL_DLLEXPORT_DEFINE
#define DLLEXPORT_CAL __declspec(dllexport)
#else
#define DLLEXPORT_CAL __declspec(dllimport)
#endif

#include<array>
#include<vector>
#include<map>
#include<set>
#include<Eigen/Dense>

#include "clashTypeDefine.h"
#include "calculateOccultedOCL.h"
