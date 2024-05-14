#include <vector>
#include <Eigen/Dense>
#include "calculateOccultedCuda.h"
//#include "clashTypeDefine.h" //using container vector



int calculateFrontJudgeOfTrigon()
{
    //for (const auto& j : preInter2d)
    //{
    //    const TrigonPart& trigonB = trigonVct[j];
    //    if (trigonB.m_visible == OcclusionState::HIDDEN)
    //    {
    //        continue; //means coverd by front trigon
    //    }
    //    if (trigonB.m_box3d.max().z() < trigonA.m_box3d.min().z())
    //    {
    //        continue;
    //    }
    //    if (!isTwoTrianglesIntersectSAT(trigonA.m_triangle2d, trigonB.m_triangle2d))
    //    {
    //        continue;
    //    }
    //    if (FrontState::B_FRONTOF == isFrontJudgeOfTrigon(trigonA, trigonB, toleDist, toleAngle, toleFixed))
    //    {
    //        if (isTriangleInsideTriangleOther(trigonA, trigonB, toleDist)) //only judge and change trigonA
    //        {
    //            trigonA.m_visible = OcclusionState::HIDDEN;
    //            break;
    //        }
    //        trigonA.m_visible = OcclusionState::SHIELDED;
    //        trigonA.m_shielded.push_back(trigonB.m_index);
    //    }
    //}

    return 0;
}
