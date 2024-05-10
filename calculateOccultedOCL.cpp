#include"pch.h"
using namespace std;
using namespace eigen;

void ocl::calculateFrontJudgeOfTrigon(std::vector<eigen::TrigonPart>& trigonVct, double toleDist, double toleAngle, double toleFixed)
{
    for (int i = 0; i < trigonVct.size(); ++i)
    {
        TrigonPart& trigonA = trigonVct[i];
        for (const auto& j : trigonA.m_findInter)
        {
            const TrigonPart& trigonB = trigonVct[j];
            if (trigonB.m_visible == OcclusionState::HIDDEN)
            {
                continue; //means coverd by front trigon
            }
            if (trigonB.m_box3d.max().z() < trigonA.m_box3d.min().z())
            {
                continue;
            }
            if (!isTwoTrianglesIntersectSAT(trigonA.m_triangle2d, trigonB.m_triangle2d))
            {
                continue;
            }
            if (FrontState::B_FRONTOF == isFrontJudgeOfTrigon(trigonA, trigonB, toleDist, toleAngle, toleFixed))
            {
                if (isTriangleInsideTriangleOther(trigonA, trigonB, toleDist))
                {
                    trigonA.m_visible = OcclusionState::HIDDEN;
                    break;
                }
                trigonA.m_visible = OcclusionState::SHIELDED;
                trigonA.m_shielded.push_back(trigonB.m_index);
            }
        }
    }
    //return;
}
