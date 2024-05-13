           
__kernel void vecAdd(  __global double *a,                    
                       __global double *b,                    
                       __global double *c,                    
                       const unsigned int n)                   
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
}       

struct Trigon2d
{
    double2 m_triangle[3];
};
struct Trigon3d
{
    double3 m_triangle[3];
};

const double DBL_MAX = 1e308;

__kernel void isTwoTrianglesIntersectSAT(Trigon2d* triA, Trigon2d* triB, bool* isinter)
{
    std::array<Eigen::Vector2d, 6> edgesAB = {
    triA[1] - triA[0],
    triA[2] - triA[1],
    triA[0] - triA[2],
    triB[1] - triB[0],
    triB[2] - triB[1],
    triB[0] - triB[2], };
    for (auto& axis : edgesAB)
        axis = Vector2d(-axis[1], axis[0]); //rotz(pi/2)
    // Check for overlap along each axis
    double minA, maxA, minB, maxB, projection;
    for (int i = 0; i < 6; i++)
    {
        minA = DBL_MAX;
        maxA = -DBL_MAX;
        minB = DBL_MAX;
        maxB = -DBL_MAX;
        for (int j = 0; j < 6; j++)
        {
            projection = axis.dot(vertex - triA[0]);
            minA = min(minA, projection);
            maxA = max(maxA, projection);
        }
        for (const auto& vertex : triB)
        {
            projection = axis.dot(vertex - triA[0]);
            minB = min(minB, projection);
            maxB = max(maxB, projection);
        }
        if (maxA <= minB || maxB <= minA)
        {
            *isinter = false;
            return;
        }
    }
    *isinter = true;
    return;
}
