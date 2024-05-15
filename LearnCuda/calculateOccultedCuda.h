#pragma once

namespace cuda
{
    enum class OcclusionState :int //means cover
    {
        EXPOSED = 0,
        HIDDEN,
        SHIELDED, //shielded by other triangle
        COPLANAR, //COPLANAR with other-triangle
        INTERSECT, //ignore
        OCCLUSION, //shielded+intersect
        DEGENERACY, // become segment
        UNKNOWN,
    };

    enum class FrontState :int
    {
        // state of 3d trigon, all 2d projection penetration
        COPLANAR = 0, //and intersect
        A_FRONTOF,
        B_FRONTOF,
        INTERSECT, //3d intersect
        UNKNOWN,
    };

    typedef struct {
        Eigen::Vector2d data[3];
    } Triangle2d;

    typedef struct {
        Eigen::Vector3d data[3];
    } Triangle3d;

    //typedef struct {
    //    Eigen::Vector2d data[2];
    //} Segment2d;

    struct TrigonPart
    {
        int m_index;
        OcclusionState m_visible = OcclusionState::EXPOSED;
        Eigen::AlignedBox3d m_box3d;
        Eigen::AlignedBox2d m_box2d;
        Eigen::Vector3d m_normal; //normal of m_triangle3d, always upward
        Triangle3d m_triangle3d;
        Triangle2d m_triangle2d;
        size_t m_occ_size;
        int* m_occ_ptr;
    };

    int calculateFrontJudgeOfTrigon(std::vector<TrigonPart>& trigonVct, double toleDist, double toleAngle, double toleFixed);

}