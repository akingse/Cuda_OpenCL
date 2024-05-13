#ifdef FP64
#pragma OpenCL EXTENSION cl_khr_fp64 : enable
#endif
const double DBL_MAX = 1e308;

typedef struct {
    double3 trigon[3];
} Trigon3d;

__kernel void isTwoTrianglesIntersectSAT(Trigon3d* trisA, Trigon3d* trisB, bool* isinter)
{
    int k = get_global_id(0);
    Trigon3d triA = trisA[k];
    Trigon3d triB = trisB[k];
    //SAT
    double3 edgesA[3];
    edgesA[0] = triA.trigon[1] - triA.trigon[0];
    edgesA[1] = triA.trigon[2] - triA.trigon[1];
    edgesA[2] = triA.trigon[0] - triA.trigon[2];
    double3 edgesB[3];
    edgesB[0] = triB.trigon[1] - triB.trigon[0];
    edgesB[1] = triB.trigon[2] - triB.trigon[1];
    edgesB[2] = triB.trigon[0] - triB.trigon[2];
	double3 normalA = cross(edgesA[0], edgesA[1]);
	double3 normalB = cross(edgesB[0], edgesB[1]);
    double3 axes[2];
    axes[0] = normalA;
    axes[1] = normalB;
            //cross(normalA, edgesA[0]),
            //cross(normalA, edgesA[1]),
            //cross(normalA, edgesA[2]),
            //cross(normalB, edgesB[0]),
            //cross(normalB, edgesB[1]),
            //cross(normalB, edgesB[2]),
            //cross(edgesA[0],edgesB[0]),
            //cross(edgesA[0],edgesB[1]),
            //cross(edgesA[0],edgesB[2]),
            //cross(edgesA[1],edgesB[0]),
            //cross(edgesA[1],edgesB[1]),
            //cross(edgesA[1],edgesB[2]),
            //cross(edgesA[2],edgesB[0]),
            //cross(edgesA[2],edgesB[1]),
            //cross(edgesA[2],edgesB[2]) };
    // Check for overlap along each axis
    double minA, maxA, minB, maxB, projection;
    for (int i = 0; i < 2; i++)
    {
        minA = DBL_MAX;
        maxA = -DBL_MAX;
        minB = DBL_MAX;
        maxB = -DBL_MAX;
        for (int j = 0; j < 3; j++)
        {
            projection = dot(axes[i], triA.trigon[j]);
            minA = min(minA, projection);
            maxA = max(maxA, projection);
            projection = dot(axes[i], triB.trigon[j]);
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
