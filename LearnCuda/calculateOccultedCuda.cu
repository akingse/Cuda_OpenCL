#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <Eigen/Dense>
#include "pch.h"
#include "clashTypeDefine.h"
#include "calculateOccultedCuda.h"

using namespace cuda;
using namespace Eigen;

__device__ bool isPointInTriangle(const Eigen::Vector2d& point, const Triangle2d& trigon, double tolerance) // 2D
{
	//tole>0 less judge, tole<0 more judge
	Eigen::Vector2d trigonR[3]; // is origin inside trigonR
	double precision = 0;
	for (int i = 0; i < 3; ++i)
	{
		trigonR[i] = trigon.data[i] - point;
		precision = max(max(fabs(trigonR[i][0]), fabs(trigonR[i][1])), precision);
	}
	precision = precision * tolerance; // assert(precision > tolerance);
	if (-precision < min(min(trigonR[0][0], trigonR[1][0]), trigonR[2][0]) ||
		precision > max(max(trigonR[0][0], trigonR[1][0]), trigonR[2][0]) ||
		-precision < min(min(trigonR[0][1], trigonR[1][1]), trigonR[2][1]) ||
		precision > max(max(trigonR[0][1], trigonR[1][1]), trigonR[2][1])) //box judge
	{
		return false;
	}
	// been legal judge, no zero vector
	Eigen::Vector2d v0 = (trigon.data[1] - trigon.data[0]).normalized();
	Eigen::Vector2d v1 = (trigon.data[2] - trigon.data[1]).normalized();
	Eigen::Vector2d v2 = (trigon.data[0] - trigon.data[2]).normalized();
	// (p1-p0).cross(p2-p1)
	double axisz = v0[0] * v1[1] - v0[1] * v1[0];
	axisz = (0.0 < axisz) ? 1.0 : -1.0;
	return
		precision <= axisz * (v0[1] * trigonR[0][0] - v0[0] * trigonR[0][1]) &&
		precision <= axisz * (v1[1] * trigonR[1][0] - v1[0] * trigonR[1][1]) &&
		precision <= axisz * (v2[1] * trigonR[2][0] - v2[0] * trigonR[2][1]); // = decide whether include point on edge
}

__device__ bool isTriangleInsideTriangleOther(const TrigonPart& triA, const TrigonPart& other, const double tolerance = 0)
{
	if (!other.m_box2d.contains(triA.m_box2d)) //without tolerance
		return false;
	for (int i = 0; i < 3; i++)
	{
		if (!isPointInTriangle(triA.m_triangle2d.data[i], other.m_triangle2d, -tolerance))
			return false;
	}
	return true;
}

__device__ bool isTwoSegmentsIntersect(const Vector2d segmA[2], const Vector2d segmB[2], double tolerance)
{
	//tole<0 less judge, tole>0 more judge
	Vector2d vecA = segmA[1] - segmA[0];
	Vector2d vecB = segmB[1] - segmB[0];
	double precision = 0;
	precision = max(max(fabs(vecA[0]), fabs(vecA[1])), precision);
	precision = max(max(fabs(vecB[0]), fabs(vecB[1])), precision);
	precision = precision * tolerance;// assert(precision > tolerance);
	if (precision < min(segmB[0][0], segmB[1][0]) - max(segmA[0][0], segmA[1][0]) ||
		precision < min(segmA[0][0], segmA[1][0]) - max(segmB[0][0], segmB[1][0]) ||
		precision < min(segmB[0][1], segmB[1][1]) - max(segmA[0][1], segmA[1][1]) ||
		precision < min(segmA[0][1], segmA[1][1]) - max(segmB[0][1], segmB[1][1]))
		return false;
	Vector2d AB_0 = segmB[0] - segmA[0];
	Vector2d AB_1 = segmB[1] - segmA[0];
	Vector2d BA_0 = segmA[0] - segmB[0];
	Vector2d BA_1 = segmA[1] - segmB[0];
	return //double straddle test, cross2d opposite direction
		(AB_0[0] * vecA[1] - AB_0[1] * vecA[0]) * (AB_1[0] * vecA[1] - AB_1[1] * vecA[0]) <= precision &&
		(BA_0[0] * vecB[1] - BA_0[1] * vecB[0]) * (BA_1[0] * vecB[1] - BA_1[1] * vecB[0]) <= precision; //not support both near zero
}

__device__ bool isTwoTrianglesIntersectSAT(const Triangle2d& triA, const Triangle2d& triB)
{
	Eigen::Vector2d edgesAB[6] = {
		triA.data[1] - triA.data[0],
		triA.data[2] - triA.data[1],
		triA.data[0] - triA.data[2],
		triB.data[1] - triB.data[0],
		triB.data[2] - triB.data[1],
		triB.data[0] - triB.data[2],
	};
	for (int i = 0; i < 6; i++)
		edgesAB[i] = { -edgesAB[i][1], edgesAB[i][0] }; //rotz(pi/2)
	double minA, maxA, minB, maxB, projection;
	for (int i = 0; i < 6; i++)
	{
		//if (axis.isZero()) //degeneracy triangle, regard as not shield
		//	continue;
		minA = DBL_MAX;
		maxA = -DBL_MAX;
		minB = DBL_MAX;
		maxB = -DBL_MAX;
		for (int j = 0; j < 3; j++)
		{
			projection = edgesAB[i].dot(triA.data[j]);
			minA = min(minA, projection);
			maxA = max(maxA, projection);
			projection = edgesAB[i].dot(triB.data[j]);
			minB = min(minB, projection);
			maxB = max(maxB, projection);
		}
		if (maxA <= minB || maxB <= minA) //contact, regard as not shield
		{
			return false;
		}
	}
	return true;
}

__device__ FrontState isFrontJudgeOfTrigon(const cuda::TrigonPart& trigonA, const cuda::TrigonPart& trigonB,
	const double toleDist, const double toleAngle, const double toleFixed) //tolerance>0
{
	if (trigonA.m_normal.cross(trigonB.m_normal).norm() <= toleAngle) // normal.isZero(tolerance)
	{
		Eigen::Vector3d pd = trigonB.m_triangle3d.data[0] - trigonA.m_triangle3d.data[0];
		double precision = max(max(fabs(pd[0]), fabs(pd[1])), fabs(pd[2]));
		double d = fabs(trigonA.m_normal.dot(pd));
		if (precision < toleDist || d < precision * toleDist) // support zero pd
		{
			return FrontState::COPLANAR;
		}
	}
	for (int k = 0; k < 3; ++k)
	{
		if (isPointInTriangle(trigonB.m_triangle2d.data[k], trigonA.m_triangle2d, -toleDist)) // point in triA
		{
			Eigen::Vector3d pd = trigonB.m_triangle3d.data[k] - trigonA.m_triangle3d.data[0];
			double precision = max(max(fabs(pd[0]), fabs(pd[1])), fabs(pd[2]));
			if (toleFixed < precision) //pd approx zero vector
			{
				double d = trigonA.m_normal.dot(pd); //normal is upwards and normalized
				if (precision * toleFixed < fabs(d)) // point not on plane
					return d < 0.0 ? FrontState::A_FRONTOF : FrontState::B_FRONTOF;
			}
		}
		if (isPointInTriangle(trigonA.m_triangle2d.data[k], trigonB.m_triangle2d, -toleDist)) // point in triB
		{
			Eigen::Vector3d pd = trigonA.m_triangle3d.data[k] - trigonB.m_triangle3d.data[0];
			double precision = max(max(fabs(pd[0]), fabs(pd[1])), fabs(pd[2]));
			if (toleFixed < precision) //pd approx zero vector
			{
				double d = trigonB.m_normal.dot(pd); //normal is upwards and normalized
				if (precision* toleFixed < fabs(d)) // point not on plane
					return d > 0.0 ? FrontState::A_FRONTOF : FrontState::B_FRONTOF;
			}
		}
	}
	//hexagram intersect
	Eigen::Vector2d segm2A[2];
	Eigen::Vector2d segm2B[2];
	Eigen::Vector3d vect3A, vect3B;
	for (int i = 0; i < 3; ++i) // using vector substract, so not relative coordinate
	{
		int i1 = (i + 1) % 3;
		segm2A[0] = trigonA.m_triangle2d.data[i];
		segm2A[1] = trigonA.m_triangle2d.data[i1];
		vect3A = trigonA.m_triangle3d.data[i1] - trigonA.m_triangle3d.data[i];
		for (int j = 0; j < 3; ++j)
		{
			int j1 = (j + 1) % 3;
			segm2B[0] = trigonB.m_triangle2d.data[j];
			segm2B[1] = trigonB.m_triangle2d.data[j1];
			vect3B = trigonB.m_triangle3d.data[j1] - trigonB.m_triangle3d.data[j];
			if (!isTwoSegmentsIntersect(segm2A, segm2B, toleDist))
				continue;
			double precision = 0;
			for (int k = 0; k < 3; ++k)
				precision = max(max(fabs(vect3A[k]), fabs(vect3B[k])), precision);
			Eigen::Vector3d normal = vect3A.cross(vect3B);
			if (normal.isZero(precision * toleDist))
				continue; //means collieanr in 3d
			Eigen::Vector3d pd = trigonA.m_triangle3d.data[i] - trigonB.m_triangle3d.data[j];
			precision = max(max(fabs(pd[0]), fabs(pd[1])), fabs(pd[2]));
			if (precision < toleFixed) //pd approx zero vector
				continue;
			normal.normalize();
			double d = 0.0 < normal.z() ? 1.0 : -1.0;
			d *= normal.dot(pd); //the projection difference of two segment
			if (precision * toleFixed < fabs(d))
				return 0.0 < d ? FrontState::A_FRONTOF : FrontState::B_FRONTOF;
		}
	}
	return FrontState::UNKNOWN; //error, not intersect
}

__global__ void kernel_calculateFrontJudgeOfTrigon(cuda::TrigonPart* trigonVct, double toleDist, double toleAngle, double toleFixed)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	TrigonPart& trigonA = trigonVct[i];
    for (size_t j = 0; j < trigonA.m_occ_size; ++j)
    {
        const TrigonPart& trigonB = trigonVct[trigonA.m_occ_ptr[j]];
        if (trigonB.m_visible == OcclusionState::HIDDEN ||
            trigonB.m_box3d.max().z() < trigonA.m_box3d.min().z() ||
            !isTwoTrianglesIntersectSAT(trigonA.m_triangle2d, trigonB.m_triangle2d))
        {
			trigonA.m_occ_ptr[j] = -1; //means not shield
            continue;
        }
        if (FrontState::B_FRONTOF == isFrontJudgeOfTrigon(trigonA, trigonB, toleDist, toleAngle, toleFixed))
        {
            if (isTriangleInsideTriangleOther(trigonA, trigonB, toleDist)) //only judge and change trigonA
            {
                trigonA.m_visible = OcclusionState::HIDDEN;
				// record this, reset all other
                for (size_t k = j + 1; k < trigonA.m_occ_size; ++k)
					trigonA.m_occ_ptr[k] = -1;
                break;
            }
            trigonA.m_visible = OcclusionState::SHIELDED;
            //trigonA.m_shielded.push_back(trigonB.m_index);
			continue;
		}
		// else
		trigonA.m_occ_ptr[j] = -1;
    }
}

int cuda::calculateFrontJudgeOfTrigon(std::vector<eigen::TrigonPart>& trigonVct, double toleDist, double toleAngle, double toleFixed)
{
	cuda::TrigonPart* device;
    size_t size = trigonVct.size() * sizeof(TrigonPart) + 4 * sizeof(double);
	std::vector<cuda::TrigonPart> trigonCuda(trigonVct.size());
	for (int i = 0; i < trigonVct.size(); i++)
	{
		cuda::TrigonPart& trigon = trigonCuda[i];
		trigon.m_index = trigonVct[i].m_index;
        trigon.m_visible = trigonVct[i].m_visible;
        trigon.m_box3d = trigonVct[i].m_box3d;
        trigon.m_box2d = trigonVct[i].m_box2d;
        trigon.m_normal = trigonVct[i].m_normal;
        trigon.m_triangle3d = { trigonVct[i].m_triangle3d[0], trigonVct[i].m_triangle3d[1], trigonVct[i].m_triangle3d[2] };
        trigon.m_triangle2d = { trigonVct[i].m_triangle2d[0], trigonVct[i].m_triangle2d[1], trigonVct[i].m_triangle2d[2] };
		trigon.m_occ_size = trigonVct[i].m_preInter.size();
		trigon.m_occ_ptr = trigonVct[i].m_preInter.data();
		size += sizeof(int) * trigon.m_occ_size;
		//trigonVct[i].m_visible = OcclusionState::USING_CUDA;
	}
	cudaMallocManaged(&device, size); //shared memory
	//for (int i = 0; i < trigonVct.size(); ++i)
	//{
	//	//cudaMallocManaged(&(devData[i].variableLengthData), hostData[i].variableLengthData.size() * sizeof(int));
	//	//cudaMemcpy(devData[i].variableLengthData, 
	//	//	hostData[i].variableLengthData.data(),
	//	//	hostData[i].variableLengthData.size() * sizeof(int), 
	//	//	cudaMemcpyHostToDevice);
	//}
	//cudaMemcpy(device, trigonVct.data(), size, cudaMemcpyHostToDevice);
	
	//copy in
	for (int i = 0; i < trigonCuda.size(); i++)
	{
		device[i] = trigonCuda[i];
		size_t count = trigonCuda[i].m_occ_size * sizeof(int);
		if (count == 0)
			continue;
		cudaMalloc((void**)&device[i].m_occ_ptr, count);
		cudaMemcpy(device[i].m_occ_ptr, trigonCuda[i].m_occ_ptr, count, cudaMemcpyHostToDevice);
	}
	//kernelName<<< grid_size, block_size >>>( ... );
	kernel_calculateFrontJudgeOfTrigon << <1, trigonVct.size() >> > (device, toleDist, toleAngle, toleFixed);
	cudaDeviceSynchronize();
	//cudaMemcpy(trigonVct.data(), device, size, cudaMemcpyDeviceToHost);
	//copy out
	//size_t count = 0;
	for (int i = 0; i < trigonVct.size(); i++)
	{
		//count += sizeof(TrigonPart);
		size_t count = trigonCuda[i].m_occ_size * sizeof(int);
		if (count == 0)
			continue;
		cudaMemcpy(trigonCuda[i].m_occ_ptr, device[i].m_occ_ptr, count, cudaMemcpyDeviceToHost);
		//bool exposed = true; // specially assign EXPOSED
		for (int j = 0; j < trigonVct[i].m_preInter.size(); j++)
		{
			if (trigonVct[i].m_preInter[j] != -1)
				trigonVct[i].m_shielded.push_back(trigonVct[i].m_preInter[j]);
		}
		if (!trigonVct[i].m_shielded.empty())
			trigonVct[i].m_visible = OcclusionState::SHIELDED;
		//cudaFree(&device[i]);
	}
	cudaFree(device);
    return 0;
}

