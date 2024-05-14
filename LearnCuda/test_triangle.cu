#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
//non-support array
#include <Eigen/Dense>
#include "calculateOccultedCuda.h"
//#include "clashTypeDefine.h" //using container vector
using namespace Eigen;
using namespace cuda;

__global__ void isTwoTrianglesIntersectSAT_P(bool* isinter, const Triangle2d* triA, const Triangle2d* triB)
{
	Eigen::Vector2d edgesAB[6] = {
		triA->data[1] - triA->data[0],
		triA->data[2] - triA->data[1],
		triA->data[0] - triA->data[2],
		triB->data[1] - triB->data[0],
		triB->data[2] - triB->data[1],
		triB->data[0] - triB->data[2],
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
			projection = edgesAB[i].dot(triA->data[j]);
			minA = min(minA, projection);
			maxA = max(maxA, projection);
			projection = edgesAB[i].dot(triB->data[j]);
			minB = min(minB, projection);
			maxB = max(maxB, projection);
		}
		if (maxA <= minB || maxB <= minA) //contact, regard as not shield
		{
			*isinter = false;
			printf("sat=false.");
			return;
		}
	}
	*isinter = true;
	printf("sat=true.");
}

static void test_triangle()
{
	const int N = 1;
	std::vector<Triangle2d> triA(N), triB(N);
	//Triangle2d* triA, *triB;
	Triangle2d* ptrA = triA.data();
	Triangle2d* ptrB = triB.data();
    cudaMallocManaged(&ptrA, N * 6 * sizeof(Triangle2d));
    cudaMallocManaged(&ptrB, N * 6 * sizeof(Triangle2d));
	for (int i = 0; i < N; i++) {
        triA[i] = Triangle2d{
                Vector2d{0,0},
                Vector2d{0.2,0},
                Vector2d{0,1},
        };
        triB[i] = Triangle2d{
                Vector2d{1,0},
                Vector2d{2,0},
                Vector2d{2,1},
        };
    }

	bool* isinter; //gpu mem
	cudaMallocManaged(&isinter, N * sizeof(bool));

	isTwoTrianglesIntersectSAT_P << <1, N >> > (isinter, triA.data(), triB.data());
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	//bool* res;
	bool res[N]; //cpu mem
	cudaError_t cudaStatus = cudaMemcpy(res, isinter, N * sizeof(bool), cudaMemcpyDeviceToHost);

	return;
}

static int _enrol = []()
    {
        test_triangle();
        return 0;
    }();
