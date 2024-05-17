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

//#define USING_POINTER
static void test_triangle()
{
	const int N = 1;
#ifdef USING_POINTER
	Triangle2d* triA, *triB;
	cudaMallocManaged(&triA, N * 6 * sizeof(Triangle2d));
	cudaMallocManaged(&triB, N * 6 * sizeof(Triangle2d));
	for (int i = 0; i < N; i++) 
	{
		triA[i] = Triangle2d{
				Vector2d{0,0},
				Vector2d{2,0},
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
	//kernal
	isTwoTrianglesIntersectSAT_P << <1, N >> > (isinter, triA, triB);
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	//bool* res;
	bool res[N]; //cpu mem
	cudaError_t cudaStatus = cudaMemcpy(res, isinter, N * sizeof(bool), cudaMemcpyDeviceToHost);
#else
	std::vector<Triangle2d> triA(N), triB(N);
	for (int i = 0; i < N; i++) 
	{
		triA[i] = Triangle2d{
				Vector2d{0,0},
				Vector2d{2,0},
				Vector2d{0,1},
		};
		triB[i] = Triangle2d{
				Vector2d{1,0},
				Vector2d{2,0},
				Vector2d{2,1},
		};
	}
	Triangle2d* dev_a; //device var
	Triangle2d* dev_b;
	bool* dev_c;
	cudaMallocManaged(&dev_a, N * sizeof(Triangle2d));
	cudaMallocManaged(&dev_b, N * sizeof(Triangle2d));
	cudaMallocManaged(&dev_c, N * sizeof(bool));
	cudaMemcpy(dev_a, triA.data(), N * sizeof(Triangle2d), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, triB.data(), N * sizeof(Triangle2d), cudaMemcpyHostToDevice);
	isTwoTrianglesIntersectSAT_P << <1, N >> > (dev_c, dev_a, dev_b);
	cudaDeviceSynchronize();
	//copy out from gpu
	bool h_isinter[N];
	cudaMemcpy(h_isinter, dev_c, N * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
#endif
	return;
}

struct MyClass 
{
	//std::vector<int> data;
	size_t size;
	int* ptr;
	std::vector<int> to_vector() const
	{
		std::vector<int> data(size);
		for (int i = 0; i < size; i++)
			data[i] = ptr[i];
		return data;
	}
};

__global__ void variable_length(MyClass* varray)
{
	int i = threadIdx.x;
	MyClass& arr = varray[i];
	int k;
	for (int j = 0; j < arr.size; j++)
		k = arr.ptr[i];

	return;
}

static void test_vector()
{
	size_t size = 10;
	std::vector<MyClass> hostData(size);
	for (int i = 0; i < size; i++)
	{
		//std::vector<int> temp;
		int* temp = new int[i];
		for (int j = 0; j < i; j++)
			temp[j] = i;
		//hostData[i].data = temp;
		hostData[i].size = i;
		hostData[i].ptr = temp;
	}

	MyClass* devData;
	size_t c_size = size * sizeof(MyClass);
	for (int i = 0; i < size; i++)
		c_size += hostData[i].size * sizeof(int);
	cudaMallocManaged(&devData, c_size);

	for (int i = 0; i < size; ++i) 
	{
		//cudaMallocManaged(&devData, hostData[i].size * sizeof(int));
		cudaMalloc((void**)&devData[i], hostData[i].size * sizeof(int));
		cudaMemcpy(devData, hostData[i].ptr, hostData[i].size * sizeof(int), cudaMemcpyHostToDevice);
	}
	variable_length << <1, size >> > (devData);
	cudaDeviceSynchronize();
	//cudaMemcpy(trigonVct.data(), device, size, cudaMemcpyDeviceToHost);
	cudaFree(devData);
}

static int _enrol = []()
    {
        //test_triangle();
		//test_vector();
        return 0;
    }();
