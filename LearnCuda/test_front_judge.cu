#include <vector>
#include <Eigen/Dense>
#include "clashTypeDefine.h"
#include "calculateOccultedCuda.h"
using namespace Eigen;

//template<class T>
Eigen::AlignedBox3d getBox(const clash::Triangle3d& triangle)
{
	Eigen::AlignedBox3d box;
	for (int i = 0; i < 3; i++)
		box.extend(triangle[i]);
	return box;
}

Eigen::AlignedBox2d getBox(const clash::Triangle2d& triangle)
{
	Eigen::AlignedBox2d box;
	for (int i = 0; i < 3; i++)
		box.extend(triangle[i]);
	return box;
}

void test_cuda0()
{
	using namespace cuda;
	std::vector<int> m_preInterA = { 1,2 };
	std::vector<int> m_preInterB = { 0 };
	Triangle2d triA2 = {
		Vector2d{0,0},
		Vector2d{2,0},
		Vector2d{0,1},
	};
	Triangle3d triA3 = {
		Vector3d{0,0,0},
		Vector3d{2,0,1},
		Vector3d{0,1,0},
	};

	Triangle2d triB2 = {
		Vector2d{1,0},
		Vector2d{2,0},
		Vector2d{2,1},
	};
	Triangle3d triB3 = {
		Vector3d{1,0,0},
		Vector3d{2,0,0},
		Vector3d{2,1,0},
	};

	//TrigonPart trigonA = {
	//	0,
	//	OcclusionState::EXPOSED,
	//	getBox3d(triA3),
	//	getBox2d(triA2),
	//	Eigen::Vector3d(0,0,1),
	//	triA3,
	//	triA2,
	//	m_preInterA.size(),
	//	m_preInterA.data(),
	//};
	//TrigonPart trigonB = {
	//	1,
	//	OcclusionState::EXPOSED,
	//	getBox3d(triB3),
	//	getBox2d(triB2),
	//	Eigen::Vector3d(0,0,1),
	//	triB3,
	//	triB2,
	//	m_preInterB.size(),
	//	m_preInterB.data(),
	//};
	//std::vector<TrigonPart> trigonVct = { trigonA ,trigonB };
	//calculateFrontJudgeOfTrigon(trigonVct, 0, 0, 0);

	return;
}

void test_cuda1()
{
	using namespace eigen;
	using namespace clash;
	TrigonPart trigonA;
	TrigonPart trigonB;
	Triangle2d triA2 = {
		Vector2d{0,0},
		Vector2d{2,0},
		Vector2d{0,1},
	};
	Triangle3d triA3 = {
		Vector3d{0,0,0},
		Vector3d{2,0,1},
		Vector3d{0,1,0},
	};

	Triangle2d triB2 = {
		Vector2d{1,0},
		Vector2d{2,0},
		Vector2d{2,1},
	};
	Triangle3d triB3 = {
		Vector3d{1,0,0},
		Vector3d{2,0,0},
		Vector3d{2,1,0},
	};
	//trigonA
	trigonA.m_index = 0;
	trigonA.m_box3d = getBox(triA3);
	trigonA.m_box2d = getBox(triA2);
	trigonA.m_normal = Vector3d(0, 0, 1);
	trigonA.m_triangle3d = triA3;
	trigonA.m_triangle2d = triA2;
	trigonA.m_preInter = { 1 };
	//trigonB
	trigonB.m_index = 1;
	trigonB.m_box3d = getBox(triB3);
	trigonB.m_box2d = getBox(triB2);
	trigonB.m_normal = Vector3d(0, 0, 1);
	trigonB.m_triangle3d = triB3;
	trigonB.m_triangle2d = triB2;
	trigonB.m_preInter = { 0 };

	std::vector<TrigonPart> trigonVct = { trigonA ,trigonB };
	cuda::calculateFrontJudgeOfTrigon(trigonVct, 0, 0, 0);
	return;
}

static int _enrol = []()
	{
		test_cuda1();
		return 0;
	}();
