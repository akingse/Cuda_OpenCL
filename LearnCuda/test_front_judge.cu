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

inline std::array<Eigen::Vector2d, 3> to_vec2(const std::array<Eigen::Vector3d, 3>& vec3s)
{
	std::array<Eigen::Vector2d, 3> vec2s;
	for (int i = 0; i < 3; ++i)
		vec2s[i] = Eigen::Vector2d(vec3s[i][0], vec3s[i][1]);// to_vec2(vec3s[i]);
	return vec2s;
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

	//Triangle3d triA3 = {
	//	Vector3d{0,0,0},
	//	Vector3d{2,0,1},
	//	Vector3d{0,1,0},
	//};
	Triangle3d triA3 = {
		Vector3d{-7804.2692441684903, 3000.0000000000000, -12708.512238739511},
		Vector3d{-7821.5827255163249, 0.0000000000000000, -13680.920579031232},
		Vector3d{-7804.2692441684903, 0.0000000000000000, -12708.512238739511},
	};

	//Triangle3d triB3 = {
	//	Vector3d{1,0,0},
	//	Vector3d{2,0,0},
	//	Vector3d{2,1,0},
	//};
	Triangle3d triB3 = {
		Vector3d{-7820.6802403375332, 0.0000000000000000, -13630.232673045370},
		Vector3d{-7820.6802403375323, 3000.0000000000000, -13630.232673045370},
		Vector3d{-8361.3850305782234, 3000.0000000000000, -13710.892917700528},
	};

	//trigonA
	trigonA.m_index = 0;//3;
	trigonA.m_box3d = getBox(triA3);
	trigonA.m_box2d = getBox(to_vec2(triA3));
	trigonA.m_triangle3d = triA3;
	trigonA.m_normal = (triA3[1]- triA3[0]).cross(triA3[2] - triA3[1]).normalized();
	trigonA.m_triangle2d = to_vec2(triA3);
	trigonA.m_preInter = { 1 };
	//trigonB
	trigonB.m_index = 1;// 18;
	trigonB.m_box3d = getBox(triB3);
	trigonB.m_box2d = getBox(to_vec2(triB3));
	trigonB.m_normal = (triB3[1] - triB3[0]).cross(triB3[2] - triB3[1]).normalized();
	trigonB.m_triangle3d = triB3;
	trigonB.m_triangle2d = to_vec2(triB3);
	trigonB.m_preInter = { 0 };

	std::vector<TrigonPart> trigonVct = { trigonA ,trigonB };
	cuda::calculateFrontJudgeOfTrigon(trigonVct, 1e-5, 0.003, 1e-8);
	return;
}

static int _enrol = []()
	{
		test_cuda1();
		return 0;
	}();
