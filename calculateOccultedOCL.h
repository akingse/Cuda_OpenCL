#pragma once
#ifdef CAL_DLLEXPORT_DEFINE
#define DLLEXPORT_CAL __declspec(dllexport)
#else
#define DLLEXPORT_CAL __declspec(dllimport)
#endif

inline std::string ClReadString(const std::string& filename)
{
    std::ifstream fs(filename);
    if (!fs.is_open())
    {
        std::cout << "open " << filename << " fail!" << std::endl;
    }
    std::string text = std::string(std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>());
    return text;
}

inline const char* cl_ReadString(const std::string& filename)
{
    std::ifstream fs(filename);
    if (!fs.is_open())
    {
        std::cout << "open " << filename << " fail!" << std::endl;
        return nullptr;
    }
    std::string* text = new std::string(std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>());
    return text->c_str();
}


namespace ocl
{
    DLLEXPORT_CAL int calculateFrontJudgeOfTrigon(std::vector<eigen::TrigonPart>& trigonVct, double toleDist, double toleAngle, double toleFixed);
}
