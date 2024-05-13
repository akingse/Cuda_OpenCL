//https://zhuanlan.zhihu.com/p/563178964

size_t global_work_size[2] = { (size_t)width, (size_t)height };  // 两个维度分别是宽度和高度
size_t local_work_size[2] = { 1, 1 };  // 两个维度的最小处理数据单元是1
cl_event filter_event = NULL;
err_code =
clEnqueueNDRangeKernel(cmd_queue_,
    kernel_filter_,
    2,                 // 数据的维度: 二维数据
    NULL,
    global_work_size,  // 这里指定每一维度的数据项大小
    local_work_size,   // 这里指定最小并行单元的数据大小
    0,
    NULL, &filter_event);
CHK_CLERR(err_code);


__kernel void image_filter(
    const __global uchar* in_img_data,
    const int               pxl_bytes,
    const int               img_line_bytes,
    const __global double* in_coeff,
    const int               coeff_wnd_size,
    __global uchar* out_img_data)
{
    int dims = get_work_dim();       // 获取数据维度，返回2
    int width = get_global_size(0);   // 获取第一维度全局数据大小，就是主机传递的width
    int height = get_global_size(1);   // 获取第二维度全局数据大小，就是主机传递的height
    int center_x = get_global_id(0);     // 获取第一维度当前工作项的索引，即当前像素x坐标
    int center_y = get_global_id(1);     // 获取第二维度当前工作项的索引，即当前像素y坐标
    int local_w = get_local_size(0);     // 获取第一维度局部数据大小，返回1
    int local_h = get_local_size(1);     // 获取第二维度局部数据大小，返回1
    int local_x = get_local_id(0);       // 获取第一维度当前局部工作项的索引，总是返回0
    int local_y = get_local_id(1);       // 获取第二维度当前局部工作项的索引，总是返回0
    //......
}
