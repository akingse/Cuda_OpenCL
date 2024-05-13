//https://zhuanlan.zhihu.com/p/563178964

size_t global_work_size[2] = { (size_t)width, (size_t)height };  // ����ά�ȷֱ��ǿ�Ⱥ͸߶�
size_t local_work_size[2] = { 1, 1 };  // ����ά�ȵ���С�������ݵ�Ԫ��1
cl_event filter_event = NULL;
err_code =
clEnqueueNDRangeKernel(cmd_queue_,
    kernel_filter_,
    2,                 // ���ݵ�ά��: ��ά����
    NULL,
    global_work_size,  // ����ָ��ÿһά�ȵ��������С
    local_work_size,   // ����ָ����С���е�Ԫ�����ݴ�С
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
    int dims = get_work_dim();       // ��ȡ����ά�ȣ�����2
    int width = get_global_size(0);   // ��ȡ��һά��ȫ�����ݴ�С�������������ݵ�width
    int height = get_global_size(1);   // ��ȡ�ڶ�ά��ȫ�����ݴ�С�������������ݵ�height
    int center_x = get_global_id(0);     // ��ȡ��һά�ȵ�ǰ�����������������ǰ����x����
    int center_y = get_global_id(1);     // ��ȡ�ڶ�ά�ȵ�ǰ�����������������ǰ����y����
    int local_w = get_local_size(0);     // ��ȡ��һά�Ⱦֲ����ݴ�С������1
    int local_h = get_local_size(1);     // ��ȡ�ڶ�ά�Ⱦֲ����ݴ�С������1
    int local_x = get_local_id(0);       // ��ȡ��һά�ȵ�ǰ�ֲ�����������������Ƿ���0
    int local_y = get_local_id(1);       // ��ȡ�ڶ�ά�ȵ�ǰ�ֲ�����������������Ƿ���0
    //......
}
