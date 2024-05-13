// to using double
#ifdef FP64
#pragma OpenCL EXTENSION cl_khr_fp64 : enable
#endif   
__kernel void vecAdd(  __global double *a,                    
                       __global double *b,                    
                       __global double *c,                    
                       const unsigned int n)                   
{                                                              
    //Get our global thread ID                                 
    int id = get_global_id(0);                                 
                                                               
    //Make sure we do not go out of bounds                     
    if (id < n)                                                
        c[id] = a[id] + b[id];                                 
}                                                              

/*
每个内核函数的声明都以__kernel或者kernel开头；
内核函数的返回类型必须是void类型；


*/