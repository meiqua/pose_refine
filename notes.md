renderer cpu can be faster, int32_t to uint16_t  
renderer cuda can't, because atomicMin only support 32bit  
icp can be faster, fuse pcd2Ab with pcd transform;  
thrust is an exception, because thrust::transform_reduce can't can change original data,  
This version is fastest for cuda because thrust is faster than our custom transform_reduce  
