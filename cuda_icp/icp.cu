#include "icp.h"
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>
namespace cuda_icp {

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct is_positive_int{

    __host__ __device__
    int32_t operator()(const int32_t x) const
    {
      return (x>0)? 1: 0;
    }
};

__global__ void depth2cloud(int* depth, Vec3f* pcd, size_t width, size_t height, int* scan, Mat3x3f K,
                          size_t tl_x, size_t tl_y){
    size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x>=width) return;
    if(y>=height) return;
    size_t index = x + y*width;
    if(depth[index] == 0) return;

    pcd[scan[index]] = dep2pcd(x, y, depth[index], K, tl_x, tl_y);
}


}

