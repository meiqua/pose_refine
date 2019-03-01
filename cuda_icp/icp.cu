#include "icp.h"
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

// for matrix multi
#include "cublas_v2.h"
#define IDX2C(i, j, ld) (((j)*(ld))+(i))

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

//__global__ void depth2cloud(int* depth, Vec3f* pcd, size_t width, size_t height, int* scan, Mat3x3f K,
//                          size_t tl_x, size_t tl_y){
//    size_t x = blockIdx.x*blockDim.x + threadIdx.x;
//    size_t y = blockIdx.y*blockDim.y + threadIdx.y;
//    if(x>=width) return;
//    if(y>=height) return;
//    size_t index = x + y*width;
//    if(depth[index] == 0) return;

//    pcd[scan[index]] = dep2pcd(x, y, depth[index], K, tl_x, tl_y);
//}

template<class Scene>
__global__ void get_Ab(const Scene scene, Vec3f* model_pcd_ptr, size_t model_pcd_size,
                        float* A_buffer_ptr, float* b_buffer_ptr, uint8_t* valid_buffer_ptr){
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= model_pcd_size) return;

    const auto& src_pcd = model_pcd_ptr[i];

    Vec3f dst_pcd, dst_normal; bool valid;
    scene.query(src_pcd, dst_pcd, dst_normal, valid);

    if(valid){

        // dot
        b_buffer_ptr[i] = (dst_pcd - src_pcd).x * dst_normal.x +
                      (dst_pcd - src_pcd).y * dst_normal.y +
                      (dst_pcd - src_pcd).z * dst_normal.z;

        // cross
        A_buffer_ptr[i*6 + 0] = dst_normal.z*src_pcd.y - dst_normal.y*src_pcd.z;
        A_buffer_ptr[i*6 + 1] = dst_normal.x*src_pcd.z - dst_normal.z*src_pcd.x;
        A_buffer_ptr[i*6 + 2] = dst_normal.y*src_pcd.x - dst_normal.x*src_pcd.y;

        A_buffer_ptr[i*6 + 3] = dst_normal.x;
        A_buffer_ptr[i*6 + 4] = dst_normal.y;
        A_buffer_ptr[i*6 + 5] = dst_normal.z;

        valid_buffer_ptr[i] = 1;
    }
    // else: invalid is 0 in A & b, ATA ATb means adding 0,
    // so don't need to consider valid_buffer, just multi matrix
}

template<class Scene>
RegistrationResult ICP_Point2Plane_cuda(PointCloud_cuda &model_pcd, const Scene scene,
                                        const ICPConvergenceCriteria criteria)
{
    RegistrationResult result;
    RegistrationResult backup;

    // buffer can make pcd handling indenpendent
    // may waste memory, but make it easy to parallel
    thrust::device_vector<float> A_buffer(model_pcd.size()*6, 0);
    thrust::device_vector<float> b_buffer(model_pcd.size(), 0);
    thrust::device_vector<uint8_t> valid_buffer(model_pcd.size(), 0);

    thrust::device_vector<float> A_dev(36);
    thrust::device_vector<float> b_dev(6);

    // cast to pointer, ready to feed kernel
    Vec3f* model_pcd_ptr = thrust::raw_pointer_cast(model_pcd.data());
    float* A_buffer_ptr =  thrust::raw_pointer_cast(A_buffer.data());
    float* b_buffer_ptr =  thrust::raw_pointer_cast(b_buffer.data());
    uint8_t* valid_buffer_ptr =  thrust::raw_pointer_cast(valid_buffer.data());

    float* A_dev_ptr =  thrust::raw_pointer_cast(A_dev.data());
    float* b_dev_ptr =  thrust::raw_pointer_cast(b_dev.data());

    const size_t threadsPerBlock = 256;
    const size_t numBlocks = (model_pcd.size() + threadsPerBlock - 1)/threadsPerBlock;

    /// cublas ----------------------------------------->
    cublasStatus_t stat;  // CUBLAS functions status
    cublasHandle_t cublas_handle;  // CUBLAS context
    stat = cublasCreate(&cublas_handle);
    float alpha =1.0f;  // al =1
    float beta =1.0f;  // bet =1
    /// cublas <-----------------------------------------

    thrust::host_vector<float> A_host(36);
    thrust::host_vector<float> b_host(36);

    float* A_host_ptr = A_host.data();
    float* b_host_ptr = b_host.data();

    for(int iter=0; iter<criteria.max_iteration_; iter++){

        get_Ab<<<numBlocks, threadsPerBlock>>>(scene, model_pcd_ptr, model_pcd.size(),
                                               A_buffer_ptr, b_buffer_ptr, valid_buffer_ptr);
        cudaDeviceSynchronize();

        int count = thrust::reduce(valid_buffer.begin(), valid_buffer.end());
        float total_error = thrust::reduce(b_buffer.begin(), b_buffer.end());

        backup = result;

        result.fitness_ = float(count) / model_pcd.size();
        result.inlier_rmse_ = total_error / count;

        if(std::abs(result.fitness_ - backup.fitness_) < criteria.relative_fitness_ &&
           std::abs(result.inlier_rmse_ - backup.inlier_rmse_) < criteria.relative_rmse_){
            return result;
        }

        // A = A_buffer.transpose()*A_buffer;
        stat = cublasSsyrk(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                            6, model_pcd.size(), &alpha, A_buffer_ptr, 6, &beta, A_dev_ptr, 6);
        stat = cublasGetMatrix(6, 6, sizeof(float), A_dev_ptr , 6, A_host_ptr, 6);

        // b = A_buffer.transpose()*b_buffer;
        stat = cublasSgemv(cublas_handle, CUBLAS_OP_N, 6, model_pcd.size(), &alpha, A_buffer_ptr,
                          6, b_buffer_ptr, 1, &beta, b_dev_ptr, 1);
        stat = cublasGetVector(6, sizeof(float), b_dev_ptr, 1, b_host_ptr, 1);
    }

    return result;
}


}

