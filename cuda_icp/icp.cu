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

__global__ void getCorrespondences(
       const Vec3f* model_pcd, int pcd_count, int* corrSet_ptr, int* corrSet_bool_ptr, float* rmse,
        Image scene_dep, Mat3x3f K, const ICPRejectionCriteria criteria_rej){

    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=pcd_count) return;

    Vec3i x_y_dep = pcd2dep(model_pcd[i], K, scene_dep.tl_x_, scene_dep.tl_y_);

    int index = x_y_dep.x + x_y_dep.y*int(scene_dep.width_);

    float diff = std__abs(scene_dep.data_[index] - x_y_dep.z)/1000.0f;

    if( diff < criteria_rej.max_dist_diff_){
        rmse[i] = diff;
        corrSet_ptr[i] = index;
        corrSet_bool_ptr[i] = 1;
    }
}

std::vector<RegistrationResult> RegistrationICP_cuda(const Image model_deps, const Image scene_dep,
                                                  Mat3x3f K, const ICPRejectionCriteria criteria_rej,
                                                  const ICPConvergenceCriteria criteria_conv)
{
    const size_t threadsPerBlock = 256;
    const size_t image_len = model_deps.height_*model_deps.height_;
    const size_t pose_size = model_deps.pose_size_;
    assert(image_len>0); assert(pose_size>0);

    thrust::device_vector<Mat4x4f> transform_buffer(pose_size);
    thrust::device_vector<float> fitness_buffer(pose_size);
    thrust::device_vector<float> inlier_rmse_buffer(pose_size);

    thrust::device_vector<int> deps_mask(image_len*pose_size, 0);
    thrust::device_vector<int> pcd_counts(pose_size, 0);

    // refer to
    // https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
    cudaStream_t streams[model_deps.pose_size_];
    for (size_t i = 0; i < model_deps.pose_size_; i++) cudaStreamCreate(&streams[i]);

    for (size_t i = 0; i < model_deps.pose_size_; i++){
         const auto& model_deps_entry = model_deps.data_+i*image_len;
         const auto& deps_mask_entry = thrust::raw_pointer_cast(deps_mask.data()) + i*image_len;
         thrust::transform(thrust::cuda::par.on(streams[i]), model_deps_entry, model_deps_entry+image_len,
                           deps_mask_entry, is_positive_int());
    }
    cudaDeviceSynchronize(); // sync all streams
    // may be better to use multi-threads for multi-streams so we don't need to sync all?
    // sync each thream in for loop seems wrong, becuase it will block the loop.

    for (size_t i = 0; i < model_deps.pose_size_; i++){
         const auto& deps_mask_entry = thrust::raw_pointer_cast(deps_mask.data()) + i*image_len;
         pcd_counts[i] = thrust::reduce(thrust::cuda::par.on(streams[i]), deps_mask_entry, deps_mask_entry+image_len);
    }
    cudaDeviceSynchronize(); // sync all streams


    int total_counts = thrust::reduce(pcd_counts.begin(), pcd_counts.end());

    thrust::device_vector<Vec3f> model_pcds(total_counts);
    thrust::device_vector<int> corrSet(total_counts, -1);
    thrust::device_vector<int> corrSet_bool(total_counts, 0);
    thrust::device_vector<float> rmse(total_counts, 0);

    thrust::host_vector<int> pcd_counts_host = pcd_counts;
    thrust::exclusive_scan(pcd_counts.begin(), pcd_counts.end(), pcd_counts.begin());
    thrust::host_vector<int> pcd_counts_entry_host = pcd_counts;

    for (size_t i = 0; i < model_deps.pose_size_; i++){
        const auto& deps_mask_entry = thrust::raw_pointer_cast(deps_mask.data()) + i*image_len;
        thrust::exclusive_scan(thrust::cuda::par.on(streams[i]), deps_mask_entry, deps_mask_entry+image_len, deps_mask_entry);
    }
    cudaDeviceSynchronize(); // sync all streams

    for (size_t i = 0; i < model_deps.pose_size_; i++){
        const auto& model_deps_entry = model_deps.data_+i*image_len;
        const auto& deps_mask_entry = thrust::raw_pointer_cast(deps_mask.data()) + i*image_len;
        const auto& model_pcds_entry = thrust::raw_pointer_cast(model_pcds.data()) + pcd_counts_entry_host[i];

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((model_deps.width_ + 15)/16, (model_deps.height_ + 15)/16);
        depth2cloud<<< numBlocks, threadsPerBlock, 0, streams[i] >>>(model_deps_entry, model_pcds_entry, model_deps.width_,
                                                                 model_deps.height_, deps_mask_entry, K,
                                                                 model_deps.tl_x_, model_deps.tl_y_);
    }        // model pcd OK
    cudaDeviceSynchronize(); // sync all streams

    for (size_t i = 0; i < model_deps.pose_size_; i++){
        const auto& model_pcds_entry = thrust::raw_pointer_cast(model_pcds.data()) + pcd_counts_entry_host[i];
        const auto& corrSet_entry = thrust::raw_pointer_cast(corrSet.data()) + pcd_counts_entry_host[i];
        const auto& corrSet_bool_entry = thrust::raw_pointer_cast(corrSet_bool.data()) + pcd_counts_entry_host[i];
        const auto& rmse_entry = thrust::raw_pointer_cast(rmse.data()) + pcd_counts_entry_host[i];

        size_t numBlocks = (pcd_counts_host[i] + threadsPerBlock-1)/threadsPerBlock;
        getCorrespondences<<< numBlocks, threadsPerBlock, 0, streams[i] >>>
                (model_pcds_entry, pcd_counts_host[i],
                 corrSet_entry, corrSet_bool_entry, rmse_entry, scene_dep, K, criteria_rej);
    }
    cudaDeviceSynchronize();

    for (size_t i = 0; i < model_deps.pose_size_; i++){
        const auto& corrSet_bool_entry = thrust::raw_pointer_cast(corrSet_bool.data()) + pcd_counts_entry_host[i];
        const auto& rmse_entry = thrust::raw_pointer_cast(rmse.data()) + pcd_counts_entry_host[i];

        fitness_buffer[i] = thrust::reduce(thrust::cuda::par.on(streams[i]),
                                           corrSet_bool_entry, corrSet_bool_entry+pcd_counts[i]);
        inlier_rmse_buffer[i] = thrust::reduce(thrust::cuda::par.on(streams[i]),
                                               rmse_entry, rmse_entry+pcd_counts[i]);
    }
    cudaDeviceSynchronize();


    cudaDeviceSynchronize();

    for (size_t i = 0; i < model_deps.pose_size_; i++) cudaStreamDestroy(streams[i]);

//    return results;
}


}

