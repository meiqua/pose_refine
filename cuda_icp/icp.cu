#include "icp.h"
#include <thrust/scan.h>
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

__global__
void interleave_size(int* counts, int* masks, size_t size, size_t stride, int last_count){
    size_t index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index >= size) return;

    if(index==0) counts[index] = masks[index*stride + stride];
    else if(index==size-1) counts[index] = masks[index*stride + stride - 1] + last_count - masks[index*stride];
    else counts[index] = masks[index*stride + stride] - masks[index*stride];
}

std::vector<RegistrationResult> RegistrationICP_cuda(const Image model_deps, const Image scene_dep,
                                                  Mat3x3f K, const ICPRejectionCriteria criteria_rej,
                                                  const ICPConvergenceCriteria criteria_conv)
{
    std::vector<RegistrationResult> results(model_deps.pose_size_);

    const size_t threadsPerBlock = 256;

    thrust::device_vector<int> device_counts(model_deps.pose_size_, 0);
    int total_counts = 0;

    thrust::device_vector<int> deps_mask(model_deps.pose_size_*model_deps.width_*model_deps.height_, 0);
    thrust::device_ptr<int> start(model_deps.data_);
    thrust::device_ptr<int> end(model_deps.data_+deps_mask.size());
    thrust::transform(start, end, deps_mask.begin(), is_positive_int());

    // can we read device_vector directly to cpu side?
    int last_count = deps_mask[deps_mask.size()-1];
    thrust::exclusive_scan(deps_mask.begin(), deps_mask.end(), deps_mask.begin());
    total_counts = deps_mask[deps_mask.size()-1];
    total_counts += last_count;

    size_t numBlocks = (device_counts.size() + threadsPerBlock-1)/threadsPerBlock;
    interleave_size<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(device_counts.data()),
                                                    thrust::raw_pointer_cast(deps_mask.data()),
                                                    device_counts.size(), model_deps.width_*model_deps.height_,
                                                    last_count);
    cudaThreadSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    thrust::device_vector<Vec3f> model_pcds(total_counts);

    return results;
}


}

