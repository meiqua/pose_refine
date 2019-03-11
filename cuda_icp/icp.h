#pragma once

#include "geometry.h"

#ifdef CUDA_ON
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "cublas_v2.h"

#endif

#include "scene/depth_scene/depth_scene.h"
#include "scene/pcd_scene/pcd_scene.h"

namespace cuda_icp {

// use custom mat/vec here, otherwise we have to mix eigen with cuda
// then we may face some error due to eigen vesrion
//class defination refer to open3d
struct RegistrationResult
{
    __device__ __host__
    RegistrationResult(const Mat4x4f &transformation =
            Mat4x4f::identity()) : transformation_(transformation),
            inlier_rmse_(0.0), fitness_(0.0) {}

    Mat4x4f transformation_;
    float inlier_rmse_;
    float fitness_;
};

struct ICPConvergenceCriteria
{
public:
    __device__ __host__
    ICPConvergenceCriteria(float relative_fitness = 1e-5f,
            float relative_rmse = 1e-5f, int max_iteration = 30) :
            relative_fitness_(relative_fitness), relative_rmse_(relative_rmse),
            max_iteration_(max_iteration) {}

    float relative_fitness_;
    float relative_rmse_;
    int max_iteration_;
};

// to be used by icp cuda & cpu
// in this way we can avoid eigen mixed with cuda
Mat4x4f eigen_slover_666(float* A, float* b);

// depth can be int32, if we use our cuda renderer
// tl_x tl_y: depth may be cropped by renderer directly
template <class T>
std::vector<Vec3f> depth2cloud_cpu(T* depth, uint32_t width, uint32_t height, Mat3x3f& K, uint32_t stride = 1,
                               uint32_t tl_x = 0, uint32_t tl_y = 0);

template <class Scene>
RegistrationResult ICP_Point2Plane_cpu(std::vector<Vec3f>& model_pcd,
        const Scene scene,
        const ICPConvergenceCriteria criteria = ICPConvergenceCriteria());

#ifdef CUDA_ON
// depth can be int32, if we use our cuda renderer
// tl_x tl_y: depth may be cropped by renderer directly
template <class T>
device_vector_holder<Vec3f> depth2cloud_cuda(T* depth, uint32_t width, uint32_t height, Mat3x3f& K, uint32_t stride = 1,
                     uint32_t tl_x = 0, uint32_t tl_y = 0);


template <class Scene>
RegistrationResult ICP_Point2Plane_cuda(device_vector_holder<Vec3f>& model_pcd,
        const Scene scene,
        const ICPConvergenceCriteria criteria = ICPConvergenceCriteria());

// hold buffer so not to free it after function finished
// independent from the pure func
class ICP_cuda_buffer_holder{
public:
    // buffer can make pcd handling indenpendent
    // may waste memory, but make it easy to parallel
    thrust::device_vector<float> A_buffer;
    thrust::device_vector<float> b_buffer;
//    thrust::device_vector<float> b_squre_buffer;
    thrust::device_vector<uint32_t> valid_buffer;

    thrust::device_vector<float> A_dev;
    thrust::device_vector<float> b_dev;

    thrust::host_vector<float> A_host;
    thrust::host_vector<float> b_host;

//    cublasStatus_t stat;  // CUBLAS functions status
    cublasHandle_t cublas_handle;  // CUBLAS context
    float alpha =1.0f;
    float beta =0.0f;

    ICP_cuda_buffer_holder(size_t pcd_size);

    template <class Scene>
    RegistrationResult ICP_Point2Plane_cuda(device_vector_holder<Vec3f>& model_pcd,
            const Scene scene,
            const ICPConvergenceCriteria criteria = ICPConvergenceCriteria());
};



#endif
}


