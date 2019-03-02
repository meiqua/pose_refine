#pragma once

#include "geometry.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

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
    ICPConvergenceCriteria(float relative_fitness = 1e-6f,
            float relative_rmse = 1e-6f, int max_iteration = 30) :
            relative_fitness_(relative_fitness), relative_rmse_(relative_rmse),
            max_iteration_(max_iteration) {}

    float relative_fitness_;
    float relative_rmse_;
    int max_iteration_;
};

// to be used by icp cuda & cpu
// in this way we can avoid eigen mixed with cuda
Mat4x4f eigen_slover_666(float* A, float* b);

typedef std::vector<Vec3f> PointCloud_cpu;
template <class Scene>
RegistrationResult ICP_Point2Plane_cpu(PointCloud_cpu& model_pcd,
        const Scene scene,
        const ICPConvergenceCriteria criteria = ICPConvergenceCriteria());

// depth can be int32, if we use our cuda renderer
// tl_x tl_y: depth may be cropped by renderer directly
template <class T>
PointCloud_cpu depth2cloud_cpu(T* depth, size_t width, size_t height, Mat3x3f& K,
                               size_t tl_x = 0, size_t tl_y = 0);

#ifdef CUDA_ON
typedef thrust::device_vector<Vec3f> PointCloud_cuda;
template <class Scene>
RegistrationResult ICP_Point2Plane_cuda(PointCloud_cuda& model_pcd,
        const Scene scene,
        const ICPConvergenceCriteria criteria = ICPConvergenceCriteria());

// depth can be int32, if we use our cuda renderer
// tl_x tl_y: depth may be cropped by renderer directly
template <class T>
PointCloud_cuda depth2cloud_cuda(T* depth, size_t width, size_t height, Mat3x3f& K,
                                size_t tl_x = 0, size_t tl_y = 0);
#endif

}
