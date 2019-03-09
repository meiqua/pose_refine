#pragma once

#include "geometry.h"

#ifdef CUDA_ON
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
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

//template <class Scene>
RegistrationResult ICP_Point2Plane_cpu(std::vector<Vec3f>& model_pcd,
        const Scene_projective scene,
        const ICPConvergenceCriteria criteria = ICPConvergenceCriteria());

RegistrationResult ICP_Point2Plane_cpu(std::vector<Vec3f>& model_pcd,
        const Scene_nn scene,
        const ICPConvergenceCriteria criteria = ICPConvergenceCriteria());

// depth can be int32, if we use our cuda renderer
// tl_x tl_y: depth may be cropped by renderer directly

//avoid template in headers
//template <class T>
std::vector<Vec3f> depth2cloud_cpu(int32_t* depth, uint32_t width, uint32_t height, Mat3x3f& K, uint32_t stride = 1,
                               uint32_t tl_x = 0, uint32_t tl_y = 0);
std::vector<Vec3f> depth2cloud_cpu(uint16_t* depth, uint32_t width, uint32_t height, Mat3x3f& K, uint32_t stride = 1,
                               uint32_t tl_x = 0, uint32_t tl_y = 0);

#ifdef CUDA_ON
//template <class Scene>
RegistrationResult ICP_Point2Plane_cuda(device_vector_holder<Vec3f>& model_pcd,
        const Scene_projective scene,
        const ICPConvergenceCriteria criteria = ICPConvergenceCriteria());

RegistrationResult ICP_Point2Plane_cuda(device_vector_holder<Vec3f>& model_pcd,
        const Scene_nn scene,
        const ICPConvergenceCriteria criteria = ICPConvergenceCriteria());

// depth can be int32, if we use our cuda renderer
// tl_x tl_y: depth may be cropped by renderer directly

// avoid use template in header
//template <class T>
device_vector_holder<Vec3f> depth2cloud_cuda(int32_t* depth, uint32_t width, uint32_t height, Mat3x3f& K, uint32_t stride = 1,
                     uint32_t tl_x = 0, uint32_t tl_y = 0);
device_vector_holder<Vec3f> depth2cloud_cuda(uint16_t* depth, uint32_t width, uint32_t height, Mat3x3f& K, uint32_t stride = 1,
                                uint32_t tl_x = 0, uint32_t tl_y = 0);
#endif
}


