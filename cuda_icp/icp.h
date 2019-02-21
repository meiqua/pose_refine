//refer to open3d

# pragma once

# include "geometry.h"

namespace cuda_icp {

/// Class that contains the registration result
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

struct ICPRejectionCriteria
{
public:
    __device__ __host__
    ICPRejectionCriteria(float max_dist_diff = 0.1f) : max_dist_diff_(max_dist_diff) {}

    float max_dist_diff_;
};

struct Image{
    int32_t* data_;  //pointer may take risks of memeory managment, but can unify host & device vector
    size_t tl_x_, tl_y_, width_, height_, pose_size_;

    __device__ __host__
    Image(size_t tl_x=0, size_t tl_y=0, size_t width=640, size_t height=480, size_t pose_size=1) :
    tl_x_(tl_x), tl_y_(tl_y), width_(width), height_(height), pose_size_(pose_size){}
};

struct PointCloud{
    Vec3f* data_;  //pointer may take risks of memeory managment, but can unify host & device vector
    size_t size_;

    __device__ __host__
    PointCloud(Vec3f* data, size_t size): data_(data), size_(size){}
};

// dep: mm
__device__ __host__
Vec3f dep2pcd(size_t x, size_t y, int32_t dep, Mat3x3f K, size_t tl_x=0, size_t tl_y=0){
    float z_pcd = dep/1000.0f;
    float x_pcd = (x + tl_x - K[0][2])/K[0][0]*z_pcd;
    float y_pcd = (y + tl_y - K[1][2])/K[1][1]*z_pcd;
    return {
        x_pcd,
        y_pcd,
        z_pcd
    };
}

__device__ __host__
Vec3i pcd2dep(Vec3f pcd, Mat3x3f K, size_t tl_x=0, size_t tl_y=0){
    int dep = int(pcd.z*1000.0f + 0.5f);
    int x = int(pcd.x/pcd.z*K[0][0] + K[0][2] - tl_x +0.5f);
    int y = int(pcd.y/pcd.z*K[1][1] + K[1][2] - tl_y +0.5f);
    return {
        x,
        y,
        dep
    };
}

template<typename T>
__device__ __host__
T std__abs(T in){return (in > 0)? in: (-in);}


/// Functions for ICP registration
/// depth, mm
std::vector<RegistrationResult> RegistrationICP_cuda(const Image model_deps,
        const Image scene_dep, Mat3x3f K,
        const ICPRejectionCriteria criteria_rej = ICPRejectionCriteria(),
        const ICPConvergenceCriteria criteria_conv = ICPConvergenceCriteria());

std::vector<RegistrationResult> RegistrationICP_cpu(const Image model_deps,
        const Image scene_dep, Mat3x3f K,
        const ICPRejectionCriteria criteria_rej = ICPRejectionCriteria(),
        const ICPConvergenceCriteria criteria_conv = ICPConvergenceCriteria());
}
