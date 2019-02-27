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

// dep: mm
__device__ __host__ inline
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

__device__ __host__ inline
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
__device__ __host__ inline
T std__abs(T in){return (in > 0)? in: (-in);}

struct Scene_info{
    size_t width = 640, height = 480;
    float max_dist_diff = 0.1f; // m
    Mat3x3f K;
    Vec3f* pcd_ptr;
    Vec3f* normal_ptr;  // layout: 1d, width*height length, array of Vec3f

    __device__ __host__
    void query(const Vec3f& src_pcd, Vec3f& dst_pcd, Vec3f& dst_normal, bool& valid) const {
        Vec3i x_y_dep = pcd2dep(src_pcd, K);
        size_t idx = x_y_dep.x + x_y_dep.y * width;
        dst_pcd = pcd_ptr[idx];

        if(dst_pcd.z <= 0 || std__abs(src_pcd.z - dst_pcd.z) > max_dist_diff){
            valid = false;
            return;
        }else valid = true;

        dst_normal = normal_ptr[idx];
    }
};


typedef std::vector<Vec3f> PointCloud_cpu;

#ifdef CUDA_ON
typedef thrust::device_vector<Vec3f> PointCloud_cuda;
/// Functions for ICP registration
RegistrationResult RegistrationICP_cuda(const PointCloud_cuda& model_pcd,
        const Scene_info scene,
        const ICPConvergenceCriteria criteria_conv = ICPConvergenceCriteria());
#endif

RegistrationResult RegistrationICP_cpu(const PointCloud_cpu& model_pcd,
        const Scene_info scene,
        const ICPConvergenceCriteria criteria_conv = ICPConvergenceCriteria());

}
