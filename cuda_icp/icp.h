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

#ifdef CUDA_ON
    using V3f_holder = device_vector_holder<Vec3f>;
#else
    using V3f_holder = std::vector<Vec3f>;
#endif

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

extern template std::vector<Vec3f> depth2cloud_cpu(int32_t* depth, uint32_t width, uint32_t height, Mat3x3f& K,
                                uint32_t stride, uint32_t tl_x, uint32_t tl_y);
extern template std::vector<Vec3f> depth2cloud_cpu(uint16_t* depth, uint32_t width, uint32_t height, Mat3x3f& K,
                                            uint32_t stride, uint32_t tl_x, uint32_t tl_y);


template <class Scene>
RegistrationResult ICP_Point2Plane_cpu(std::vector<Vec3f>& model_pcd,
        Scene scene,
        const ICPConvergenceCriteria criteria = ICPConvergenceCriteria());

extern template RegistrationResult ICP_Point2Plane_cpu(std::vector<Vec3f> &model_pcd, Scene_projective scene,
const ICPConvergenceCriteria criteria);
extern template RegistrationResult ICP_Point2Plane_cpu(std::vector<Vec3f> &model_pcd, Scene_nn scene,
const ICPConvergenceCriteria criteria);

#ifdef CUDA_ON
// depth can be int32, if we use our cuda renderer
// tl_x tl_y: depth may be cropped by renderer directly
template <class T>
device_vector_holder<Vec3f> depth2cloud_cuda(T* depth, uint32_t width, uint32_t height, Mat3x3f& K, uint32_t stride = 1,
                     uint32_t tl_x = 0, uint32_t tl_y = 0);

extern template device_vector_holder<Vec3f> depth2cloud_cuda(uint16_t *depth, uint32_t width, uint32_t height, Mat3x3f& K,
                                 uint32_t stride, uint32_t tl_x, uint32_t tl_y);
extern template device_vector_holder<Vec3f> depth2cloud_cuda(int32_t *depth, uint32_t width, uint32_t height, Mat3x3f& K,
                                 uint32_t stride, uint32_t tl_x, uint32_t tl_y);


template<class Scene>
RegistrationResult ICP_Point2Plane_cuda(device_vector_holder<Vec3f> &model_pcd, Scene scene,
                                        const ICPConvergenceCriteria criteria = ICPConvergenceCriteria());

extern template RegistrationResult ICP_Point2Plane_cuda(device_vector_holder<Vec3f>&,
Scene_projective, const ICPConvergenceCriteria);
extern template RegistrationResult ICP_Point2Plane_cuda(device_vector_holder<Vec3f>&,
Scene_nn, const ICPConvergenceCriteria);
#endif

template<typename ...Params>
V3f_holder depth2cloud(Params&&...params)
{
#ifdef CUDA_ON
    return cuda_icp::depth2cloud_cuda(std::forward<Params>(params)...);
#else
    return cuda_icp::depth2cloud_cpu(std::forward<Params>(params)...);
#endif
}

template<typename ...Params>
RegistrationResult ICP_Point2Plane(Params&&...params)
{
#ifdef CUDA_ON
    return cuda_icp::ICP_Point2Plane_cuda(std::forward<Params>(params)...);
#else
    return cuda_icp::ICP_Point2Plane_cpu(std::forward<Params>(params)...);
#endif
}


/// !!!!!!!!!!!!!!!!!! low level

typedef vec<29,  float> Vec29f;
// tight: A(symetric 6x6 --> 15+6) + ATb 6 + mse(b*b 1) + count 1 = 29

template<class Scene>
struct thrust__pcd2Ab
{
    Scene __scene;

    __host__ __device__
    thrust__pcd2Ab(Scene scene): __scene(scene){

    }

    __host__ __device__ Vec29f operator()(const Vec3f &src_pcd) const {
        Vec29f result;
        Vec3f dst_pcd, dst_normal; bool valid;
        __scene.query(src_pcd, dst_pcd, dst_normal, valid);
        if(!valid) return result;
        else{
            result[28] = 1;  //valid count
            // dot
            float b_temp = (dst_pcd - src_pcd).x * dst_normal.x +
                          (dst_pcd - src_pcd).y * dst_normal.y +
                          (dst_pcd - src_pcd).z * dst_normal.z;

            // according to https://github.com/intel-isl/Open3D/issues/874#issuecomment-476747366
            // this is better than pow2(t_temp)
            result[27] = pow2((dst_pcd - src_pcd).x) + pow2((dst_pcd - src_pcd).y) + pow2((dst_pcd - src_pcd).z); // mse

            // cross
            float A_temp[6];
            A_temp[0] = dst_normal.z*src_pcd.y - dst_normal.y*src_pcd.z;
            A_temp[1] = dst_normal.x*src_pcd.z - dst_normal.z*src_pcd.x;
            A_temp[2] = dst_normal.y*src_pcd.x - dst_normal.x*src_pcd.y;

            A_temp[3] = dst_normal.x;
            A_temp[4] = dst_normal.y;
            A_temp[5] = dst_normal.z;

            // ATA lower
            // 0  x  x  x  x  x
            // 1  6  x  x  x  x
            // 2  7 11  x  x  x
            // 3  8 12 15  x  x
            // 4  9 13 16 18  x
            // 5 10 14 17 19 20
            result[ 0] = A_temp[0] * A_temp[0];
            result[ 1] = A_temp[0] * A_temp[1];
            result[ 2] = A_temp[0] * A_temp[2];
            result[ 3] = A_temp[0] * A_temp[3];
            result[ 4] = A_temp[0] * A_temp[4];
            result[ 5] = A_temp[0] * A_temp[5];

            result[ 6] = A_temp[1] * A_temp[1];
            result[ 7] = A_temp[1] * A_temp[2];
            result[ 8] = A_temp[1] * A_temp[3];
            result[ 9] = A_temp[1] * A_temp[4];
            result[10] = A_temp[1] * A_temp[5];

            result[11] = A_temp[2] * A_temp[2];
            result[12] = A_temp[2] * A_temp[3];
            result[13] = A_temp[2] * A_temp[4];
            result[14] = A_temp[2] * A_temp[5];

            result[15] = A_temp[3] * A_temp[3];
            result[16] = A_temp[3] * A_temp[4];
            result[17] = A_temp[3] * A_temp[5];

            result[18] = A_temp[4] * A_temp[4];
            result[19] = A_temp[4] * A_temp[5];

            result[20] = A_temp[5] * A_temp[5];

            // ATb
            result[21] = A_temp[0] * b_temp;
            result[22] = A_temp[1] * b_temp;
            result[23] = A_temp[2] * b_temp;
            result[24] = A_temp[3] * b_temp;
            result[25] = A_temp[4] * b_temp;
            result[26] = A_temp[5] * b_temp;
            return result;
        }
    }
};

// used for first only T trans
template<class Scene>
struct thrust__pcd2Ab__only_T
{
    Scene __scene;

    __host__ __device__
    thrust__pcd2Ab__only_T(Scene scene): __scene(scene){

    }

    __host__ __device__ Vec29f operator()(const Vec3f &src_pcd) const {
        Vec29f result;
        Vec3f dst_pcd, dst_normal; bool valid;
        __scene.query(src_pcd, dst_pcd, dst_normal, valid);
        if(!valid) return result;
        else{
            result[28] = 1;  //valid count
            // dot
            float b_temp = (dst_pcd - src_pcd).x * dst_normal.x +
                          (dst_pcd - src_pcd).y * dst_normal.y +
                          (dst_pcd - src_pcd).z * dst_normal.z;

            // according to https://github.com/intel-isl/Open3D/issues/874#issuecomment-476747366
            // this is better than pow2(t_temp)
            result[27] = pow2((dst_pcd - src_pcd).x) + pow2((dst_pcd - src_pcd).y) + pow2((dst_pcd - src_pcd).z); // mse

            // cross
            float A_temp[6];
            A_temp[0] = 0;
            A_temp[1] = 0;
            A_temp[2] = 0;

            A_temp[3] = dst_normal.x;
            A_temp[4] = dst_normal.y;
            A_temp[5] = dst_normal.z;

            // ATA lower
            // 0  x  x  x  x  x
            // 1  6  x  x  x  x
            // 2  7 11  x  x  x
            // 3  8 12 15  x  x
            // 4  9 13 16 18  x
            // 5 10 14 17 19 20
            result[ 0] = A_temp[0] * A_temp[0];
            result[ 1] = A_temp[0] * A_temp[1];
            result[ 2] = A_temp[0] * A_temp[2];
            result[ 3] = A_temp[0] * A_temp[3];
            result[ 4] = A_temp[0] * A_temp[4];
            result[ 5] = A_temp[0] * A_temp[5];

            result[ 6] = A_temp[1] * A_temp[1];
            result[ 7] = A_temp[1] * A_temp[2];
            result[ 8] = A_temp[1] * A_temp[3];
            result[ 9] = A_temp[1] * A_temp[4];
            result[10] = A_temp[1] * A_temp[5];

            result[11] = A_temp[2] * A_temp[2];
            result[12] = A_temp[2] * A_temp[3];
            result[13] = A_temp[2] * A_temp[4];
            result[14] = A_temp[2] * A_temp[5];

            result[15] = A_temp[3] * A_temp[3];
            result[16] = A_temp[3] * A_temp[4];
            result[17] = A_temp[3] * A_temp[5];

            result[18] = A_temp[4] * A_temp[4];
            result[19] = A_temp[4] * A_temp[5];

            result[20] = A_temp[5] * A_temp[5];

            // ATb
            result[21] = A_temp[0] * b_temp;
            result[22] = A_temp[1] * b_temp;
            result[23] = A_temp[2] * b_temp;
            result[24] = A_temp[3] * b_temp;
            result[25] = A_temp[4] * b_temp;
            result[26] = A_temp[5] * b_temp;
            return result;
        }
    }
};

struct thrust__plus{
    __host__ __device__ Vec29f operator()(const Vec29f &in1, const Vec29f &in2) const{
        return in1 + in2;
    }
};

/// !!!!!!!!!!!!!!!!!!!!!!!legacy

#ifdef CUDA_ON
// just for test and comparation
template <class Scene>
RegistrationResult ICP_Point2Plane_cuda_global_memory_version(device_vector_holder<Vec3f>& model_pcd,
        Scene scene,
        const ICPConvergenceCriteria criteria = ICPConvergenceCriteria());

extern template RegistrationResult ICP_Point2Plane_cuda_global_memory_version(device_vector_holder<Vec3f>&,
Scene_projective, const ICPConvergenceCriteria);
extern template RegistrationResult ICP_Point2Plane_cuda_global_memory_version(device_vector_holder<Vec3f>&,
Scene_nn, const ICPConvergenceCriteria);
#endif

template <class Scene>
RegistrationResult ICP_Point2Plane_cpu_global_memory_version(std::vector<Vec3f>& model_pcd,
        Scene scene,
        const ICPConvergenceCriteria criteria = ICPConvergenceCriteria());

extern template RegistrationResult ICP_Point2Plane_cpu_global_memory_version(std::vector<Vec3f> &model_pcd, Scene_projective scene,
const ICPConvergenceCriteria criteria);
extern template RegistrationResult ICP_Point2Plane_cpu_global_memory_version(std::vector<Vec3f> &model_pcd, Scene_nn scene,
const ICPConvergenceCriteria criteria);
}


