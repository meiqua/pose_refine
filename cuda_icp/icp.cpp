#include "icp.h"
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>

namespace cuda_icp{
Eigen::Matrix4d TransformVector6dToMatrix4d(const Eigen::Matrix<double, 6, 1> &input) {
    Eigen::Matrix4d output;
    output.setIdentity();
    output.block<3, 3>(0, 0) =
            (Eigen::AngleAxisd(input(2), Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(input(1), Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(input(0), Eigen::Vector3d::UnitX()))
                    .matrix();
    output.block<3, 1>(0, 3) = input.block<3, 1>(3, 0);
    return output;
}

Mat4x4f eigen_to_custom(const Eigen::Matrix4f& extrinsic){
    Mat4x4f result;
    for(uint32_t i=0; i<4; i++){
        for(uint32_t j=0; j<4; j++){
            result[i][j] = extrinsic(i, j);
        }
    }
    return result;
}

Mat4x4f eigen_slover_666(float *A, float *b)
{
    Eigen::Matrix<float, 6, 6> A_eigen(A);
    Eigen::Matrix<float, 6, 1> b_eigen(b);
    // ICP point to plane may be unstable, refer to
    // https://www.cs.princeton.edu/~smr/papers/icpstability.pdf
    // add a term ||x|| to make update reasonably small:
    // f = ||(Rp + T - q) * n|| + penalty * ||X||   ==>
    // (ATA + Identity * penalty) * X = B
    Eigen::Matrix6d iden = Eigen::Matrix6d::Identity();
    double penalty = 0.01;
    Eigen::Matrix6d ATA_with_pen = A_eigen.cast<double>() + penalty*iden;
    
    const Eigen::Matrix<double, 6, 1> update = ATA_with_pen.ldlt().solve(b_eigen.cast<double>());    
    Eigen::Matrix4d extrinsic = TransformVector6dToMatrix4d(update);
    return eigen_to_custom(extrinsic.cast<float>());
}

void transform_pcd(std::vector<Vec3f>& model_pcd, Mat4x4f& trans){

#pragma omp parallel for
    for(uint32_t i=0; i < model_pcd.size(); i++){
        Vec3f& pcd = model_pcd[i];
        float new_x = trans[0][0]*pcd.x + trans[0][1]*pcd.y + trans[0][2]*pcd.z + trans[0][3];
        float new_y = trans[1][0]*pcd.x + trans[1][1]*pcd.y + trans[1][2]*pcd.z + trans[1][3];
        float new_z = trans[2][0]*pcd.x + trans[2][1]*pcd.y + trans[2][2]*pcd.z + trans[2][3];
        pcd.x = new_x;
        pcd.y = new_y;
        pcd.z = new_z;
    }
}

template<class T>
void cpu_exclusive_scan_serial(T* start, uint32_t N){
    T cache = start[0];
    start[0] = 0;
    for (uint32_t i = 1; i < N; i++)
    {
        T temp = cache + start[i-1];
        cache = start[i];
        start[i] = temp;
    }
}

template<class T>
std::vector<Vec3f> depth2cloud_cpu(T* depth, uint32_t width, uint32_t height, Mat3x3f& K,
                                uint32_t stride, uint32_t tl_x, uint32_t tl_y)
{
    std::vector<uint32_t> mask(width*height/stride/stride, 0);

#pragma omp parallel for collapse(2)
    for(uint32_t x=0; x<width/stride; x++){
        for(uint32_t y=0; y<height/stride; y++){
            if(depth[x*stride + y*stride*width] > 0) mask[x + y*width] = 1;
        }
    }

    // scan to find map: depth idx --> cloud idx
    uint32_t mask_back_temp = mask.back();

    // without cuda this can't be used
#ifdef CUDA_ON
//    thrust::exclusive_scan(thrust::host, mask.begin(), mask.end(), mask.begin(), 0); // in-place scan
    cpu_exclusive_scan_serial(mask.data(), mask.size()); // serial version is better in cpu maybe
#else
    cpu_exclusive_scan_serial(mask.data(), mask.size());
#endif
    uint32_t total_pcd_num = mask.back() + mask_back_temp;

    std::vector<Vec3f> cloud(total_pcd_num);

#pragma omp parallel for collapse(2)
    for(uint32_t x=0; x<width/stride; x++){
        for(uint32_t y=0; y<height/stride; y++){

            uint32_t idx_depth = x*stride + y*stride*width;
            uint32_t idx_mask = x + y*width;

            if(depth[idx_depth] <= 0) continue;

            float z_pcd = depth[idx_depth]/1000.0f;
            float x_pcd = (x + tl_x - K[0][2])/K[0][0]*z_pcd;
            float y_pcd = (y + tl_y - K[1][2])/K[1][1]*z_pcd;

            cloud[mask[idx_mask]] = {x_pcd, y_pcd, z_pcd};
        }
    }
    return cloud;
}

template std::vector<Vec3f> depth2cloud_cpu(int32_t* depth, uint32_t width, uint32_t height, Mat3x3f& K,
                                uint32_t stride, uint32_t tl_x, uint32_t tl_y);
template std::vector<Vec3f> depth2cloud_cpu(uint16_t* depth, uint32_t width, uint32_t height, Mat3x3f& K,
                                            uint32_t stride, uint32_t tl_x, uint32_t tl_y);


template<class Scene>
RegistrationResult ICP_Point2Plane_cpu(std::vector<Vec3f> &model_pcd, const Scene scene,
                                       const ICPConvergenceCriteria criteria)
{
    RegistrationResult result;
    RegistrationResult backup;

    std::vector<float> A_host(36, 0);
    std::vector<float> b_host(6, 0);
    thrust__pcd2Ab<Scene> trasnformer(scene);

    // use one extra turn
    for(uint32_t iter=0; iter<=criteria.max_iteration_; iter++){

        Vec29f reducer;

#pragma omp declare reduction( + : Vec29f : omp_out += omp_in) \
                       initializer (omp_priv = Vec29f::Zero())

#pragma omp parallel for reduction(+: reducer)
        for(size_t pcd_iter=0; pcd_iter<model_pcd.size(); pcd_iter++){
            Vec29f result = trasnformer(model_pcd[pcd_iter]);
            reducer += result;
        }

        Vec29f& Ab_tight = reducer;

        backup = result;

        float& count = Ab_tight[28];
        float& total_error = Ab_tight[27];
        if(count == 0) return result;  // avoid divid 0

        result.fitness_ = float(count) / model_pcd.size();
        result.inlier_rmse_ = std::sqrt(total_error / count);

        // last extra iter, just compute fitness & mse
        if(iter == criteria.max_iteration_) return result;

        if(std::abs(result.fitness_ - backup.fitness_) < criteria.relative_fitness_ &&
           std::abs(result.inlier_rmse_ - backup.inlier_rmse_) < criteria.relative_rmse_){
            return result;
        }

        for(int i=0; i<6; i++) b_host[i] = Ab_tight[21 + i];

        int shift = 0;
        for(int y=0; y<6; y++){
            for(int x=y; x<6; x++){
                A_host[x + y*6] = Ab_tight[shift];
                A_host[y + x*6] = Ab_tight[shift];
                shift++;
            }
        }

        Mat4x4f extrinsic = eigen_slover_666(A_host.data(), b_host.data());

        transform_pcd(model_pcd, extrinsic);
        result.transformation_ = extrinsic * result.transformation_;
    }

    // never arrive here
    return result;
}

template RegistrationResult ICP_Point2Plane_cpu(std::vector<Vec3f> &model_pcd, const Scene_projective scene,
const ICPConvergenceCriteria criteria);
template RegistrationResult ICP_Point2Plane_cpu(std::vector<Vec3f> &model_pcd, const Scene_nn scene,
const ICPConvergenceCriteria criteria);


/// !!!!!!!!!!!!!!!!!!!!! legacy
// just for test and comparation
template<class Scene>
RegistrationResult ICP_Point2Plane_cpu_global_memory_version(std::vector<Vec3f> &model_pcd, const Scene scene,
                                       const ICPConvergenceCriteria criteria)
{
    RegistrationResult result;
    RegistrationResult backup;

    // buffer can make pcd handling indenpendent
    // may waste memory, but make it easy to parallel
    Eigen::Matrix<float, Eigen::Dynamic, 6> A_buffer(model_pcd.size(), 6); A_buffer.setZero();
    Eigen::Matrix<float, Eigen::Dynamic, 1> b_buffer(model_pcd.size(), 1); b_buffer.setZero();

    std::vector<uint32_t> valid_buffer(model_pcd.size(), 0);

    // use one extra turn
    for(uint32_t iter=0; iter<=criteria.max_iteration_; iter++){

#pragma omp parallel for
        for(uint32_t i = 0; i<model_pcd.size(); i++){
            const auto& src_pcd = model_pcd[i];

            Vec3f dst_pcd, dst_normal; bool valid;
            scene.query(src_pcd, dst_pcd, dst_normal, valid);
            if(valid){

                // dot
                b_buffer(i) = (dst_pcd - src_pcd).x * dst_normal.x +
                              (dst_pcd - src_pcd).y * dst_normal.y +
                              (dst_pcd - src_pcd).z * dst_normal.z;

                // cross
                A_buffer(i, 0) = dst_normal.z*src_pcd.y - dst_normal.y*src_pcd.z;
                A_buffer(i, 1) = dst_normal.x*src_pcd.z - dst_normal.z*src_pcd.x;
                A_buffer(i, 2) = dst_normal.y*src_pcd.x - dst_normal.x*src_pcd.y;

                A_buffer(i, 3) = dst_normal.x;
                A_buffer(i, 4) = dst_normal.y;
                A_buffer(i, 5) = dst_normal.z;

                valid_buffer[i] = 1;
            }else{
                b_buffer(i) = 0;

                A_buffer(i, 0) = 0;
                A_buffer(i, 1) = 0;
                A_buffer(i, 2) = 0;
                A_buffer(i, 3) = 0;
                A_buffer(i, 4) = 0;
                A_buffer(i, 5) = 0;

                valid_buffer[i] = 0;
            }
            // else: invalid is 0 in A & b, ATA ATb means adding 0,
            // so don't need to consider valid_buffer, just multi matrix
        }

        uint32_t count = 0;
        float total_error = 0;
#pragma omp parallel for reduction(+:count, total_error)
        for(uint32_t i=0; i<model_pcd.size(); i++){
            count += valid_buffer[i];
            total_error += (b_buffer(i)*b_buffer(i));
        }

        backup = result;

        if(count == 0) return result;  // avoid divid 0

        result.fitness_ = float(count) / model_pcd.size();
        result.inlier_rmse_ = std::sqrt(total_error / count);

//        {
//            std::cout << " --- cpu --- " << iter << " --- cpu ---" << std::endl;
//            std::cout << "total error: " << total_error << std::endl;
//            std::cout << "result.fitness_: " << result.fitness_ << std::endl;
//            std::cout << "result.inlier_rmse_: " << result.inlier_rmse_ << std::endl;
//            std::cout << " --- cpu --- " << iter << " --- cpu ---" << std::endl << std::endl;
//        }

        // last extra iter, just compute fitness & mse
        if(iter == criteria.max_iteration_) return result;

        if(std::abs(result.fitness_ - backup.fitness_) < criteria.relative_fitness_ &&
           std::abs(result.inlier_rmse_ - backup.inlier_rmse_) < criteria.relative_rmse_){
            return result;
        }

        Eigen::Matrix<float, 6, 6> A = A_buffer.transpose()*A_buffer;
        Eigen::Matrix<float, 6, 1> b = A_buffer.transpose()*b_buffer;

//        std::cout << "~~~~~~~~A~~~~~~" << std::endl;
//        std::cout << A;
//        std::cout << "\n~~~~~~~~~~~~~~\n" << std::endl;

//        std::cout << "~~~~~~~~b~~~~~~" << std::endl;
//        std::cout << b;
//        std::cout << "\n~~~~~~~~~~~~~~\n" << std::endl;

        Mat4x4f extrinsic = eigen_slover_666(A.data(), b.data());

//        std::cout << "~~extrinsic~~~~" << std::endl;
//        std::cout << extrinsic;
//        std::cout << "\n~~~~~~~~~~~~~~\n" << std::endl;

        transform_pcd(model_pcd, extrinsic);
        result.transformation_ = extrinsic * result.transformation_;
    }

    // never arrive here
    return result;
}

template RegistrationResult ICP_Point2Plane_cpu_global_memory_version(std::vector<Vec3f> &model_pcd, const Scene_projective scene,
const ICPConvergenceCriteria criteria);
template RegistrationResult ICP_Point2Plane_cpu_global_memory_version(std::vector<Vec3f> &model_pcd, const Scene_nn scene,
const ICPConvergenceCriteria criteria);
}





