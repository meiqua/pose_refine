#include "icp.h"
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
namespace cuda_icp {

Eigen::Matrix4f TransformVector6dToMatrix4f(const Eigen::Matrix<float, 6, 1> &input) {
    Eigen::Matrix4f output;
    output.setIdentity();
    output.block<3, 3>(0, 0) =
            (Eigen::AngleAxisf(input(2), Eigen::Vector3f::UnitZ()) *
             Eigen::AngleAxisf(input(1), Eigen::Vector3f::UnitY()) *
             Eigen::AngleAxisf(input(0), Eigen::Vector3f::UnitX()))
                    .matrix();
    output.block<3, 1>(0, 3) = input.block<3, 1>(3, 0);
    return output;
}

Mat4x4f eigen_to_custom(const Eigen::Matrix4f& extrinsic){
    Mat4x4f result;
    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++){
            result[i][j] = extrinsic(i, j);
        }
    }
    return result;
}

Mat4x4f eigen_slover_666(float *A, float *b)
{
    Eigen::Matrix<float, 6, 6> A_eigen(A);
    Eigen::Matrix<float, 6, 1> b_eigen(b);
    const Eigen::Matrix<float, 6, 1> update = A_eigen.cast<float>().ldlt().solve(b_eigen.cast<float>());
    Eigen::Matrix4f extrinsic = TransformVector6dToMatrix4f(update);
    return eigen_to_custom(extrinsic);
}

void transform_pcd(PointCloud_cpu& model_pcd, Mat4x4f& trans){

    for(size_t i=0; i < model_pcd.size(); i++){
        Vec3f& pcd = model_pcd[i];
        float new_x = trans[0][0]*pcd.x + trans[0][1]*pcd.y + trans[0][2]*pcd.z + trans[0][3];
        float new_y = trans[1][0]*pcd.x + trans[1][1]*pcd.y + trans[1][2]*pcd.z + trans[1][3];
        float new_z = trans[2][0]*pcd.x + trans[2][1]*pcd.y + trans[2][2]*pcd.z + trans[2][3];
        pcd.x = new_x;
        pcd.y = new_y;
        pcd.z = new_z;
    }
}

template<class Scene>
RegistrationResult ICP_Point2Plane_cpu(PointCloud_cpu &model_pcd, const Scene scene,
                                       const ICPConvergenceCriteria criteria)
{
    RegistrationResult result;
    RegistrationResult backup;

    // buffer can make pcd handling indenpendent
    // may waste memory, but make it easy to parallel
    Eigen::Matrix<float, Eigen::Dynamic, 6> A_buffer(model_pcd.size(), 6); A_buffer.setZero();
    Eigen::Matrix<float, Eigen::Dynamic, 1> b_buffer(model_pcd.size(), 1); b_buffer.setZero();

    std::vector<uint8_t> valid_buffer(model_pcd.size(), 0);

    // use one extra turn
    for(int iter=0; iter<=criteria.max_iteration_; iter++){

#pragma omp parallel for
        for(int i = 0; i<model_pcd.size(); i++){
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
            }
            // else: invalid is 0 in A & b, ATA ATb means adding 0,
            // so don't need to consider valid_buffer, just multi matrix
        }

        int count = 0;
        float total_error = 0;
#pragma omp parallel for reduction(+:count, total_error)
        for(size_t i=0; i<model_pcd.size(); i++){
            count += valid_buffer[i];
            total_error += (b_buffer(i)*b_buffer(i));
        }
        total_error = std::sqrt(total_error);

        backup = result;

        result.fitness_ = float(count) / model_pcd.size();
        result.inlier_rmse_ = total_error / count;

        // last extra iter, just compute fitness & mse
        if(iter == criteria.max_iteration_) return result;

        if(std::abs(result.fitness_ - backup.fitness_) < criteria.relative_fitness_ &&
           std::abs(result.inlier_rmse_ - backup.inlier_rmse_) < criteria.relative_rmse_){
            return result;
        }

        Eigen::Matrix<float, 6, 6> A = A_buffer.transpose()*A_buffer;
        Eigen::Matrix<float, 6, 1> b = A_buffer.transpose()*b_buffer;

        Mat4x4f extrinsic = eigen_slover_666(A.data(), b.data());
        transform_pcd(model_pcd, extrinsic);
        result.transformation_ = extrinsic * result.transformation_;
    }

    // never arrive here
    return result;
}

template<class T>
PointCloud_cpu depth2cloud_cpu(T *depth, size_t width, size_t height, Mat3x3f K, size_t tl_x, size_t tl_y)
{

}


}


