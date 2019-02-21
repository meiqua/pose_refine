#include "icp.h"

namespace cuda_icp {

std::vector<RegistrationResult> RegistrationICP_cpu(const cuda_icp::Image model_deps, const cuda_icp::Image scene_dep,
                                                 Mat3x3f K, const cuda_icp::ICPRejectionCriteria criteria_rej,
                                                 const cuda_icp::ICPConvergenceCriteria criteria_conv)
{
    std::vector<RegistrationResult> results(model_deps.pose_size_);

    std::vector<Vec3f> scene_pcd_data;
    scene_pcd_data.reserve(scene_dep.width_*scene_dep.height_);
    std::vector<size_t> idx_dep2pcd(scene_dep.width_*scene_dep.height_);
    for(size_t x=0; x<scene_dep.width_; x++){
        for(size_t y=0; y<scene_dep.height_; y++){

            size_t idx = x + y*scene_dep.width_;
            const int32_t dep = scene_dep.data_[idx];
            if(dep == 0) continue;

            idx_dep2pcd[idx] = scene_pcd_data.size();
            scene_pcd_data.emplace_back(dep2pcd(x, y, dep, K));
        }
    }


    // only need model pcd, kdtree(for closest corr) or scene_dep(for proj corr) are constructed before called
    auto GetResultAndCorr = [&](
           const PointCloud model_pcd, std::vector<Vec2i>& corrSet,
            const ICPRejectionCriteria criteria_rej = ICPRejectionCriteria()){

        corrSet.clear();
        corrSet.reserve(model_pcd.size_);
        RegistrationResult result;
        size_t inliers = 0;
        float rmse = 0;
        for(size_t i=0; i<model_pcd.size_; i++){
            Vec3i x_y_dep = pcd2dep(model_pcd.data_[i], K, scene_dep.tl_x_, scene_dep.tl_y_);

            auto index = x_y_dep.x + x_y_dep.y*int(scene_dep.width_);

            float diff = std__abs(scene_dep.data_[index] - x_y_dep.z)/1000.0f;

            if( diff < criteria_rej.max_dist_diff_){
                inliers ++;
                rmse += diff;
                corrSet.emplace_back(i, idx_dep2pcd[index]);
            }
        }
        result.fitness_ = float(inliers)/model_pcd.size_;
        result.inlier_rmse_ = rmse/inliers;
        return result;
    };

    auto ComputeTransformation = [](
                const PointCloud &source,
                const PointCloud &target,
            const std::vector<Vec2i> &corres){
        Mat4x4f transform;


        return transform;
    };

    auto ComputeRMSE = [](
                const PointCloud &source,
                const PointCloud &target,
                const std::vector<Vec2i> &corres){
        float rmse = 0;

        return rmse;
    };

//#pragma omp parallel for
    for(size_t i=0; i<results.size(); i++){
        const auto img_len = model_deps.width_*model_deps.height_;
        const auto& depth_entry = model_deps.data_ + i*img_len;

        std::vector<Vec3f> model_pcd_data;
        model_pcd_data.reserve(img_len);
        for(size_t x=0; x<model_deps.width_; x++){
            for(size_t y=0; y<model_deps.height_; y++){
                const int32_t dep = depth_entry[x + y*model_deps.width_];
                if(dep == 0) continue;
                model_pcd_data.emplace_back(dep2pcd(x, y, dep, K));
            }
        }
        PointCloud model_pcd(model_pcd_data.data(), model_pcd_data.size());

        std::vector<Vec2i> corrSet;
        auto result_init = GetResultAndCorr(model_pcd, corrSet);
    }

    return results;
}

}


