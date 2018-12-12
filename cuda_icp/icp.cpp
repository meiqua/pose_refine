#include "icp.h"

namespace cuda_icp {

std::vector<RegistrationResult> RegistrationICP_cpu(const cuda_icp::Image model_deps, const cuda_icp::Image scene_dep,
                                                 Mat3x3f K, const cuda_icp::ICPRejectionCriteria criteria_rej,
                                                 const cuda_icp::ICPConvergenceCriteria criteria_conv)
{
    std::vector<RegistrationResult> results(model_deps.pose_size_);

#pragma omp parallel for
    for(size_t i=0; i<results.size(); i++){
        const auto& depth_entry = model_deps.data_ + i*model_deps.width_*model_deps.height_;

        for(size_t x=0; x<model_deps.width_; x++){
            for(size_t y=0; y<model_deps.height_; y++){

                const int32_t& dep = depth_entry[x + y*model_deps.width_];
                if(dep == 0) continue;

                float z_pcd = dep/1000.0f;
                float x_pcd = (x + model_deps.tl_x_ - K[0][2])/K[0][0]*z_pcd;
                float y_pcd = (y + model_deps.tl_y_ - K[1][2])/K[1][1]*z_pcd;

            }
        }

    }

    return results;
}

std::vector<RegistrationResult> RegistrationICP_cpu(const PointCloud model_pcds, const PointCloud scene_pcd,
                                                    const ICPRejectionCriteria criteria_rej,
                                                    const ICPConvergenceCriteria criteria_conv)
{

}

}


