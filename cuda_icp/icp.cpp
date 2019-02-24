#include "icp.h"

namespace cuda_icp {

std::vector<RegistrationResult> RegistrationICP_cpu(const PointCloud model_pcds,
                                                    const Scene_info scene, const ICPRejectionCriteria criteria_rej,
                                                    const ICPConvergenceCriteria criteria_conv)
{
    auto query_proj = [&](){
        Vec3f pcd, normal;
        bool exist = true;

        return std::make_tuple(exist, pcd, normal);
    };
}

}


