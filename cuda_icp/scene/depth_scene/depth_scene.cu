#include "depth_scene.h"

void Scene_projective::init_Scene_projective_cuda(cv::Mat &scene_depth, Mat3x3f &scene_K,
                                                  device_vector_holder<Vec3f> &pcd_buffer,
                                                  device_vector_holder<Vec3f> &normal_buffer,
                                                  size_t width_, size_t height_, float max_dist_diff_){
    std::vector<Vec3f> pcd_buffer_host;
    std::vector<Vec3f> normal_buffer_host;
    init_Scene_projective_cpu(scene_depth, scene_K, pcd_buffer_host, normal_buffer_host, width_,
                               height_, max_dist_diff_);

    pcd_buffer.__malloc(pcd_buffer_host.size());
    thrust::copy(pcd_buffer_host.begin(), pcd_buffer_host.end(), pcd_buffer.begin_thr());

    normal_buffer.__malloc(normal_buffer_host.size());
    thrust::copy(normal_buffer_host.begin(), normal_buffer_host.end(), normal_buffer.begin_thr());

    pcd_ptr = pcd_buffer.data();
    normal_ptr = normal_buffer.data();
}
