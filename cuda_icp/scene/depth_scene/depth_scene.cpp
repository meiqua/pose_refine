#include "depth_scene.h"

void Scene_projective::init_Scene_projective_cpu(cv::Mat& scene_depth, Mat3x3f& scene_K,
                               std::vector<Vec3f>& pcd_buffer, std::vector<Vec3f>& normal_buffer,
                              size_t width_, size_t height_, float max_dist_diff_){
        K = scene_K;
        width = width_;
        height = height_;
        max_dist_diff = max_dist_diff_;

        pcd_buffer.clear();
        pcd_buffer.resize(width * height);
        for(int r=0; r<height; r++){
            for(int c=0; c<width; c++){
                pcd_buffer[c + r*width] = dep2pcd(c, r, scene_depth.at<uint16_t>(r, c), K);
            }
        }

        normal_buffer = get_normal(scene_depth, K);

        pcd_ptr = pcd_buffer.data();
        normal_ptr = normal_buffer.data();
}
