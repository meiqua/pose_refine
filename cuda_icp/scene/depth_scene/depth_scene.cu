#include "depth_scene.h"

void Scene_projective::init_Scene_projective_cuda(cv::Mat &scene_depth, Mat3x3f &scene_K,
                                                  device_vector_holder<Vec3f> &pcd_buffer,
                                                  device_vector_holder<Vec3f> &normal_buffer,
                                                  size_t width_, size_t height_, float max_dist_diff_){
    K = scene_K;
    width = width_;
    height = height_;
    max_dist_diff = max_dist_diff_;

    int depth_type = scene_depth.type();
    assert(depth_type == CV_16U || depth_type == CV_32S);

    std::vector<Vec3f> pcd_buffer_host(width * height);
    std::vector<Vec3f> normal_buffer_host;

    if(depth_type == CV_16U){
        for(int r=0; r<height; r++){
            for(int c=0; c<width; c++){
                pcd_buffer_host[c + r*width] = dep2pcd(c, r, scene_depth.at<uint16_t>(r, c), K);
            }
        }
    }else if(depth_type == CV_32S){
        for(int r=0; r<height; r++){
            for(int c=0; c<width; c++){
                pcd_buffer_host[c + r*width] = dep2pcd(c, r, scene_depth.at<uint32_t>(r, c), K);
            }
        }
    }

    normal_buffer_host = get_normal(scene_depth, K);

    pcd_buffer.__malloc(pcd_buffer_host.size());
    thrust::copy(pcd_buffer_host.begin(), pcd_buffer_host.end(), pcd_buffer.begin_thr());

    normal_buffer.__malloc(normal_buffer_host.size());
    thrust::copy(normal_buffer_host.begin(), normal_buffer_host.end(), normal_buffer.begin_thr());

    pcd_ptr = pcd_buffer.data();
    normal_ptr = normal_buffer.data();
}
