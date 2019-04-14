#include "depth_scene.h"

void Scene_projective::init_Scene_projective_cpu(cv::Mat& scene_depth, Mat3x3f& scene_K,
                               std::vector<Vec3f>& pcd_buffer, std::vector<Vec3f>& normal_buffer){
        K = scene_K;
        width = scene_depth.cols;
        height = scene_depth.rows;

        int depth_type = scene_depth.type();
        assert(depth_type == CV_16U || depth_type == CV_32S);

        pcd_buffer.clear();
        pcd_buffer.resize(width * height);

        if(depth_type == CV_16U){
            for(int r=0; r<height; r++){
                for(int c=0; c<width; c++){
                    pcd_buffer[c + r*width] = dep2pcd(c, r, scene_depth.at<uint16_t>(r, c), K);
                }
            }
        }else if(depth_type == CV_32S){
            for(int r=0; r<height; r++){
                for(int c=0; c<width; c++){
                    pcd_buffer[c + r*width] = dep2pcd(c, r, scene_depth.at<uint32_t>(r, c), K);
                }
            }
        }

        normal_buffer = get_normal(scene_depth, K);

        pcd_ptr = pcd_buffer.data();
        normal_ptr = normal_buffer.data();
}
