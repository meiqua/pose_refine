# pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

#include "cuda_renderer/renderer.h"
#include "cuda_icp/icp.h"

class PoseRefine {
public:
    cv::Mat scene_depth;
    cv::Mat scene_dep_edge;

    // for rendering
    cv::Mat K;
    int width, height;
    cuda_renderer::Model model;
    cuda_renderer::Model::mat4x4 proj_mat;
#ifdef CUDA_ON
    device_vector_holder<::Vec3f> pcd_buffer_cuda, normal_buffer_cuda;
#else
    std::vector<::Vec3f> pcd_buffer, normal_buffer;
#endif
    Scene_projective scene;

    // render & icp batch size, 8 is better on CPU
#ifdef CUDA_ON
    int batch_size = 100;
#else
    int batch_size = 8;
#endif
    PoseRefine(cv::Mat depth, cv::Mat K, std::string model_path);

    // Only search rotation neibor, default is 18 degree.
    // Because linemod can make sure tanslation error is in 4 pixels.
    std::vector<cv::Mat> poses_extend(std::vector<cv::Mat>& init_poses, float degree_var = CV_PI/10);

    std::vector<cuda_icp::RegistrationResult> process_batch(std::vector<cv::Mat>& init_poses,
                                                            int down_sample = 2, bool depth_aligned = false);

    std::vector<cuda_icp::RegistrationResult> results_filter(std::vector<cuda_icp::RegistrationResult>& results,
                                                            float edge_hit_rate_thresh = 0.7f,
                                                            float fitness_thresh = 0.7f,
                                                            float rmse_thresh = 0.07f);

    static cv::Mat get_normal(cv::Mat& depth, cv::Mat K = cv::Mat());
    static cv::Mat get_depth_edge(cv::Mat& depth, cv::Mat K = cv::Mat());
};
