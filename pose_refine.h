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
    cuda_renderer::device_vector_holder<cuda_renderer::Model::Triangle> device_tris;
    ::device_vector_holder<::Vec3f> pcd_buffer_cuda, normal_buffer_cuda;
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

    PoseRefine(std::string model_path, cv::Mat depth=cv::Mat(), cv::Mat K=cv::Mat());
    void set_depth(cv::Mat depth);
    void set_K(cv::Mat K);
    void set_K_width_height(cv::Mat K, int width, int height);

    // Only search rotation neibor, default is 18 degree.
    // Because linemod can make sure tanslation error is in 4 pixels.
    std::vector<cv::Mat> poses_extend(std::vector<cv::Mat>& init_poses, float degree_var = CV_PI/10);

    std::vector<cuda_icp::RegistrationResult> process_batch(std::vector<cv::Mat>& init_poses,
                                                            int down_sample = 2, bool depth_aligned = false);

    std::vector<cuda_icp::RegistrationResult> results_filter(std::vector<cuda_icp::RegistrationResult>& results,
                                                            float edge_hit_rate_thresh = 0.7f,
                                                            float fitness_thresh = 0.7f,
                                                            float rmse_thresh = 0.07f);

    std::vector<cv::Mat> render_depth(std::vector<cv::Mat>& init_poses, int down_sample = 1);
    std::vector<cv::Mat> render_mask(std::vector<cv::Mat>& init_poses, int down_sample = 1);
    std::vector<std::vector<cv::Mat>> render_depth_mask(std::vector<cv::Mat>& init_poses, int down_sample = 1);

    template<typename F>
    auto render_what(F f, std::vector<cv::Mat>& init_poses, int down_sample = 2);

    static cv::Mat get_normal(cv::Mat& depth, cv::Mat K = cv::Mat());
    static cv::Mat get_depth_edge(cv::Mat& depth, cv::Mat K = cv::Mat());
    cv::Mat view_dep(cv::Mat dep);
};
