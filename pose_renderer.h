# pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include "cuda_renderer/renderer.h"

class PoseRenderer{
public:
    // for rendering
    cv::Mat K;
    int width, height;
    cuda_renderer::Model model;
#ifdef CUDA_ON
    cuda_renderer::device_vector_holder<cuda_renderer::Model::Triangle> tris;
#else
    std::vector<cuda_renderer::Model::Triangle>& tris;
#endif
    cuda_renderer::Model::mat4x4 proj_mat;

    PoseRenderer(std::string model_path, cv::Mat depth=cv::Mat(), cv::Mat K=cv::Mat());
    void set_K_width_height(cv::Mat K, int width, int height);

    std::vector<cv::Mat> render_depth(std::vector<cv::Mat>& init_poses, float down_sample = 1);
    std::vector<cv::Mat> render_mask(std::vector<cv::Mat>& init_poses, float down_sample = 1);
    std::vector<std::vector<cv::Mat>> render_depth_mask(std::vector<cv::Mat>& init_poses, float down_sample = 1);

    template<typename F>
    auto render_what(F f, std::vector<cv::Mat>& init_poses, float down_sample = 1);
    cv::Mat view_dep(cv::Mat dep);
};
