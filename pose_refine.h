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

    device_vector_holder<::Vec3f> pcd_buffer_cuda, normal_buffer_cuda;
    Scene_projective scene;

    int batch_size = 100;

    PoseRefine(cv::Mat depth, cv::Mat K, std::string model_path);

    std::vector<cuda_icp::RegistrationResult> process_batch(std::vector<cv::Mat>& init_poses,
                                                            int down_sample = 2, bool depth_aligned = false);

    static cv::Mat get_normal(cv::Mat& depth, cv::Mat K = cv::Mat());
    static cv::Mat get_depth_edge(cv::Mat& depth, cv::Mat K = cv::Mat());

private:
    std::vector<cuda_icp::RegistrationResult> process_batch__(std::vector<cv::Mat>& init_poses, int down_sample = 2,
                                                              cuda_icp::ICPConvergenceCriteria criteria =
                                                                        cuda_icp::ICPConvergenceCriteria());
};
