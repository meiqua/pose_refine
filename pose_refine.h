# pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

#include "cuda_renderer/renderer.h"

class PoseRefine {
public:
    cv::Mat scene_depth;
    cv::Mat scene_dep_edge;

    // for rendering
    cv::Mat K;
    size_t width, height;
    cuda_renderer::Model model;
};
