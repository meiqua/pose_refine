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
 
    //in: [pose start_x, start_y] x N
    //out: best_n poses and scores



    // helper
    static bool isRotationMatrix(cv::Mat &R){
        cv::Mat Rt;
        transpose(R, Rt);
        cv::Mat shouldBeIdentity = Rt * R;
        cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
        return  norm(I, shouldBeIdentity) < 1e-6;
    }
    template<class type>
    static cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R){
        assert(isRotationMatrix(R));
        float sy = std::sqrt(R.at<type>(0,0) * R.at<type>(0,0) +  R.at<type>(1,0) * R.at<type>(1,0) );

        bool singular = sy < 1e-6f; // If

        float x, y, z;
        if (!singular)
        {
            x = std::atan2(R.at<type>(2,1) , R.at<type>(2,2));
            y = std::atan2(-R.at<type>(2,0), sy);
            z = std::atan2(R.at<type>(1,0), R.at<type>(0,0));
        }
        else
        {
            x = std::atan2(-R.at<type>(1,2), R.at<type>(1,1));
            y = std::atan2(-R.at<type>(2,0), sy);
            z = 0;
        }
        return cv::Vec3f(x, y, z);
    }
    template<class type>
    static cv::Mat eulerAnglesToRotationMatrix(cv::Vec3f &theta)
    {
        // Calculate rotation about x axis
        cv::Mat R_x = (cv::Mat_<type>(3,3) <<
                   1,       0,              0,
                   0,       std::cos(theta[0]),   -std::sin(theta[0]),
                   0,       std::sin(theta[0]),   std::cos(theta[0])
                   );
        // Calculate rotation about y axis
        cv::Mat R_y = (cv::Mat_<type>(3,3) <<
                   std::cos(theta[1]),    0,      std::sin(theta[1]),
                   0,               1,      0,
                   -std::sin(theta[1]),   0,      std::cos(theta[1])
                   );
        // Calculate rotation about z axis
        cv::Mat R_z = (cv::Mat_<type>(3,3) <<
                   std::cos(theta[2]),    -std::sin(theta[2]),      0,
                   std::sin(theta[2]),    std::cos(theta[2]),       0,
                   0,               0,                  1);
        // Combined rotation matrix
        cv::Mat R = R_z * R_y * R_x;
        return R;
    }
};
