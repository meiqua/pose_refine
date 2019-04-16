# pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

#include "cuda_renderer/renderer.h"
#include "cuda_icp/icp.h"

#include <chrono>
#include "Open3D/Core/Registration/Registration.h"
#include "Open3D/Core/Geometry/Image.h"
#include "Open3D/Core/Camera/PinholeCameraIntrinsic.h"
#include "Open3D/Core/Geometry/PointCloud.h"
#include "Open3D/Visualization/Visualization.h"

namespace helper {

static cv::Rect get_bbox(cv::Mat depth){
    cv::Mat mask = depth > 0;
    cv::Mat Points;
    findNonZero(mask,Points);
    return boundingRect(Points);
}

static cv::Mat mat4x4f2cv(Mat4x4f& mat4){
    cv::Mat mat_cv(4, 4, CV_32F);
    mat_cv.at<float>(0, 0) = mat4[0][0];mat_cv.at<float>(0, 1) = mat4[0][1];
    mat_cv.at<float>(0, 2) = mat4[0][2];mat_cv.at<float>(0, 3) = mat4[0][3];

    mat_cv.at<float>(1, 0) = mat4[1][0];mat_cv.at<float>(1, 1) = mat4[1][1];
    mat_cv.at<float>(1, 2) = mat4[1][2];mat_cv.at<float>(1, 3) = mat4[1][3];

    mat_cv.at<float>(2, 0) = mat4[2][0];mat_cv.at<float>(2, 1) = mat4[2][1];
    mat_cv.at<float>(2, 2) = mat4[2][2];mat_cv.at<float>(2, 3) = mat4[2][3];

    mat_cv.at<float>(3, 0) = mat4[3][0];mat_cv.at<float>(3, 1) = mat4[3][1];
    mat_cv.at<float>(3, 2) = mat4[3][2];mat_cv.at<float>(3, 3) = mat4[3][3];

    return mat_cv;
}

static void view_dep_open3d(cv::Mat& modelDepth, cv::Mat modelK = cv::Mat()){

    if(modelK.empty()){
        // from hinter dataset
        modelK = (cv::Mat_<float>(3,3) << 572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0);
    }

    open3d::Image model_depth_open3d;
    model_depth_open3d.PrepareImage(modelDepth.cols, modelDepth.rows, 1, 2);

    std::copy_n(modelDepth.data, model_depth_open3d.data_.size(),
                model_depth_open3d.data_.begin());
    open3d::PinholeCameraIntrinsic K_model_open3d(modelDepth.cols, modelDepth.rows,
                                                  double(modelK.at<float>(0, 0)), double(modelK.at<float>(1, 1)),
                                                  double(modelK.at<float>(0, 2)), double(modelK.at<float>(1, 2)));

    auto model_pcd = open3d::CreatePointCloudFromDepthImage(model_depth_open3d, K_model_open3d);

    double voxel_size = 0.002;
    auto model_pcd_down = open3d::VoxelDownSample(*model_pcd, voxel_size);

//    auto model_pcd_down = open3d::UniformDownSample(*model_pcd, 5);
//    auto model_pcd_down = model_pcd;

    model_pcd_down->PaintUniformColor({1, 0.706, 0});
    open3d::DrawGeometries({model_pcd_down});
}

static void view_pcd(std::vector<::Vec3f>& pcd_in){
    open3d::PointCloud model_pcd;
    for(auto& p: pcd_in){
        if(p.z > 0)
        model_pcd.points_.emplace_back(double(p.x), double(p.y), double(p.z));
    }

    open3d::EstimateNormals(model_pcd);

    double voxel_size = 0.002;
    auto model_pcd_down = open3d::VoxelDownSample(model_pcd, voxel_size);

//    auto model_pcd_down = open3d::UniformDownSample(*model_pcd, 5);
//    auto model_pcd_down = model_pcd;

    model_pcd_down->PaintUniformColor({1, 0.706, 0});
    open3d::DrawGeometries({model_pcd_down});
}

static void view_pcd(std::vector<::Vec3f>& pcd_in, std::vector<::Vec3f>& pcd_in2){
    open3d::PointCloud model_pcd, model_pcd2;
    for(auto& p: pcd_in){
        if(p.z > 0)
        model_pcd.points_.emplace_back(double(p.x), double(p.y), double(p.z));
    }

    for(auto& p: pcd_in2){
        if(p.z > 0)
        model_pcd2.points_.emplace_back(double(p.x), double(p.y), double(p.z));
    }

    open3d::EstimateNormals(model_pcd2);
    open3d::EstimateNormals(model_pcd);

    double voxel_size = 0.002;
    auto model_pcd_down = open3d::VoxelDownSample(model_pcd, voxel_size);
    auto model_pcd_down2 = open3d::VoxelDownSample(model_pcd2, voxel_size);

//    auto model_pcd_down = open3d::UniformDownSample(*model_pcd, 5);
//    auto model_pcd_down = model_pcd;

    model_pcd_down->PaintUniformColor({1, 0.706, 0});
    model_pcd_down2->PaintUniformColor({0, 0.651, 0.929});
    open3d::DrawGeometries({model_pcd_down, model_pcd_down2});
}

static void view_pcd(open3d::PointCloud& model_pcd, open3d::PointCloud& model_pcd2){

    open3d::EstimateNormals(model_pcd2);
    open3d::EstimateNormals(model_pcd);

    double voxel_size = 0.005;
    auto model_pcd_down = open3d::VoxelDownSample(model_pcd, voxel_size);
    auto model_pcd_down2 = open3d::VoxelDownSample(model_pcd2, voxel_size);

    model_pcd_down->PaintUniformColor({1, 0.706, 0});
    model_pcd_down2->PaintUniformColor({0, 0.651, 0.929});
    open3d::DrawGeometries({model_pcd_down, model_pcd_down2});
};

static cv::Mat view_dep(cv::Mat dep){
    cv::Mat map = dep;
    double min;
    double max;
    cv::minMaxIdx(map, &min, &max);
    cv::Mat adjMap;
    map.convertTo(adjMap,CV_8UC1, 255 / (max-min), -min);
    cv::Mat falseColorsMap;
    applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_HOT);
    return falseColorsMap;
}

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s\n" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

static bool isRotationMatrix(cv::Mat &R){
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
    return  norm(I, shouldBeIdentity) < 1e-5;
}

static cv::Vec3f rotationMatrixToEulerAngles(cv::Mat R){
    assert(isRotationMatrix(R));
    float sy = std::sqrt(R.at<float>(0,0) * R.at<float>(0,0) +  R.at<float>(1,0) * R.at<float>(1,0) );

    bool singular = sy < 1e-6f; // If

    float x, y, z;
    if (!singular)
    {
        x = std::atan2(R.at<float>(2,1) , R.at<float>(2,2));
        y = std::atan2(-R.at<float>(2,0), sy);
        z = std::atan2(R.at<float>(1,0), R.at<float>(0,0));
    }
    else
    {
        x = std::atan2(-R.at<float>(1,2), R.at<float>(1,1));
        y = std::atan2(-R.at<float>(2,0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);
}

static cv::Mat eulerAnglesToRotationMatrix(cv::Vec3f theta)
{
    // Calculate rotation about x axis
    cv::Mat R_x = (cv::Mat_<float>(3,3) <<
               1,       0,              0,
               0,       std::cos(theta[0]),   -std::sin(theta[0]),
               0,       std::sin(theta[0]),   std::cos(theta[0])
               );
    // Calculate rotation about y axis
    cv::Mat R_y = (cv::Mat_<float>(3,3) <<
               std::cos(theta[1]),    0,      std::sin(theta[1]),
               0,               1,      0,
               -std::sin(theta[1]),   0,      std::cos(theta[1])
               );
    // Calculate rotation about z axis
    cv::Mat R_z = (cv::Mat_<float>(3,3) <<
               std::cos(theta[2]),    -std::sin(theta[2]),      0,
               std::sin(theta[2]),    std::cos(theta[2]),       0,
               0,               0,                  1);
    // Combined rotation matrix
    cv::Mat R = R_z * R_y * R_x;
    return R;
}
}

//#define USE_PROJ
class PoseRefine {
public:
    cv::Mat scene_depth;
    cv::Mat scene_dep_edge;

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


#ifdef USE_PROJ
    #ifdef CUDA_ON
        ::device_vector_holder<::Vec3f> pcd_buffer, normal_buffer;
    #else
        std::vector<::Vec3f> pcd_buffer, normal_buffer;
    #endif
    Scene_projective scene;
#else
    #ifdef CUDA_ON
        KDTree_cuda kdtree;
    #else
        KDTree_cpu kdtree;
    #endif
    Scene_nn scene;
#endif

    int batch_size = 8;

    PoseRefine(std::string model_path, cv::Mat depth=cv::Mat(), cv::Mat K=cv::Mat());
    void set_depth(cv::Mat depth);
    void set_K(cv::Mat K);
    void set_K_width_height(cv::Mat K, int width, int height);

    // Only search rotation neibor, default is 18 degree.
    // Because linemod can make sure tanslation error is in 4 pixels.
    std::vector<cv::Mat> poses_extend(std::vector<cv::Mat>& init_poses, float degree_var = CV_PI/10);

    std::vector<cuda_icp::RegistrationResult> process_batch(std::vector<cv::Mat>& init_poses,
                                                            int down_sample = 2);

    std::vector<cuda_icp::RegistrationResult> results_filter(std::vector<cuda_icp::RegistrationResult>& results,
                                                            float edge_hit_rate_thresh = 0.5f,
                                                            float fitness_thresh = 0.7f,
                                                            float rmse_thresh = 0.01f);

    std::vector<cv::Mat> render_depth(std::vector<cv::Mat>& init_poses, int down_sample = 1);
    std::vector<cv::Mat> render_mask(std::vector<cv::Mat>& init_poses, int down_sample = 1);
    std::vector<std::vector<cv::Mat>> render_depth_mask(std::vector<cv::Mat>& init_poses, int down_sample = 1);

    template<typename F>
    auto render_what(F f, std::vector<cv::Mat>& init_poses, int down_sample = 1);

    static cv::Mat get_normal(cv::Mat& depth, cv::Mat K = cv::Mat());
    static cv::Mat get_depth_edge(cv::Mat& depth, cv::Mat K = cv::Mat());
    cv::Mat view_dep(cv::Mat dep);
};
