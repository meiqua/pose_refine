#include "pose_refine.h"
#include "cuda_icp/icp.h"
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>

using namespace cv;
using namespace std;

namespace helper {

cv::Rect get_bbox(cv::Mat depth){
    cv::Mat mask = depth > 0;
    cv::Mat Points;
    findNonZero(mask,Points);
    return boundingRect(Points);
}

cv::Mat mat4x4f2cv(Mat4x4f& mat4){
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

cv::Mat view_dep(cv::Mat dep){
    cv::Mat map = dep;
    double min;
    double max;
    cv::minMaxIdx(map, &min, &max);
    cv::Mat adjMap;
    map.convertTo(adjMap,CV_8UC1, 255 / (max-min), -min);
    cv::Mat falseColorsMap;
    applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_HOT);
    return falseColorsMap;
};

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

}

static std::string prefix = "/home/meiqua/pose_refine/test/";

int main(int argc, char const *argv[]){

    return 0;
}
