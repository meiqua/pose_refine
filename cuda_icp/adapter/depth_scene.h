#pragma once

#include "../geometry.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

// dep: mm
// only used in host side
//__device__ __host__ inline
Vec3f dep2pcd(size_t x, size_t y, int32_t dep, Mat3x3f& K, size_t tl_x=0, size_t tl_y=0){
    float z_pcd = dep/1000.0f;
    float x_pcd = (x + tl_x - K[0][2])/K[0][0]*z_pcd;
    float y_pcd = (y + tl_y - K[1][2])/K[1][1]*z_pcd;
    return {
        x_pcd,
        y_pcd,
        z_pcd
    };
}

__device__ __host__ inline
Vec3i pcd2dep(const Vec3f& pcd, const Mat3x3f& K, size_t tl_x=0, size_t tl_y=0){
    int dep = int(pcd.z*1000.0f + 0.5f);
    int x = int(pcd.x/pcd.z*K[0][0] + K[0][2] - tl_x +0.5f);
    int y = int(pcd.y/pcd.z*K[1][1] + K[1][2] - tl_y +0.5f);
    return {
        x,
        y,
        dep
    };
}

template<typename T>
__device__ __host__ inline
T std__abs(T in){return (in > 0)? in: (-in);}

// just implement query func,
// no matter it's projective or ANN
struct Scene_projective{
    size_t width = 640, height = 480;
    float max_dist_diff = 0.1f; // m
    Mat3x3f K;
    Vec3f* pcd_ptr;  // pointer can unify cpu & cuda version
    Vec3f* normal_ptr;  // layout: 1d, width*height length, array of Vec3f

    void accumBilateral(long delta, long i, long j, long *A, long *b, int threshold)
    {
        long f = std::abs(delta) < threshold ? 1 : 0;

        const long fi = f * i;
        const long fj = f * j;

        A[0] += fi * i;
        A[1] += fi * j;
        A[3] += fj * j;
        b[0] += fi * delta;
        b[1] += fj * delta;
    }

    std::vector<Vec3f> get_normal(const cv::Mat& depth, const Mat3x3f& K){

        std::vector<Vec3f> normals;
        normals.resize(depth.rows * depth.cols);
        // method from linemod depth modality
        {
            cv::Mat src = depth;
            int distance_threshold = 2000;
            int difference_threshold = 50;

            const unsigned short *lp_depth = src.ptr<ushort>();
            Vec3f *lp_normals = normals.data();

            const int l_W = src.cols;
            const int l_H = src.rows;

            const int l_r = 5; // used to be 7
            const int l_offset0 = -l_r - l_r * l_W;
            const int l_offset1 = 0 - l_r * l_W;
            const int l_offset2 = +l_r - l_r * l_W;
            const int l_offset3 = -l_r;
            const int l_offset4 = +l_r;
            const int l_offset5 = -l_r + l_r * l_W;
            const int l_offset6 = 0 + l_r * l_W;
            const int l_offset7 = +l_r + l_r * l_W;

            for (int l_y = l_r; l_y < l_H - l_r - 1; ++l_y)
            {
                const unsigned short *lp_line = lp_depth + (l_y * l_W + l_r);
                Vec3f *lp_norm = lp_normals + (l_y * l_W + l_r);

                for (int l_x = l_r; l_x < l_W - l_r - 1; ++l_x)
                {
                    long l_d = lp_line[0];
                    if (l_d < distance_threshold /*&& l_d > 0*/)
                    {
                        // accum
                        long l_A[4];
                        l_A[0] = l_A[1] = l_A[2] = l_A[3] = 0;
                        long l_b[2];
                        l_b[0] = l_b[1] = 0;
                        accumBilateral(lp_line[l_offset0] - l_d, -l_r, -l_r, l_A, l_b, difference_threshold);
                        accumBilateral(lp_line[l_offset1] - l_d, 0, -l_r, l_A, l_b, difference_threshold);
                        accumBilateral(lp_line[l_offset2] - l_d, +l_r, -l_r, l_A, l_b, difference_threshold);
                        accumBilateral(lp_line[l_offset3] - l_d, -l_r, 0, l_A, l_b, difference_threshold);
                        accumBilateral(lp_line[l_offset4] - l_d, +l_r, 0, l_A, l_b, difference_threshold);
                        accumBilateral(lp_line[l_offset5] - l_d, -l_r, +l_r, l_A, l_b, difference_threshold);
                        accumBilateral(lp_line[l_offset6] - l_d, 0, +l_r, l_A, l_b, difference_threshold);
                        accumBilateral(lp_line[l_offset7] - l_d, +l_r, +l_r, l_A, l_b, difference_threshold);

                        // solve
                        long l_det = l_A[0] * l_A[3] - l_A[1] * l_A[1];
                        long l_ddx = l_A[3] * l_b[0] - l_A[1] * l_b[1];
                        long l_ddy = -l_A[1] * l_b[0] + l_A[0] * l_b[1];

                        /// @todo Magic number 1150 is focal length? This is something like
                        /// f in SXGA mode, but in VGA is more like 530.
                        float l_nx = static_cast<float>(K[0][0] * l_ddx);
                        float l_ny = static_cast<float>(K[1][1] * l_ddy);
                        float l_nz = static_cast<float>(-l_det * l_d);

                        float l_sqrt = sqrtf(l_nx * l_nx + l_ny * l_ny + l_nz * l_nz);

                        if (l_sqrt > 0)
                        {
                            float l_norminv = 1.0f / (l_sqrt);

                            l_nx *= l_norminv;
                            l_ny *= l_norminv;
                            l_nz *= l_norminv;

                            *lp_norm = {l_nx, l_ny, l_nz};
                        }
                    }
                    ++lp_line;
                    ++lp_norm;
                }
            }
        }

        return normals;
    }

    // buffer provided by user, this class only holds pointers,
    // becuase we will pass them to device.
    void init_Scene_projective_cpu(cv::Mat& scene_depth, Mat3x3f& scene_K,
                                   std::vector<Vec3f>& pcd_buffer, std::vector<Vec3f>& normal_buffer,
                                  size_t width_ = 640, size_t height_ = 480,
                                   float max_dist_diff_ = 0.1f){
        K = scene_K;
        width = width_;
        height = height_;
        max_dist_diff = max_dist_diff_;

        pcd_buffer.clear();
        pcd_buffer.resize(width * height);
        for(int r=0; r<height; r++){
            for(int c=0; c<width; c++){
                pcd_buffer[c + r*width] = dep2pcd(c, r, scene_depth.at<uint16_t>(r, c), K);
            }
        }

        normal_buffer = get_normal(scene_depth, K);

        pcd_ptr = pcd_buffer.data();
        normal_ptr = normal_buffer.data();
    }

    void init_Scene_projective_cuda(cv::Mat& scene_depth, Mat3x3f& scene_K,
                                   thrust::device_vector<Vec3f>& pcd_buffer,
                                    thrust::device_vector<Vec3f>& normal_buffer,
                                  size_t width_ = 640, size_t height_ = 480,
                                   float max_dist_diff_ = 0.1f){

        K = scene_K;
        width = width_;
        height = height_;
        max_dist_diff = max_dist_diff_;

        thrust::host_vector<Vec3f> pcd_buffer_host, normal_buffer_host;
        pcd_buffer_host.resize(width * height);
        for(int r=0; r<height; r++){
            for(int c=0; c<width; c++){
                pcd_buffer_host[c + r*width] = dep2pcd(c, r, scene_depth.at<uint16_t>(r, c), K);
            }
        }
        normal_buffer_host = get_normal(scene_depth, K);

        pcd_buffer = pcd_buffer_host;
        normal_buffer = normal_buffer_host;

        pcd_ptr = thrust::raw_pointer_cast(pcd_buffer.data());
        normal_ptr = thrust::raw_pointer_cast(normal_buffer.data());
    }

    __device__ __host__
    void query(const Vec3f& src_pcd, Vec3f& dst_pcd, Vec3f& dst_normal, bool& valid) const {
        Vec3i x_y_dep = pcd2dep(src_pcd, K);
        size_t idx = x_y_dep.x + x_y_dep.y * width;
        dst_pcd = pcd_ptr[idx];

        if(dst_pcd.z <= 0 || std__abs(src_pcd.z - dst_pcd.z) > max_dist_diff){
            valid = false;
            return;
        }else valid = true;

        dst_normal = normal_ptr[idx];
    }
};
