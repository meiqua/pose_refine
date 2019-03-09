#include "common.h"

static void accumBilateral(long delta, long i, long j, long *A, long *b, int threshold)
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

std::vector<Vec3f> get_normal(const cv::Mat& depth__, const Mat3x3f& K){

    cv::Mat depth;
    int depth_type = depth__.type();
    assert(depth_type == CV_16U || depth_type == CV_32S);
    if(depth_type == CV_32S){
        depth__.convertTo(depth, CV_16U);
    }else{
        depth = depth__;
    }

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
