#pragma once

// cuda
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

// load ply
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
namespace cuda_renderer {

class Model{
public:
    Model();
    ~Model();

    Model(const std::string & fileName);

    const struct aiScene* scene;
    void LoadModel(const std::string & fileName);

    struct int3 {
        size_t v0;
        size_t v1;
        size_t v2;
    };

    struct ROI{
        size_t x;
        size_t y;
        size_t width;
        size_t height;
    };

    struct float3{
        float x;
        float y;
        float z;
        friend std::ostream& operator<<(std::ostream& os, const float3& dt)
        {
            os << dt.x << '\t' << dt.y << '\t' << dt.z << std::endl;
            return os;
        }
    };
    struct Triangle{
        float3 v0;
        float3 v1;
        float3 v2;

        friend std::ostream& operator<<(std::ostream& os, const Triangle& dt)
        {
            os << dt.v0 << dt.v1 << dt.v2;
            return os;
        }
    };
    struct mat4x4{
        float a0=1; float a1=0; float a2=0; float a3=0;
        float b0=0; float b1=1; float b2=0; float b3=0;
        float c0=0; float c1=0; float c2=1; float c3=0;
        float d0=0; float d1=0; float d2=0; float d3=1;

        void t(){
            float temp;
            temp = a1; a1=b0; b0=temp;
            temp = a2; a2=c0; c0=temp;
            temp = a3; a3=d0; d0=temp;
            temp = b2; b2=c1; c1=temp;
            temp = b3; b3=d1; d1=temp;
            temp = c3; c3=d2; d2=temp;
        }

        void init_from_cv(const cv::Mat& pose){ // so stupid
            assert(pose.type() == CV_32F);

            a0 = pose.at<float>(0, 0); a1 = pose.at<float>(0, 1);
            a2 = pose.at<float>(0, 2); a3 = pose.at<float>(0, 3);

            b0 = pose.at<float>(1, 0); b1 = pose.at<float>(1, 1);
            b2 = pose.at<float>(1, 2); b3 = pose.at<float>(1, 3);

            c0 = pose.at<float>(2, 0); c1 = pose.at<float>(2, 1);
            c2 = pose.at<float>(2, 2); c3 = pose.at<float>(2, 3);

            d0 = pose.at<float>(3, 0); d1 = pose.at<float>(3, 1);
            d2 = pose.at<float>(3, 2); d3 = pose.at<float>(3, 3);
        }

        void init_from_cv(const cv::Mat& R, const cv::Mat& t){
            assert(R.type() == CV_32F);
            assert(t.type() == CV_32F);

            a0 = R.at<float>(0, 0); a1 = R.at<float>(0, 1);
            a2 = R.at<float>(0, 2); a3 = t.at<float>(0, 0);

            b0 = R.at<float>(1, 0); b1 = R.at<float>(1, 1);
            b2 = R.at<float>(1, 2); b3 = t.at<float>(1, 0);

            c0 = R.at<float>(2, 0); c1 = R.at<float>(2, 1);
            c2 = R.at<float>(2, 2); c3 = t.at<float>(2, 0);

            d0 = 0; d1 = 0;
            d2 = 0; d3 = 1;
        }
    };

    // wanted data
    std::vector<Triangle> tris;
    std::vector<float3> vertices;
    std::vector<int3> faces;
    aiVector3D bbox_min, bbox_max;

    void recursive_render(const struct aiScene *sc, const struct aiNode* nd, aiMatrix4x4 m = aiMatrix4x4());

    static float3 mat_mul_vec(const aiMatrix4x4& mat, const aiVector3D& vec);

    void get_bounding_box_for_node(const aiNode* nd, aiVector3D& min, aiVector3D& max, aiMatrix4x4* trafo) const;
    void get_bounding_box(aiVector3D& min, aiVector3D& max) const;
};

std::vector<Model::mat4x4> mat_to_compact_4x4(const std::vector<cv::Mat>& poses);
Model::mat4x4 compute_proj(const cv::Mat& K, int width, int height, float near=10, float far=10000);


//roi: directly crop while rendering, expected to save much time & space
std::vector<int32_t> render_cuda(const std::vector<Model::Triangle>& tris,const std::vector<Model::mat4x4>& poses,
                            size_t width, size_t height, const Model::mat4x4& proj_mat,
                                 const Model::ROI roi= {0, 0, 0, 0});

std::vector<int32_t> render_cpu(const std::vector<Model::Triangle>& tris,const std::vector<Model::mat4x4>& poses,
                            size_t width, size_t height, const Model::mat4x4& proj_mat,
                                const Model::ROI roi= {0, 0, 0, 0});

thrust::device_vector<int32_t> render_cuda_keep_in_gpu(const std::vector<Model::Triangle>& tris,const std::vector<Model::mat4x4>& poses,
                            size_t width, size_t height, const Model::mat4x4& proj_mat,
                                                       const Model::ROI roi= {0, 0, 0, 0});


//low_level
namespace normal_functor{
    __host__ __device__
    Model::float3 minus(const Model::float3& one, const Model::float3& the_other);

    __host__ __device__
    Model::float3 cross(const Model::float3& one, const Model::float3& the_other);

    __host__ __device__
    Model::float3 normalized(const Model::float3& one);

    __host__ __device__
    Model::float3 get_normal(const Model::Triangle& dev_tri);

    __host__ __device__
    bool is_back(const Model::Triangle& dev_tri);
};

__host__ __device__
Model::float3 mat_mul_v(const Model::mat4x4& tran, const Model::float3& v);

__host__ __device__
Model::Triangle transform_triangle(const Model::Triangle& dev_tri, const Model::mat4x4& tran);

__host__ __device__
float calculateSignedArea(float* A, float* B, float* C);

__host__ __device__
Model::float3 barycentric(float* A, float* B, float* C, size_t* P);

__host__ __device__ inline
float std__max(float a, float b){return (a>b)? a: b;};
__host__ __device__ inline
float std__min(float a, float b){return (a<b)? a: b;};
}
