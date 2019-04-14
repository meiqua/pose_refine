#pragma once

#ifdef CUDA_ON
// cuda
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#else
// invalidate cuda macro
#define __device__
#define __host__

#endif

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
        int v0;
        int v1;
        int v2;
    };

    struct ROI{
        int x;
        int y;
        int width;
        int height;
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

        friend std::ostream& operator<<(std::ostream& os, const mat4x4& dt)
        {
            os << dt.a0 << '\t' << dt.a1 << '\t' << dt.a2 << '\t' << dt.a3 << std::endl;
            os << dt.b0 << '\t' << dt.b1 << '\t' << dt.b2 << '\t' << dt.b3 << std::endl;
            os << dt.c0 << '\t' << dt.c1 << '\t' << dt.c2 << '\t' << dt.c3 << std::endl;
            os << dt.d0 << '\t' << dt.d1 << '\t' << dt.d2 << '\t' << dt.d3 << std::endl;
            return os;
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

        void init_from_ptr(const float* data){
            a0 = data[0]; a1 = data[1]; a2 = data[2]; a3 = data[3];
            b0 = data[4]; b1 = data[5]; b2 = data[6]; b3 = data[7];
            c0 = data[8]; c1 = data[9]; c2 = data[10]; c3 = data[11];
            d0 = data[12]; d1 = data[13]; d2 = data[14]; d3 = data[15];
        }

        void init_from_ptr(const float* R, const float* t){
            a0 = R[0]; a1 = R[1]; a2 = R[2];  a3 = t[0];
            b0 = R[3]; b1 = R[4]; b2 = R[5];  b3 = t[1];
            c0 = R[6]; c1 = R[7]; c2 = R[8];  c3 = t[2];
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


#ifdef CUDA_ON
// thrust device vector can't be used in cpp by design
// same codes in cuda renderer,
// because we don't want these two related to each other
template <typename T>
class device_vector_holder{
public:
    T* __gpu_memory;
    size_t __size;
    bool valid = false;
    device_vector_holder(){}
    device_vector_holder(size_t size);
    device_vector_holder(size_t size, T init);
    ~device_vector_holder();

    T* data(){return __gpu_memory;}
    thrust::device_ptr<T> data_thr(){return thrust::device_ptr<T>(__gpu_memory);}
    T* begin(){return __gpu_memory;}
    thrust::device_ptr<T> begin_thr(){return thrust::device_ptr<T>(__gpu_memory);}
    T* end(){return __gpu_memory + __size;}
    thrust::device_ptr<T> end_thr(){return thrust::device_ptr<T>(__gpu_memory + __size);}

    size_t size(){return __size;}

    void __malloc(size_t size);
    void __free();
};

extern template class device_vector_holder<int>;
extern template class device_vector_holder<Model::Triangle>;
#endif

std::vector<Model::mat4x4> mat_to_compact_4x4(const std::vector<cv::Mat>& poses);
Model::mat4x4 compute_proj(const cv::Mat& K, int width, int height, float near=10, float far=10000);


//roi: directly crop while rendering, expected to save much time & space
std::vector<int32_t> render_cpu(const std::vector<Model::Triangle>& tris,const std::vector<Model::mat4x4>& poses,
                            size_t width, size_t height, const Model::mat4x4& proj_mat,
                                const Model::ROI roi= {0, 0, 0, 0});

std::vector<cv::Mat> raw2depth_uint16_cpu(std::vector<int32_t>& raw_data, size_t width, size_t height, size_t pose_size);
std::vector<cv::Mat> raw2mask_uint8_cpu(std::vector<int32_t>& raw_data, size_t width, size_t height, size_t pose_size);
std::vector<std::vector<cv::Mat>> raw2depth_mask_cpu(std::vector<int32_t>& raw_data, size_t width, size_t height, size_t pose_size);

#ifdef CUDA_ON
std::vector<int32_t> render_cuda(const std::vector<Model::Triangle>& tris,const std::vector<Model::mat4x4>& poses,
                            size_t width, size_t height, const Model::mat4x4& proj_mat,
                                 const Model::ROI roi= {0, 0, 0, 0});

std::vector<int32_t> render_cuda(device_vector_holder<Model::Triangle>& tris,const std::vector<Model::mat4x4>& poses,
                            size_t width, size_t height, const Model::mat4x4& proj_mat,
                                 const Model::ROI roi= {0, 0, 0, 0});

device_vector_holder<int> render_cuda_keep_in_gpu(const std::vector<Model::Triangle>& tris,const std::vector<Model::mat4x4>& poses,
                            size_t width, size_t height, const Model::mat4x4& proj_mat,
                                                       const Model::ROI roi= {0, 0, 0, 0});

device_vector_holder<int> render_cuda_keep_in_gpu(device_vector_holder<Model::Triangle>& tris,const std::vector<Model::mat4x4>& poses,
                            size_t width, size_t height, const Model::mat4x4& proj_mat,
                                                       const Model::ROI roi= {0, 0, 0, 0});

std::vector<cv::Mat> raw2depth_uint16_cuda(device_vector_holder<int>& raw_data, size_t width, size_t height, size_t pose_size);
std::vector<cv::Mat> raw2mask_uint8_cuda(device_vector_holder<int>& raw_data, size_t width, size_t height, size_t pose_size);
std::vector<std::vector<cv::Mat>> raw2depth_mask_cuda(device_vector_holder<int32_t>& raw_data, size_t width, size_t height, size_t pose_size);
#endif

//low_level
namespace normal_functor{  // similar to thrust
    __host__ __device__ inline
    Model::float3 minus(const Model::float3& one, const Model::float3& the_other)
    {
        return {
            one.x - the_other.x,
            one.y - the_other.y,
            one.z - the_other.z
        };
    }
    __host__ __device__ inline
    Model::float3 cross(const Model::float3& one, const Model::float3& the_other)
    {
        return {
            one.y*the_other.z - one.z*the_other.y,
            one.z*the_other.x - one.x*the_other.z,
            one.x*the_other.y - one.y*the_other.x
        };
    }
    __host__ __device__ inline
    Model::float3 normalized(const Model::float3& one)
    {
        float norm = std::sqrt(one.x*one.x+one.y*one.y+one.z*one.z);
        return {
          one.x/norm,
          one.y/norm,
          one.z/norm
        };
    }

    __host__ __device__ inline
    Model::float3 get_normal(const Model::Triangle& dev_tri)
    {
//      return normalized(cross(minus(dev_tri.v1, dev_tri.v0), minus(dev_tri.v1, dev_tri.v0)));

      // no need for normalizing?
      return (cross(minus(dev_tri.v1, dev_tri.v0), minus(dev_tri.v2, dev_tri.v0)));
    }

    __host__ __device__ inline
    bool is_back(const Model::Triangle& dev_tri){
        return normal_functor::get_normal(dev_tri).z < 0;
    }
};

__host__ __device__ inline
Model::float3 mat_mul_v(const Model::mat4x4& tran, const Model::float3& v){
    return {
        tran.a0*v.x + tran.a1*v.y + tran.a2*v.z + tran.a3,
        tran.b0*v.x + tran.b1*v.y + tran.b2*v.z + tran.b3,
        tran.c0*v.x + tran.c1*v.y + tran.c2*v.z + tran.c3,
    };
}

__host__ __device__ inline
Model::Triangle transform_triangle(const Model::Triangle& dev_tri, const Model::mat4x4& tran){
    return {
        mat_mul_v(tran, (dev_tri.v0)),
        mat_mul_v(tran, (dev_tri.v1)),
        mat_mul_v(tran, (dev_tri.v2)),
    };
}

__host__ __device__ inline
float calculateSignedArea(float* A, float* B, float* C){
    return 0.5f*((C[0]-A[0])*(B[1]-A[1]) - (B[0]-A[0])*(C[1]-A[1]));
}

__host__ __device__ inline
Model::float3 barycentric(float* A, float* B, float* C, size_t* P) {

    float float_P[2] = {float(P[0]), float(P[1])};

    auto base_inv = 1/calculateSignedArea(A, B, C);
    auto beta = calculateSignedArea(A, float_P, C)*base_inv;
    auto gamma = calculateSignedArea(A, B, float_P)*base_inv;

    return {
        1.0f-beta-gamma,
        beta,
        gamma,
    };
}

__host__ __device__ inline
float std__max(float a, float b){return (a>b)? a: b;};
__host__ __device__ inline
float std__min(float a, float b){return (a<b)? a: b;};
}
