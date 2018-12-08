#include "renderer.h"
namespace cuda_renderer {

namespace normal_functor{  // similar to thrust
    __host__ __device__
    Model::float3 minus(const Model::float3& one, const Model::float3& the_other)
    {
        return {
            one.x - the_other.x,
            one.y - the_other.y,
            one.z - the_other.z
        };
    }
    __host__ __device__
    Model::float3 cross(const Model::float3& one, const Model::float3& the_other)
    {
        return {
            one.y*the_other.z - one.z*the_other.y,
            one.z*the_other.x - one.x*the_other.z,
            one.x*the_other.y - one.y*the_other.x
        };
    }
    __host__ __device__
    Model::float3 normalized(const Model::float3& one)
    {
        float norm = std::sqrt(one.x*one.x+one.y*one.y+one.z*one.z);
        return {
          one.x/norm,
          one.y/norm,
          one.z/norm
        };
    }

    __host__ __device__
    Model::float3 get_normal(const Model::Triangle& dev_tri)
    {
//      return normalized(cross(minus(dev_tri.v1, dev_tri.v0), minus(dev_tri.v1, dev_tri.v0)));

      // no need for normalizing?
      return (cross(minus(dev_tri.v1, dev_tri.v0), minus(dev_tri.v2, dev_tri.v0)));
    }

    __host__ __device__
    bool is_back(const Model::Triangle& dev_tri){
        return normal_functor::get_normal(dev_tri).z < 0;
    }
};

__host__ __device__
Model::float3 mat_mul_v(const Model::mat4x4& tran, const Model::float3& v){
    return {
        tran.a0*v.x + tran.a1*v.y + tran.a2*v.z + tran.a3,
        tran.b0*v.x + tran.b1*v.y + tran.b2*v.z + tran.b3,
        tran.c0*v.x + tran.c1*v.y + tran.c2*v.z + tran.c3,
    };
}

__host__ __device__
Model::Triangle transform_triangle(const Model::Triangle& dev_tri, const Model::mat4x4& tran){
    return {
        mat_mul_v(tran, (dev_tri.v0)),
        mat_mul_v(tran, (dev_tri.v1)),
        mat_mul_v(tran, (dev_tri.v2)),
    };
}

__host__ __device__
float calculateSignedArea(float* A, float* B, float* C){
    return 0.5f*((C[0]-A[0])*(B[1]-A[1]) - (B[0]-A[0])*(C[1]-A[1]));
}

__host__ __device__
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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void print_cuda_memory_usage(){
    // show memory usage of GPU

    size_t free_byte ;
    size_t total_byte ;
    auto cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

    if ( cudaSuccess != cuda_status ){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
    }

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

struct max2zero_functor{

    max2zero_functor(){}

    __host__ __device__
    int32_t operator()(const int32_t& x) const
    {
      return (x==INT_MAX)? 0: x;
    }
};


__device__
void rasterization(const Model::Triangle dev_tri, Model::float3 last_row,
                                        int32_t* depth_entry, size_t width, size_t height, const Model::ROI roi){
    // refer to tiny renderer
    // https://github.com/ssloy/tinyrenderer/blob/master/our_gl.cpp
    float pts2[3][2];

    // viewport transform(0, 0, width, height)
    pts2[0][0] = dev_tri.v0.x/last_row.x*width/2.0f+width/2.0f; pts2[0][1] = dev_tri.v0.y/last_row.x*height/2.0f+height/2.0f;
    pts2[1][0] = dev_tri.v1.x/last_row.y*width/2.0f+width/2.0f; pts2[1][1] = dev_tri.v1.y/last_row.y*height/2.0f+height/2.0f;
    pts2[2][0] = dev_tri.v2.x/last_row.z*width/2.0f+width/2.0f; pts2[2][1] = dev_tri.v2.y/last_row.z*height/2.0f+height/2.0f;

    float bboxmin[2] = {FLT_MAX,  FLT_MAX};
    float bboxmax[2] = {-FLT_MAX, -FLT_MAX};

    float clamp_max[2] = {float(width-1), float(height-1)};
    float clamp_min[2] = {0, 0};

    size_t real_width = width;
    if(roi.width > 0 && roi.height > 0){  // depth will be flipped
        clamp_min[0] = width - (roi.x + roi.width) + 1; // +1 avoid out of roi
        clamp_min[1] = height - (roi.y + roi.height) + 1;
        clamp_max[0] = width - roi.x;
        clamp_max[1] = height - roi.y;
        real_width = roi.width;
    }


    for (int i=0; i<3; i++) {
        for (int j=0; j<2; j++) {
            bboxmin[j] = std__max(clamp_min[j], std__min(bboxmin[j], pts2[i][j]));
            bboxmax[j] = std__min(clamp_max[j], std__max(bboxmax[j], pts2[i][j]));
        }
    }

    size_t P[2];
    for(P[1] = size_t(bboxmin[1]+0.5f); P[1]<=bboxmax[1]; P[1] += 1){
        for(P[0] = size_t(bboxmin[0]+0.5f); P[0]<=bboxmax[0]; P[0] += 1){
            Model::float3 bc_screen  = barycentric(pts2[0], pts2[1], pts2[2], P);

            if (bc_screen.x<-0.0f || bc_screen.y<-0.0f || bc_screen.z<-0.0f ||
                    bc_screen.x>1.0f || bc_screen.y>1.0f || bc_screen.z>1.0f ) continue;

            Model::float3 bc_over_z = {bc_screen.x/last_row.x, bc_screen.y/last_row.y, bc_screen.z/last_row.z};

            // refer to https://en.wikibooks.org/wiki/Cg_Programming/Rasterization, Perspectively Correct Interpolation
            float frag_depth = -(dev_tri.v0.z*bc_over_z.x + dev_tri.v1.z*bc_over_z.y + dev_tri.v2.z*bc_over_z.z)
                    /(bc_over_z.x + bc_over_z.y + bc_over_z.z);

            size_t x_to_write = (width - P[0] - roi.x);
            size_t y_to_write = (height - P[1] - roi.y);

            int32_t depth = int32_t(frag_depth/**1000*/ + 0.5f);
            int32_t& depth_to_write = depth_entry[x_to_write+y_to_write*real_width];

            atomicMin(&depth_to_write, depth);
        }
    }
}

__global__ void render_triangle(Model::Triangle* device_tris_ptr, size_t device_tris_size,
                                Model::mat4x4* device_poses_ptr, size_t device_poses_size,
                                int32_t* depth_image_vec, size_t width, size_t height, const Model::mat4x4 proj_mat,
                                 const Model::ROI roi){
    size_t pose_i = blockIdx.y;
    size_t tri_i = blockIdx.x*blockDim.x + threadIdx.x;

    if(tri_i>=device_tris_size) return;
//    if(pose_i>=device_poses_size) return;

    size_t real_width = width;
    size_t real_height = height;
    if(roi.width > 0 && roi.height > 0){
        real_width = roi.width;
        real_height = roi.height;
    }

    int32_t* depth_entry = depth_image_vec + pose_i*real_width*real_height; //length: width*height 32bits int
    Model::mat4x4* pose_entry = device_poses_ptr + pose_i; // length: 16 32bits float
    Model::Triangle* tri_entry = device_tris_ptr + tri_i; // length: 9 32bits float

    // model transform
    Model::Triangle local_tri = transform_triangle(*tri_entry, *pose_entry);
//    if(normal_functor::is_back(local_tri)) return; //back face culling, need to be disable for not well defined surfaces?

    // assume last column of projection matrix is  0 0 -1 0
    Model::float3 last_row = {
        -local_tri.v0.z,
        -local_tri.v1.z,
        -local_tri.v2.z
    };
    // projection transform
    local_tri = transform_triangle(local_tri, proj_mat);

    rasterization(local_tri, last_row, depth_entry, width, height, roi);
}

std::vector<int32_t> render_cuda(const std::vector<Model::Triangle>& tris,const std::vector<Model::mat4x4>& poses,
                            size_t width, size_t height, const Model::mat4x4& proj_mat, const Model::ROI roi){

    const size_t threadsPerBlock = 256;

    thrust::device_vector<Model::Triangle> device_tris = tris;
    thrust::device_vector<Model::mat4x4> device_poses = poses;

    size_t real_width = width;
    size_t real_height = height;
    if(roi.width > 0 && roi.height > 0){
        real_width = roi.width;
        real_height = roi.height;
        assert(roi.x + roi.width <= width && "roi out of image");
        assert(roi.y + roi.height <= height && "roi out of image");
    }
    // atomic min only support int32
    thrust::device_vector<int32_t> device_depth_int(poses.size()*real_width*real_height, INT_MAX);
    {
        Model::Triangle* device_tris_ptr = thrust::raw_pointer_cast(device_tris.data());
        Model::mat4x4* device_poses_ptr = thrust::raw_pointer_cast(device_poses.data());
        int32_t* depth_image_vec = thrust::raw_pointer_cast(device_depth_int.data());

        dim3 numBlocks((tris.size() + threadsPerBlock - 1) / threadsPerBlock, poses.size());
        render_triangle<<<numBlocks, threadsPerBlock>>>(device_tris_ptr, tris.size(),
                                                        device_poses_ptr, poses.size(),
                                                        depth_image_vec, width, height, proj_mat, roi);
        cudaThreadSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }

    std::vector<int32_t> result_depth(poses.size()*real_width*real_height);
    {
        thrust::transform(device_depth_int.begin(), device_depth_int.end(),
                          device_depth_int.begin(), max2zero_functor());
        thrust::copy(device_depth_int.begin(), device_depth_int.end(), result_depth.begin());
    }

    return result_depth;
}

thrust::device_vector<int32_t> render_cuda_keep_in_gpu(const std::vector<Model::Triangle>& tris,const std::vector<Model::mat4x4>& poses,
                            size_t width, size_t height, const Model::mat4x4& proj_mat, const Model::ROI roi){

    const size_t threadsPerBlock = 256;

    thrust::device_vector<Model::Triangle> device_tris = tris;
    thrust::device_vector<Model::mat4x4> device_poses = poses;

    size_t real_width = width;
    size_t real_height = height;
    if(roi.width > 0 && roi.height > 0){
        real_width = roi.width;
        real_height = roi.height;
    }
    // atomic min only support int32
    thrust::device_vector<int32_t> device_depth_int(poses.size()*real_width*real_height, INT_MAX);
    {
        Model::Triangle* device_tris_ptr = thrust::raw_pointer_cast(device_tris.data());
        Model::mat4x4* device_poses_ptr = thrust::raw_pointer_cast(device_poses.data());
        int32_t* depth_image_vec = thrust::raw_pointer_cast(device_depth_int.data());

        dim3 numBlocks((tris.size() + threadsPerBlock - 1) / threadsPerBlock, poses.size());
        render_triangle<<<numBlocks, threadsPerBlock>>>(device_tris_ptr, tris.size(),
                                                        device_poses_ptr, poses.size(),
                                                        depth_image_vec, width, height, proj_mat, roi);
        cudaThreadSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }

    thrust::transform(device_depth_int.begin(), device_depth_int.end(),
                      device_depth_int.begin(), max2zero_functor());

    return device_depth_int;
}

}

