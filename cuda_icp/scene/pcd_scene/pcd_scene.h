#pragma once

#include "../common.h"

struct Node_kdtree{
    // tree info
    int parent = -1;
    int child1 = -1;
    int child2 = -1;

    // non-leaf info
    float split_l;  // tight split value
    float split_h;
    int split_dim;

    // leaf info
    int left;
    int right;

    bool isleaf(){
        if(child1 < 0 && child2 < 0) return true;
        return false;
    }
};

class KDTree_cpu{
public:
    std::vector<Vec3f> pcd_buffer;
    std::vector<Vec3f> normal_buffer;
    std::vector<Node_kdtree> nodes;

    void build_tree(int max_num_pcd_in_leaf = 10);
};

class KDTree_cuda{
public:
    device_vector_holder<Vec3f> pcd_buffer;
    device_vector_holder<Vec3f> normal_buffer;
    device_vector_holder<Node_kdtree> nodes;
};

class Scene_nn{
    float max_dist_diff = 0.1f; // m
    Vec3f* pcd_ptr;
    Vec3f* normal_ptr;
    Node_kdtree* node_ptr;

    void init_Scene_nn_cpu(cv::Mat& scene_depth, Mat3x3f& scene_K, KDTree_cpu& kdtree);

#ifdef CUDA_ON
    void init_Scene_nn_cuda(cv::Mat& scene_depth, Mat3x3f& scene_K, KDTree_cuda& kdtree);
#endif

    __device__ __host__
    void query(const Vec3f& src_pcd, Vec3f& dst_pcd, Vec3f& dst_normal, bool& valid) const {

    }
};


