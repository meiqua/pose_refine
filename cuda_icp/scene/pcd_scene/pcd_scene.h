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

    std::vector<int> index;
    std::vector<Node_kdtree> nodes;

    void build_tree(int max_num_pcd_in_leaf = 10);
};

class Scene_nn{
    float max_dist_diff = 0.1f; // m

    void init_Scene_nn_cpu(cv::Mat& scene_depth, Mat3x3f& scene_K, std::vector<Vec3f>& pcd_buffer,
                           std::vector<Vec3f>& normal_buffer);
};


