#pragma once

#include "../common.h"

struct Node_kdtree{
    // tree info
    int parent = -1;
    int child1 = -1;
    int child2 = -1;

    // non-leaf info
    float split_v;
    float bbox[6];  // x_min x_max y... z..
    int split_dim;

    // leaf info
    int left;
    int right;

    __device__ __host__
    bool isleaf(){
        if(child1 < 0 || child2 < 0) return true;
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


#ifdef CUDA_ON
class KDTree_cuda{
public:
    device_vector_holder<Vec3f> pcd_buffer;
    device_vector_holder<Vec3f> normal_buffer;
    device_vector_holder<Node_kdtree> nodes;

    void build_tree(int max_num_pcd_in_leaf = 10);
};
#endif

// just implement query func,
// no matter it's projective or NN
class Scene_nn{
    float max_dist_diff = 0.1f; // m
    Vec3f* pcd_ptr;  // will be passed to kernel in cuda, so just hold pointers
    Vec3f* normal_ptr;
    Node_kdtree* node_ptr;
public:
    void init_Scene_nn_cpu(cv::Mat& scene_depth, Mat3x3f& scene_K, KDTree_cpu& kdtree);

#ifdef CUDA_ON
    void init_Scene_nn_cuda(cv::Mat& scene_depth, Mat3x3f& scene_K, KDTree_cuda& kdtree);
#endif

    __device__ __host__
    void query(const Vec3f& src_pcd, Vec3f& dst_pcd, Vec3f& dst_normal, bool& valid) const {
        bool backtrack=false;
        int lastNode=-1;
        int current=0;
        int result_idx = 0;
        float cloest_dist_sq = FLT_MAX;

        Node_kdtree node_cur;

        assert(node_ptr[0].parent == -1);
        while (current >= 0) { // parent of root is -1
             node_cur = node_ptr[current];

             float diff = 0;
             if (node_cur.split_dim == 0) diff = src_pcd.x - node_cur.split_v;
             if (node_cur.split_dim == 1) diff = src_pcd.y - node_cur.split_v;
             if (node_cur.split_dim == 2) diff = src_pcd.z - node_cur.split_v;

             int best_child = node_cur.child1;
             int the_other = node_cur.child1;
             if(diff < 0) the_other = node_cur.child2;
             else best_child = node_cur.child2;

             if(!backtrack){
                 if(node_cur.isleaf()){
                     for(int i=node_cur.left; i<node_cur.right; i++){
                         float cur_dist_sq =
                                 pow2(src_pcd.x - pcd_ptr[i].x) +
                                 pow2(src_pcd.y - pcd_ptr[i].y) +
                                 pow2(src_pcd.z - pcd_ptr[i].z) ;
                         if( cur_dist_sq < cloest_dist_sq ){
                             cloest_dist_sq = cur_dist_sq;
                             result_idx = i;
                         }
                     }
                     backtrack = true;
                     lastNode = current;
                     current = node_cur.parent;
                 }else{
                     lastNode = current;
                     current = best_child; // go down
                 }
             }else{
                 float min_possible_dist_sq = 0;

                 if(src_pcd.x < node_cur.bbox[0]) min_possible_dist_sq += pow2(node_cur.bbox[0] - src_pcd.x);
            else if(src_pcd.x > node_cur.bbox[1]) min_possible_dist_sq += pow2(node_cur.bbox[1] - src_pcd.x);
                 if(src_pcd.y < node_cur.bbox[2]) min_possible_dist_sq += pow2(node_cur.bbox[2] - src_pcd.y);
            else if(src_pcd.y > node_cur.bbox[3]) min_possible_dist_sq += pow2(node_cur.bbox[3] - src_pcd.y);
                 if(src_pcd.z < node_cur.bbox[4]) min_possible_dist_sq += pow2(node_cur.bbox[4] - src_pcd.z);
            else if(src_pcd.z > node_cur.bbox[5]) min_possible_dist_sq += pow2(node_cur.bbox[5] - src_pcd.z);

                 //  the far node was NOT the last node (== not visited yet),
                 //  AND there could be a closer point in it
                 if((lastNode == best_child) && (min_possible_dist_sq <= cloest_dist_sq)){
                     lastNode=current;
                     current=the_other;
                     backtrack=false;
                 }
                 else{
                     lastNode=current;
                     current=node_cur.parent;
                 }
             }
        }

        if(cloest_dist_sq < pow2(max_dist_diff)){
            valid = true;
            dst_pcd = pcd_ptr[result_idx];
            dst_normal = normal_ptr[result_idx];
            return;
        }else{
            valid = false;
            return;
        }
    }
};


