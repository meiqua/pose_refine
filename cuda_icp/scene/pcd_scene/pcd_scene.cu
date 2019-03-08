#include "pcd_scene.h"

void Scene_nn::init_Scene_nn_cuda(cv::Mat &scene_depth, Mat3x3f &scene_K, KDTree_cuda &kdtree)
{
    KDTree_cpu cpu_tree;
    init_Scene_nn_cpu(scene_depth, scene_K, cpu_tree);

    kdtree.pcd_buffer.__malloc(cpu_tree.pcd_buffer.size());
    thrust::copy(cpu_tree.pcd_buffer.begin(), cpu_tree.pcd_buffer.end(), kdtree.pcd_buffer.begin_thr());

    kdtree.normal_buffer.__malloc(cpu_tree.normal_buffer.size());
    thrust::copy(cpu_tree.normal_buffer.begin(), cpu_tree.normal_buffer.end(), kdtree.normal_buffer.begin_thr());

    kdtree.nodes.__malloc(cpu_tree.nodes.size());
    thrust::copy(cpu_tree.nodes.begin(), cpu_tree.nodes.end(), kdtree.nodes.begin_thr());

    pcd_ptr = kdtree.pcd_buffer.data();
    normal_ptr = kdtree.normal_buffer.data();
    node_ptr = kdtree.nodes.data();
}
