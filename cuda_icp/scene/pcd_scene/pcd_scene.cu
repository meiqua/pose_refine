#include "pcd_scene.h"

void Scene_nn::init_Scene_nn_cuda(cv::Mat &scene_depth, Mat3x3f &scene_K, KDTree_cuda &kdtree)
{
    KDTree_cpu cpu_tree;
    init_Scene_nn_cpu(scene_depth, scene_K, cpu_tree);

    kdtree.pcd_buffer.__malloc(cpu_tree.pcd_buffer.size());
    thrust::copy(cpu_tree.pcd_buffer.begin(), cpu_tree.pcd_buffer.end(), kdtree.pcd_buffer.begin_thr());

    kdtree.normal_buffer.__malloc(cpu_tree.normal_buffer.size());
    thrust::copy(cpu_tree.normal_buffer.begin(), cpu_tree.normal_buffer.end(), kdtree.normal_buffer.begin_thr());

    // kdtree.nodes.__malloc(cpu_tree.nodes.size());
    // thrust::copy(cpu_tree.nodes.begin(), cpu_tree.nodes.end(), kdtree.nodes.begin_thr());
    kdtree.build_tree();

    pcd_ptr = kdtree.pcd_buffer.data();
    normal_ptr = kdtree.normal_buffer.data();
    node_ptr = kdtree.nodes.data();
}

__global__ void iota_dev(int* idx_ptr, size_t idx_size, int init_val){
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= idx_size) return ;
    idx_ptr[i] = i + init_val;
}

__global__ void BFS_tree_grow(Node_kdtree* nodes, int start, int end, int max_num_pcd_in_leaf,
                         int* num_nodes_now_ptr, Vec3f* pcd_buffer, int* index, int* index_buffer){
    uint32_t node_iter = blockIdx.x*blockDim.x + threadIdx.x + start;
    if(node_iter >= end) return ;

    if(nodes[node_iter].right - nodes[node_iter].left > max_num_pcd_in_leaf){
        // split start <----------------------
        // get bbox
        float x_min = FLT_MAX; float x_max = -FLT_MAX;  // fxxk, FLT_MIN is 0
        float y_min = FLT_MAX; float y_max = -FLT_MAX;
        float z_min = FLT_MAX; float z_max = -FLT_MAX;
        for(int idx_iter = nodes[node_iter].left; idx_iter < nodes[node_iter].right; idx_iter++){
            const auto& p = pcd_buffer[index[idx_iter]];
            if(p.x > x_max) x_max = p.x;
            if(p.x < x_min) x_min = p.x;
            if(p.y > y_max) y_max = p.y;
            if(p.y < y_min) y_min = p.y;
            if(p.z > z_max) z_max = p.z;
            if(p.z < z_min) z_min = p.z;
        }

        // select split dim & value
        int split_dim = 0;
        float split_val = 0;
        float span_xyz[3], split_v_xyz[3];
        float max_span = -FLT_MAX;
        span_xyz[0] = x_max - x_min; split_v_xyz[0] = (x_min + x_max)/2;
        span_xyz[1] = y_max - y_min; split_v_xyz[1] = (y_min + y_max)/2;
        span_xyz[2] = z_max - z_min; split_v_xyz[2] = (z_min + z_max)/2;
        for(int span_iter=0; span_iter<3; span_iter++){
            if(span_xyz[span_iter] > max_span){
                max_span = span_xyz[span_iter];
                split_dim = span_iter;
                split_val = split_v_xyz[span_iter];
            }
        }

        // reorder index
        int left_iter = nodes[node_iter].left;
        int right_iter = nodes[node_iter].right - 1;
        float split_low = -FLT_MAX;
        float split_high = FLT_MAX;
        
        bool lr_switch = true;
        for(int idx_iter = nodes[node_iter].left; idx_iter<nodes[node_iter].right; idx_iter++){
            float p = pcd_buffer[index[idx_iter]][split_dim];

            if(p == split_val) lr_switch = !lr_switch;

            if(p < split_val || (p==split_val && lr_switch)){
                index_buffer[left_iter] = index[idx_iter];
                left_iter ++;
                if(p > split_low) split_low = p;
            }else{
                index_buffer[right_iter] = index[idx_iter];
                right_iter --;
                if(p < split_high) split_high = p;
            }
        }
        assert(left_iter == right_iter + 1 && "left & right should meet");
        split_val = (split_low + split_high)/2; // reset split_val to middle

        for(int idx_iter = nodes[node_iter].left; idx_iter<nodes[node_iter].right; idx_iter++){
            index[idx_iter] = index_buffer[idx_iter];
        }
        // split success <----------------------


        int num_nodes_now = atomicAdd(num_nodes_now_ptr, 2); // fetch and add

        // update parent
        nodes[node_iter].child1 = num_nodes_now;
        nodes[node_iter].child2 = num_nodes_now + 1;
        nodes[node_iter].split_v = split_val;
        nodes[node_iter].split_dim = split_dim;
        nodes[node_iter].bbox[0] = x_min;  nodes[node_iter].bbox[1] = x_max;
        nodes[node_iter].bbox[2] = y_min;  nodes[node_iter].bbox[3] = y_max;
        nodes[node_iter].bbox[4] = z_min;  nodes[node_iter].bbox[5] = z_max;

        // update child
        nodes[num_nodes_now].left = nodes[node_iter].left;
        nodes[num_nodes_now].right = left_iter;
        nodes[num_nodes_now].parent = node_iter;

        nodes[num_nodes_now + 1].left = left_iter;
        nodes[num_nodes_now + 1].right = nodes[node_iter].right;
        nodes[num_nodes_now + 1].parent = node_iter;
    }
}

__global__ void reorder_v3f(Vec3f* begin, Vec3f* result, int* index, int n){
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return ;
    result[i] = begin[index[i]];
}

void KDTree_cuda::build_tree(int max_num_pcd_in_leaf)
{
    assert(pcd_buffer.size() > 0 && pcd_buffer.size() == normal_buffer.size()
           && "no pcd yet, or pcd size != normal size");

    thrust::device_vector<Node_kdtree> nodes_dev;
    thrust::device_vector<int> index_dev(pcd_buffer.size());
    thrust::device_vector<int> index_buffer_dev(pcd_buffer.size());

    Node_kdtree* nodes_ptr = thrust::raw_pointer_cast(nodes_dev.data());
    int* index_ptr = thrust::raw_pointer_cast(index_dev.data());
    int* index_buffer_ptr = thrust::raw_pointer_cast(index_buffer_dev.data());

    const int threadsPerBlock = 256;
    int numBlocks = (pcd_buffer.size() + threadsPerBlock - 1)/threadsPerBlock;

    // std::iota (std::begin(index), std::end(index), 0); // Fill with 0, 1, 2, ...
    iota_dev<<<numBlocks, threadsPerBlock>>>(index_ptr, pcd_buffer.size(), 0);
    cudaStreamSynchronize(cudaStreamPerThread);

    // reserve a plausible size to avoid allocing too many times
    nodes_dev.reserve(mylog2(pcd_buffer.size()) + 10);

    //root
    nodes_dev.resize(1);
    Node_kdtree root_host;
    root_host.left = 0;
    root_host.right = pcd_buffer.size();
    nodes_dev[0] = root_host;

    size_t num_nodes_last_last_turn = 0;
    size_t num_nodes_now_last_turn = 0;
    bool stop = false;

    thrust::host_vector<int> num_nodes_now_host(1, 1);
    thrust::device_vector<int> num_nodes_now_dev(1, 1);
    int * num_nodes_now_ptr = thrust::raw_pointer_cast(num_nodes_now_dev.data());

    while (!stop) { // when we have new nodes to split, go

        nodes_dev.resize(num_nodes_now_host[0]*2+1); // we may increase now + 1 in 1 turn
        Node_kdtree* nodes_ptr = thrust::raw_pointer_cast(nodes_dev.data());

        num_nodes_last_last_turn = num_nodes_now_last_turn;
        num_nodes_now_last_turn = num_nodes_now_host[0]; // for iter, avoid reaching new node in 1 turn

        numBlocks = (num_nodes_now_last_turn - num_nodes_last_last_turn + threadsPerBlock - 1)/threadsPerBlock;
        BFS_tree_grow<<<numBlocks, threadsPerBlock>>>(nodes_ptr, num_nodes_last_last_turn,
                                                      num_nodes_now_last_turn, max_num_pcd_in_leaf,
                                                      num_nodes_now_ptr, pcd_buffer.data(),
                                                      index_ptr, index_buffer_ptr);
        cudaStreamSynchronize(cudaStreamPerThread);

        num_nodes_now_host = num_nodes_now_dev;

        if(num_nodes_now_host[0] == num_nodes_now_last_turn){ // no new nodes
            stop = true;
        }
    }

    // we may give nodes more memory while spliting
    nodes_dev.resize(num_nodes_now_host[0]);
    nodes.__malloc(num_nodes_now_host[0]);
    thrust::copy(thrust::cuda::par.on(cudaStreamPerThread),
                 nodes_dev.begin(), nodes_dev.end(), nodes.begin_thr());

    // reorder pcd normal according to index, so avoid using index when query
    thrust::device_vector<Vec3f> v3f_buffer(pcd_buffer.size());
    Vec3f* v3f_buffer_ptr = thrust::raw_pointer_cast(v3f_buffer.data());
    numBlocks = (pcd_buffer.size() + threadsPerBlock - 1)/threadsPerBlock;

    reorder_v3f<<<numBlocks, threadsPerBlock>>>(pcd_buffer.data(), v3f_buffer_ptr, index_ptr, pcd_buffer.size());
    cudaStreamSynchronize(cudaStreamPerThread);
    thrust::copy(thrust::cuda::par.on(cudaStreamPerThread),
                 v3f_buffer.begin(), v3f_buffer.end(), pcd_buffer.begin_thr());

    reorder_v3f<<<numBlocks, threadsPerBlock>>>(normal_buffer.data(), v3f_buffer_ptr, index_ptr, normal_buffer.size());
    cudaStreamSynchronize(cudaStreamPerThread);
    thrust::copy(thrust::cuda::par.on(cudaStreamPerThread),
                 v3f_buffer.begin(), v3f_buffer.end(), normal_buffer.begin_thr());

}
