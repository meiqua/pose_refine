#include "pcd_scene.h"


void Scene_nn::init_Scene_nn_cpu(cv::Mat &scene_depth__, Mat3x3f &scene_K, std::vector<Vec3f>& pcd_buffer,
                                 std::vector<Vec3f>& normal_buffer)
{
    device_vector_holder<int> test(1);

    int depth_type = scene_depth__.type();
    assert(depth_type == CV_16U || depth_type == CV_32S);

    cv::Mat scene_depth;
    if(depth_type == CV_32S){
        scene_depth__.convertTo(scene_depth, CV_16U);
    }else{
        scene_depth = scene_depth__;
    }

    auto normal = get_normal(scene_depth, scene_K);
    pcd_buffer.clear();
    pcd_buffer.reserve(scene_depth.rows * scene_depth.cols);
    normal_buffer.clear();
    normal_buffer.reserve(scene_depth.rows * scene_depth.cols);

    for(int r=0; r<scene_depth.rows; r++){
        for(int c=0; c<scene_depth.cols; c++){
            auto& dep_at_rc = scene_depth.at<uint16_t>(r, c);
            if(dep_at_rc > 0){
                pcd_buffer.push_back(dep2pcd(c, r, dep_at_rc, scene_K));
                normal_buffer.push_back(normal[c + r*scene_depth.cols]);
            }
        }
    }
}

void KDTree_cpu::build_tree(int max_num_pcd_in_leaf)
{
    assert(pcd_buffer.size() > 0 && pcd_buffer.size() == normal_buffer.size()
           && "no pcd yet, or pcd size != normal size");

    index.resize(pcd_buffer.size());
    std::iota (std::begin(index), std::end(index), 0); // Fill with 0, 1, 2, ...

    std::vector<int> index_buffer(index.size());

    //root
    nodes.resize(1);
    nodes[0].left = 0;
    nodes[0].right = index.size();
    std::vector<uint8_t> splited(1, 0); // mark if nodes are splited

    size_t new_split_num = 0;
    size_t num_nodes_now = 1;
    bool stop = false;

    while (!stop) { // when we have new nodes to split, go

        nodes.resize(num_nodes_now*2+1); // we may increase now + 1 in 1 turn
        splited.resize(nodes.size());
        std::fill(splited.begin() + num_nodes_now, splited.end(), 0);

        new_split_num = 0; // reset new split num
        size_t num_nodes_old = num_nodes_now; // for iter, avoid reaching new node in 1 turn

        // search all the tree, we search one times more, but avoid using a stack instead
        for(size_t node_iter = 0; node_iter < num_nodes_old; node_iter++){

            if(splited[node_iter] == 0){ // not splited yet
                // not a leaf
                if(nodes[node_iter].right - nodes[node_iter].left > max_num_pcd_in_leaf){

                    // split start <----------------------
                    // get bbox
                    float x_min = FLT_MAX; float x_max = FLT_MIN;
                    float y_min = FLT_MAX; float y_max = FLT_MIN;
                    float z_min = FLT_MAX; float z_max = FLT_MIN;
                    for(int idx_iter = nodes[node_iter].left; idx_iter < nodes[node_iter].right; idx_iter++){
                        const auto& p = pcd_buffer[index[idx_iter]];
                        if(p.x > x_max) x_max = p.x; if(p.x < x_min) x_min = p.x;
                        if(p.y > y_max) y_max = p.y; if(p.y < y_min) y_min = p.y;
                        if(p.z > z_max) z_max = p.z; if(p.z < z_min) z_min = p.z;
                    }

                    // select split dim & value
                    int split_dim = 0;
                    float split_val = 0;
                    float span_xyz[3], split_v_xyz[3];
                    float max_span = FLT_MIN;
                    span_xyz[0] = x_max - x_min; split_v_xyz[0] = x_min + span_xyz[0]/2;
                    span_xyz[1] = y_max - y_min; split_v_xyz[1] = y_min + span_xyz[1]/2;
                    span_xyz[2] = z_max - z_min; split_v_xyz[2] = z_min + span_xyz[2]/2;
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
                    float split_low = FLT_MIN;
                    float split_high = FLT_MAX;

                    for(int idx_iter = nodes[node_iter].left; idx_iter<nodes[node_iter].right; idx_iter++){
                        float p = pcd_buffer[index[idx_iter]][split_dim];
                        if(p < split_val){
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

                    for(int idx_iter = nodes[node_iter].left; idx_iter<nodes[node_iter].right; idx_iter++){
                        index[idx_iter] = index_buffer[idx_iter];
                    }

                    splited[node_iter] = 1;
                    // split success <----------------------

                    // update parent
                    nodes[node_iter].child1 = node_iter + 1;
                    nodes[node_iter].child2 = node_iter + 2;
                    nodes[node_iter].split_l = split_low;
                    nodes[node_iter].split_h = split_high;
                    nodes[node_iter].split_dim = split_dim;

                    // update child
                    nodes[node_iter + 1].left = nodes[node_iter].left;
                    nodes[node_iter + 1].right = left_iter;
                    nodes[node_iter + 1].parent = node_iter;

                    nodes[node_iter + 2].left = left_iter;
                    nodes[node_iter + 2].right = nodes[node_iter].right;
                    nodes[node_iter + 2].parent = node_iter;

                    num_nodes_now += 2;
                    new_split_num ++;
                }
            }
        }

        if(new_split_num == 0){
            stop = true;
        }
    }

    // we may give nodes more memory while spliting
    nodes.resize(num_nodes_now);
}
