#include "helper.h"
using namespace cv;
using namespace std;

static std::string prefix = "/home/meiqua/pose_refine/test/";

//#define USE_PROJ

#ifdef CUDA_ON
void test_cuda_icp(){

    {  // gpu need sometime to warm up
        cudaFree(0);
//        cudaSetDevice(0);

        // cublas also need
//        cublasStatus_t stat;  // CUBLAS functions status
        cublasHandle_t cublas_handle;  // CUBLAS context
        /*stat = */cublasCreate(&cublas_handle);
    }

    int width = 640; int height = 480;

    cuda_renderer::Model model(prefix+"obj_06.ply");

    Mat K = (Mat_<float>(3,3) << 572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0);
    auto proj = cuda_renderer::compute_proj(K, width, height);

    Mat R_ren = (Mat_<float>(3,3) << 0.34768538, 0.93761126, 0.00000000, 0.70540612,
                 -0.26157897, -0.65877056, -0.61767070, 0.22904489, -0.75234390);
    Mat t_ren = (Mat_<float>(3,1) << 0.0, 0.0, 300.0);
    Mat t_ren2 = (Mat_<float>(3,1) << 20.0, 20.0, 320.0);

    float angle_y = 10.0f/180.0f*3.14f;
    float angle_z = angle_y;
    float angle_x = angle_y;
    Mat rot_mat = helper::eulerAnglesToRotationMatrix({angle_x, angle_y, angle_z,});
    cout << "init angle diff y: " << angle_y*180/3.14f << endl << endl;

    Mat R_ren2 = rot_mat * R_ren;

    cuda_renderer::Model::mat4x4 mat4, mat4_2;
    mat4.init_from_cv(R_ren, t_ren);
    mat4_2.init_from_cv(R_ren2, t_ren2);

    std::vector<cuda_renderer::Model::mat4x4> mat4_v = {mat4, mat4_2};

    helper::Timer timer;

    std::vector<int> depth_cpu = cuda_renderer::render_cpu(model.tris, mat4_v, width, height, proj);
    timer.out("cpu render");

    cv::Mat depth_1 = cv::Mat(height, width, CV_32SC1, depth_cpu.data());
    cv::Mat depth_2 = cv::Mat(height, width, CV_32SC1, depth_cpu.data() + height*width);

    auto bbox1 = helper::get_bbox(depth_1);
    auto bbox2 = helper::get_bbox(depth_2);
    cout << "\nbbox:" << endl;
    cout << "depth 1: " << bbox1 << endl;
    cout << "depth 2: " << bbox2 << endl;
    cout << "init pixel diff xy: "
         << abs(bbox1.x - bbox2.x) << "----" <<  abs(bbox1.y - bbox2.y) << endl << endl;

//    cv::imshow("depth_1", helper::view_dep(depth_1));
//    cv::imshow("depth_2", helper::view_dep(depth_2));
//    cv::waitKey(0);

    Mat3x3f K_((float*)K.data); // ugly but useful
//    cout << K << endl;
//    cout << K_ << endl;
timer.reset();
    std::vector<::Vec3f> pcd1 = cuda_icp::depth2cloud_cpu(depth_cpu.data(), width, height, K_);
//    helper::view_pcd(pcd1);
timer.out("depth2cloud_cpu");
    cv::Mat scene_depth(height, width, CV_32S, depth_cpu.data() + width*height);

timer.reset();
#ifdef USE_PROJ
    Scene_projective scene;
    vector<::Vec3f> pcd_buffer, normal_buffer;
    scene.init_Scene_projective_cpu(scene_depth, K_, pcd_buffer, normal_buffer);
#else
    Scene_nn scene;
    KDTree_cpu kdtree_cpu;
    scene.init_Scene_nn_cpu(scene_depth, K_, kdtree_cpu);
#endif
timer.out("init scene cpu");

#ifdef USE_PROJ
    //view init cloud; the far point is 0 in scene
//    helper::view_pcd(pcd1, pcd_buffer);
#else
//    helper::view_pcd(pcd1, kdtree_cpu.pcd_buffer);
#endif

    {  // open3d
        open3d::geometry::PointCloud model_pcd, scene_pcd;
        for(auto& p: pcd1){
            if(p.z > 0)
            model_pcd.points_.emplace_back(float(p.x), float(p.y), float(p.z));
        }
#ifndef USE_PROJ
        for(auto& p: kdtree_cpu.pcd_buffer)
#else
        for(auto& p: pcd_buffer)
#endif
        {
            if(p.z > 0)
            scene_pcd.points_.emplace_back(float(p.x), float(p.y), float(p.z));
        }

        open3d::geometry::EstimateNormals(scene_pcd);
        open3d::geometry::EstimateNormals(model_pcd);

        timer.reset();
        auto final_result = open3d::registration::RegistrationICP(model_pcd, scene_pcd, 0.1,
                                                    Eigen::Matrix4d::Identity(4, 4),
                                                    open3d::registration::TransformationEstimationPointToPlane());
        timer.out("open3d icp");

        model_pcd.Transform(final_result.transformation_);
        cout << "open3d final rmse: " << final_result.inlier_rmse_ << endl;
        cout << "open3d final fitness: " << final_result.fitness_ << endl;
        cout << "open3d final transformation_:\n" << final_result.transformation_ << endl << endl;
//        helper::view_pcd(model_pcd, scene_pcd);
    }

timer.reset();
    auto result = cuda_icp::ICP_Point2Plane_cpu(pcd1, scene);  // notice, pcd1 are changed due to icp
timer.out("ICP_Point2Plane_cpu");
    Mat result_cv = helper::mat4x4f2cv(result.transformation_);
    Mat R = result_cv(cv::Rect(0, 0, 3, 3));
    auto R_v = helper::rotationMatrixToEulerAngles(R);

    //view icp cloud
#ifdef USE_PROJ
//    helper::view_pcd(pcd_buffer, pcd1);
#else
//    helper::view_pcd(kdtree_cpu.pcd_buffer, pcd1);
#endif

timer.reset();
    auto depth_cuda = cuda_renderer::render_cuda_keep_in_gpu(model.tris, mat4_v, width, height, proj);
timer.out("gpu render");
    // view gpu depth
//    vector<int> depth_host(depth_cuda.size());
//    thrust::copy(depth_cuda.begin_thr(), depth_cuda.end_thr(), depth_host.begin());
//    cv::Mat depth_1_cuda = cv::Mat(height, width, CV_32SC1, depth_host.data());
//    imshow("depth 1 cuda", helper::view_dep(depth_1_cuda));
//    waitKey(0);

timer.reset();
    auto pcd1_cuda = cuda_icp::depth2cloud_cuda(depth_cuda.data(), width, height, K_);
timer.out("depth2cloud_cuda");
// view gpu pcd
//    std::vector<::Vec3f> pcd1_host(pcd1_cuda.size());
//    thrust::copy(pcd1_cuda.begin_thr(), pcd1_cuda.end_thr(), pcd1_host.begin());
//    helper::view_pcd(pcd1_host);

timer.reset();
#ifdef USE_PROJ
    device_vector_holder<::Vec3f> pcd_buffer_cuda, normal_buffer_cuda;
    scene.init_Scene_projective_cuda(scene_depth, K_, pcd_buffer_cuda, normal_buffer_cuda);
#else
    KDTree_cuda kdtree_cuda;
    scene.init_Scene_nn_cuda(scene_depth, K_, kdtree_cuda);
#endif
timer.out("init scene cuda");


timer.reset();
    auto result_cuda = cuda_icp::ICP_Point2Plane_cuda(pcd1_cuda, scene);
timer.out("ICP_Point2Plane_cuda");
    Mat result_cv_cuda = helper::mat4x4f2cv(result_cuda.transformation_);


    cout << "\nresult: " << endl;
    cout << "result fitness: " << result.fitness_ << endl;
    cout << "result mse: " << result.inlier_rmse_ << endl;
    cout << "\nresult_cv:" << endl;
    cout << result_cv << endl;

    cout << "\nresult_cuda: " << endl;
    cout << "result fitness: " << result_cuda.fitness_ << endl;
    cout << "result mse: " << result_cuda.inlier_rmse_ << endl;
    cout << "\nresult_cv_cuda:" << endl;
    cout << result_cv_cuda << endl;

    cout << "\nerror in degree:" << endl;
    cout << "x: " << abs(R_v[0] - angle_x)/3.14f*180  << endl;
    cout << "y: " << abs(R_v[1] - angle_y)/3.14f*180 << endl;
    cout << "z: " << abs(R_v[2] - angle_z)/3.14f*180  << endl;
}
#endif

int main(int argc, char const *argv[]){

#ifdef CUDA_ON
    test_cuda_icp();
#endif

    return 0;
}
