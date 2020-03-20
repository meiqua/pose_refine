#include "pose_renderer.h"

PoseRenderer::PoseRenderer(std::string model_path, cv::Mat depth, cv::Mat K):
    #ifdef CUDA_ON
        tris(model.tris.size()),
    #else
        tris(model.tris),
    #endif
    model(model_path)  // model inits first
{
#ifdef CUDA_ON
    thrust::copy(model.tris.begin(), model.tris.end(), tris.begin_thr());
#endif
}

void PoseRenderer::set_K_width_height(cv::Mat K, int width, int height)
{
    assert(K.type() == CV_32F);
    this->K = K;
    this->width = width;
    this->height = height;
    proj_mat = cuda_renderer::compute_proj(K, width, height);
}

template<typename F>
auto PoseRenderer::render_what(F f, std::vector<cv::Mat> &init_poses, float down_sample)
{
    const int width_local = width/down_sample;
    const int height_local = height/down_sample;

    std::vector<cuda_renderer::Model::mat4x4> mat4_v(init_poses.size());
    for(size_t i=0; i<init_poses.size();i++) mat4_v[i].init_from_cv(init_poses[i]);

    auto depths = cuda_renderer::render(tris, mat4_v, width_local, height_local, proj_mat);
    return f(depths, width_local, height_local, init_poses.size());
}

std::vector<cv::Mat> PoseRenderer::render_depth(std::vector<cv::Mat> &init_poses, float down_sample)
{
#ifdef CUDA_ON
    return render_what(cuda_renderer::raw2depth_uint16_cuda, init_poses, down_sample);
#else
    return render_what(cuda_renderer::raw2depth_uint16_cpu, init_poses, down_sample);
#endif
}

std::vector<cv::Mat> PoseRenderer::render_mask(std::vector<cv::Mat> &init_poses, float down_sample)
{
#ifdef CUDA_ON
    return render_what(cuda_renderer::raw2mask_uint8_cuda, init_poses, down_sample);
#else
    return render_what(cuda_renderer::raw2mask_uint8_cpu, init_poses, down_sample);
#endif
}

std::vector<std::vector<cv::Mat> > PoseRenderer::render_depth_mask(std::vector<cv::Mat> &init_poses, float down_sample)
{
#ifdef CUDA_ON
    return render_what(cuda_renderer::raw2depth_mask_cuda, init_poses, down_sample);
#else
    return render_what(cuda_renderer::raw2depth_mask_cpu, init_poses, down_sample);
#endif
}

cv::Mat PoseRenderer::view_dep(cv::Mat dep)
{
    cv::Mat map = dep;
    double min;
    double max;
    cv::minMaxIdx(map, &min, &max);
    cv::Mat adjMap;
    map.convertTo(adjMap,CV_8UC1, 255 / (max-min), -min);
    cv::Mat falseColorsMap;
    applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_HOT);
    return falseColorsMap;
}


