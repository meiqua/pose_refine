#include "renderer.h"
#include <assert.h>

using namespace cuda_renderer;

cuda_renderer::Model::~Model()
{

}

cuda_renderer::Model::Model(const std::string &fileName)
{
    LoadModel(fileName);
}

void cuda_renderer::Model::LoadModel(const std::string &fileName)
{
    scene = aiImportFile(fileName.c_str(), aiProcessPreset_TargetRealtime_Quality);

    // need to?
//    for (unsigned int i=0; i<scene->mNumMeshes; i++){
//      for (unsigned int j=0; j<scene->mMeshes[i]->mNumVertices; j++){
//        scene->mMeshes[i]->mVertices[j].x /= 1000;
//        scene->mMeshes[i]->mVertices[j].y /= 1000;
//        scene->mMeshes[i]->mVertices[j].z /= 1000;
//      }
//    }

    {
        tris.clear();
        size_t guess_size = scene->mMeshes[scene->mRootNode->mMeshes[0]]->mNumFaces;
        tris.reserve(guess_size);
    }
    {
        faces.clear();
        size_t guess_size = scene->mMeshes[scene->mRootNode->mMeshes[0]]->mNumFaces;
        faces.reserve(guess_size);
    }
    {
        vertices.clear();
        size_t guess_size = scene->mMeshes[scene->mRootNode->mMeshes[0]]->mNumVertices;
        vertices.reserve(guess_size);
    }
    recursive_render(scene, scene->mRootNode);

    get_bounding_box(bbox_min, bbox_max);

    aiReleaseImport(scene);

    std::cout << "load model success    " <<                 std::endl;
    std::cout << "face(triangles) nums: " << faces.size() << std::endl;
    std::cout << "       vertices nums: " <<vertices.size()<<std::endl;

    if(faces.size() > 10000)
        std::cout << "you may want tools like meshlab to simplify models to speed up rendering" << std::endl;

    std::cout << "------------------------------------\n" << std::endl;
}

cuda_renderer::Model::float3 cuda_renderer::Model::mat_mul_vec(const aiMatrix4x4 &mat, const aiVector3D &vec)
{
    return {
        mat.a1*vec.x + mat.a2*vec.y + mat.a3*vec.z + mat.a4,
        mat.b1*vec.x + mat.b2*vec.y + mat.b3*vec.z + mat.b4,
        mat.c1*vec.x + mat.c2*vec.y + mat.c3*vec.z + mat.c4,
    };
}

void cuda_renderer::Model::recursive_render(const aiScene *sc, const aiNode *nd, aiMatrix4x4 m)
{
    aiMultiplyMatrix4(&m, &nd->mTransformation);

    for (size_t n=0; n < nd->mNumMeshes; ++n){
        const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];

        for (size_t t = 0; t < mesh->mNumFaces; ++t){
            const struct aiFace* face = &mesh->mFaces[t];

            if(face->mNumIndices < 3) continue;  // invalid face, don't know why they exists in hinter obj_03.ply
            assert(face->mNumIndices == 3 && "we only render triangle, use tools like meshlab to modify this models");

            Triangle tri_temp;
            tri_temp.v0 = mat_mul_vec(m, mesh->mVertices[face->mIndices[0]]);
            tri_temp.v1 = mat_mul_vec(m, mesh->mVertices[face->mIndices[1]]);
            tri_temp.v2 = mat_mul_vec(m, mesh->mVertices[face->mIndices[2]]);

            tris.push_back(tri_temp);

            int3 face_temp;
            face_temp.v0 = face->mIndices[0];
            face_temp.v1 = face->mIndices[1];
            face_temp.v2 = face->mIndices[2];
            faces.push_back(face_temp);
        }

        for(size_t t = 0; t < mesh->mNumVertices; ++t){
            float3 v;
            v.x = mesh->mVertices[t].x;
            v.y = mesh->mVertices[t].y;
            v.z = mesh->mVertices[t].z;
            vertices.push_back(v);
        }
    }

    // draw all children
    for (size_t n = 0; n < nd->mNumChildren; ++n)
        recursive_render(sc, nd->mChildren[n], m);
}

void cuda_renderer::Model::get_bounding_box_for_node(const aiNode *nd, aiVector3D& min, aiVector3D& max, aiMatrix4x4 *trafo) const
{
    aiMatrix4x4 prev; // Use struct keyword to show you want struct version of this, not normal typedef?
    unsigned int n = 0, t;

    prev = *trafo;
    aiMultiplyMatrix4(trafo, &nd->mTransformation);

    for (; n < nd->mNumMeshes; ++n)
    {
      const struct aiMesh* mesh = scene->mMeshes[nd->mMeshes[n]];
      for (t = 0; t < mesh->mNumVertices; ++t)
      {
        aiVector3D tmp = mesh->mVertices[t];
        aiTransformVecByMatrix4(&tmp, trafo);

        min.x = std::min(min.x,tmp.x);
        min.y = std::min(min.y,tmp.y);
        min.z = std::min(min.z,tmp.z);

        max.x = std::max(max.x,tmp.x);
        max.y = std::max(max.y,tmp.y);
        max.z = std::max(max.z,tmp.z);
      }
    }

    for (n = 0; n < nd->mNumChildren; ++n)
      get_bounding_box_for_node(nd->mChildren[n], min, max, trafo);

    *trafo = prev;
}

void cuda_renderer::Model::get_bounding_box(aiVector3D& min, aiVector3D &max) const
{
    aiMatrix4x4 trafo;
    aiIdentityMatrix4(&trafo);

    min.x = min.y = min.z = 1e10f;
    max.x = max.y = max.z = -1e10f;
    get_bounding_box_for_node(scene->mRootNode, min, max, &trafo);
}

std::vector<cuda_renderer::Model::mat4x4> cuda_renderer::mat_to_compact_4x4(const std::vector<cv::Mat> &poses)
{
    std::vector<cuda_renderer::Model::mat4x4> mat4x4s(poses.size());
    for(size_t i=0; i<poses.size(); i++){
        mat4x4s[i].init_from_cv(poses[i]);
    }
    return mat4x4s;
}

cuda_renderer::Model::mat4x4 cuda_renderer::compute_proj(const cv::Mat &K, int width, int height, float near, float far)
{
    cuda_renderer::Model::mat4x4 p;
    p.a0 = 2*K.at<float>(0, 0)/width;
    p.a1 = -2*K.at<float>(0, 1)/width; p.a1 = -p.a1;  // yz flip
    p.a2 = -2*K.at<float>(0, 2)/width + 1; p.a2 = -p.a2;
    p.a3 = 0;

    p.b0 = 0;
    p.b1 = 2*K.at<float>(1, 1)/height; p.b1 = -p.b1;
    p.b2 = 2*K.at<float>(1, 2)/height - 1; p.b2 = -p.b2;
    p.b3 = 0;

    p.c0 = 0;
    p.c1 = 0;
    p.c2 = -(far+near)/(far-near); p.c2 = -p.c2;
    p.c3 = -2*far*near/(far-near);

    p.d0 = 0;
    p.d1 = 0;
    p.d2 = -1; p.d2 = -p.d2;
    p.d3 = 0;

    return p;
}

// cpu renderer, for test

// slightly different from device one, use no atomicMin
void rasterization(const Model::Triangle dev_tri, Model::float3 last_row,
                                        int32_t* depth_entry, size_t width, size_t height, const Model::ROI roi){
    // refer to tiny renderer
    // https://github.com/ssloy/tinyrenderer/blob/master/our_gl.cpp
    float pts2[3][2];

    // viewport transform(0, 0, width, height)
    pts2[0][0] = dev_tri.v0.x/last_row.x*width/2.0f+width/2.0f;
    pts2[0][1] = dev_tri.v0.y/last_row.x*height/2.0f+height/2.0f;

    pts2[1][0] = dev_tri.v1.x/last_row.y*width/2.0f+width/2.0f;
    pts2[1][1] = dev_tri.v1.y/last_row.y*height/2.0f+height/2.0f;

    pts2[2][0] = dev_tri.v2.x/last_row.z*width/2.0f+width/2.0f;
    pts2[2][1] = dev_tri.v2.y/last_row.z*height/2.0f+height/2.0f;

    float bboxmin[2] = {FLT_MAX,  FLT_MAX};
    float bboxmax[2] = {-FLT_MAX, -FLT_MAX};

    float clamp_max[2] = {float(width-1), float(height-1)};
    float clamp_min[2] = {0, 0};

    size_t real_width = width;
    if(roi.width > 0 && roi.height > 0){  // depth will be flipped
        clamp_min[0] = roi.x;
        clamp_min[1] = height-1 - (roi.y + roi.height - 1);
        clamp_max[0] = (roi.x + roi.width) - 1;
        clamp_max[1] = height-1 - roi.y;
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
//            float frag_depth = (dev_tri.v0.z * bc_over_z.x + dev_tri.v1.z * bc_over_z.y + dev_tri.v2.z * bc_over_z.z)
//                    /(bc_over_z.x + bc_over_z.y + bc_over_z.z);

            // this seems better
            float frag_depth = (bc_screen.x + bc_screen.y + bc_screen.z)
                    /(bc_over_z.x + bc_over_z.y + bc_over_z.z);

            size_t x_to_write = (P[0] - roi.x);
            size_t y_to_write = (height - 1 - P[1] - roi.y);

            int32_t depth = int32_t(frag_depth/**1000*/ + 0.5f);
            int32_t& depth_to_write = depth_entry[x_to_write+y_to_write*real_width];

            if(depth < depth_to_write)
            depth_to_write = depth;
        }
    }
}

std::vector<int32_t> cuda_renderer::render_cpu(const std::vector<cuda_renderer::Model::Triangle> &tris,
                                           const std::vector<cuda_renderer::Model::mat4x4> &poses,
                                           size_t width, size_t height, const cuda_renderer::Model::mat4x4 &proj_mat,
                                               const Model::ROI roi)
{
    size_t real_width = width;
    size_t real_height = height;
    if(roi.width > 0 && roi.height > 0){
        real_width = roi.width;
        real_height = roi.height;
    }
    std::vector<int32_t> depth(poses.size()*real_width*real_height, INT_MAX);

#pragma omp parallel for
    for(size_t i=0; i<poses.size(); i++){

        const auto& pose = poses[i];
        for(const auto& tri: tris){
            // model transform
            Model::Triangle local_tri = transform_triangle(tri, pose);
//            if(normal_functor::is_back(local_tri)) continue;

            // assume last column of projection matrix is  0 0 1 0
            Model::float3 last_row = {
                local_tri.v0.z,
                local_tri.v1.z,
                local_tri.v2.z
            };
            // projection transform
            local_tri = transform_triangle(local_tri, proj_mat);
            rasterization(local_tri, last_row, depth.data()+i*real_width*real_height, width, height, roi);
        }
    }

    for(auto& d: depth){
        if(d==INT_MAX) d=0;
    }

    return depth;
}

std::vector<cv::Mat> cuda_renderer::raw2depth_uint16_cpu(std::vector<int32_t> &raw_data, size_t width, size_t height, size_t pose_size)
{
    assert(raw_data.size() == width*height*pose_size);

    std::vector<cv::Mat> depths(pose_size);
    for(auto& dep: depths){
        dep = cv::Mat(height, width, CV_16U, cv::Scalar(0));
    }

    size_t step = width*height;

    for(size_t i=0; i<pose_size; i++){
        for(int r=0; r<height; r++){
            for(int c=0; c<width; c++){
                depths[i].at<uint16_t>(r, c) = uint16_t(raw_data[i*step + width*r + c]);
            }
        }
    }

    return depths;
}

std::vector<cv::Mat> cuda_renderer::raw2mask_uint8_cpu(std::vector<int32_t> &raw_data, size_t width, size_t height, size_t pose_size)
{
    assert(raw_data.size() == width*height*pose_size);

    std::vector<cv::Mat> masks(pose_size);
    for(auto& mask: masks){
        mask = cv::Mat(height, width, CV_8U, cv::Scalar(0));
    }

    size_t step = width*height;
    for(size_t i=0; i<pose_size; i++){
        for(int r=0; r<height; r++){
            for(int c=0; c<width; c++){
                masks[i].at<uchar>(r, c) = ((raw_data[i*step + width*r + c] > 0)?255:0);
            }
        }
    }

    return masks;
}

std::vector<std::vector<cv::Mat> > cuda_renderer::raw2depth_mask_cpu(std::vector<int32_t> &raw_data, size_t width, size_t height, size_t pose_size)
{
    assert(raw_data.size() == width*height*pose_size);

    std::vector<std::vector<cv::Mat>> results(pose_size, std::vector<cv::Mat>(2));
    for(auto& dep_mask: results){
        dep_mask[0] = cv::Mat(height, width, CV_16U, cv::Scalar(0));
        dep_mask[1] = cv::Mat(height, width, CV_8U, cv::Scalar(0));
    }

    size_t step = width*height;
    for(size_t i=0; i<pose_size; i++){
        for(int r=0; r<height; r++){
            for(int c=0; c<width; c++){

                auto& raw = raw_data[i*step + width*r + c];
                results[i][0].at<uint16_t>(r, c) = uint16_t(raw);
                results[i][1].at<uchar>(r, c) = ((raw > 0)?255:0);
            }
        }
    }

    return results;
}
