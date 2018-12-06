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
            assert(face->mNumIndices == 3 && "we only render triangle");

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


// cpu renderer, for test
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
    p.a1 = -2*K.at<float>(0, 1)/width;
    p.a2 = -2*K.at<float>(0, 2)/width + 1;
    p.a3 = 0;

    p.b0 = 0;
    p.b1 = 2*K.at<float>(1, 1)/height;
    p.b2 = 2*K.at<float>(1, 2)/width - 1;
    p.b3 = 0;

    p.c0 = 0;
    p.c1 = 0;
    p.c2 = -(far+near)/(far-near);
    p.c3 = -2*far*near/(far-near);

    p.d0 = 0;
    p.d1 = 0;
    p.d2 = -1;
    p.d3 = 0;

    return p;
}

static Model::float3 mat_mul_v(const Model::mat4x4& tran, const Model::float3& v){
    return {
        tran.a0*v.x + tran.a1*v.y + tran.a2*v.z + tran.a3,
        tran.b0*v.x + tran.b1*v.y + tran.b2*v.z + tran.b3,
        tran.c0*v.x + tran.c1*v.y + tran.c2*v.z + tran.c3,
    };
}

static Model::Triangle transform_triangle(const Model::Triangle& dev_tri, const Model::mat4x4& tran){
    return {
        mat_mul_v(tran, (dev_tri.v0)),
        mat_mul_v(tran, (dev_tri.v1)),
        mat_mul_v(tran, (dev_tri.v2)),
    };
}

static float calculateSignedArea(float* A, float* B, float* C){
    return 0.5f*((C[0]-A[0])*(B[1]-A[1]) - (B[0]-A[0])*(C[1]-A[1]));
}

static Model::float3 barycentric(float* A, float* B, float* C, float* P) {

    float float_P[2] = {float(P[0]), float(P[1])};

    float base_inv = 1/calculateSignedArea(A, B, C);
    float beta = calculateSignedArea(A, float_P, C)*base_inv;
    float gamma = calculateSignedArea(A, B, float_P)*base_inv;

    return {
        1.0f-beta-gamma,
        beta,
        gamma,
    };
}

namespace normal_functor{

static    Model::float3 minus(const Model::float3& one, const Model::float3& the_other)
    {
        return {
            one.x - the_other.x,
            one.y - the_other.y,
            one.z - the_other.z
        };
    }

static    Model::float3 cross(const Model::float3& one, const Model::float3& the_other)
    {
        return {
            one.y*the_other.z - one.z*the_other.y,
            one.z*the_other.x - one.x*the_other.z,
            one.x*the_other.y - one.y*the_other.x
        };
    }

static    Model::float3 normalized(const Model::float3& one)
    {
        float norm = std::sqrt(one.x*one.x+one.y*one.y+one.z*one.z);
        return {
          one.x/norm,
          one.y/norm,
          one.z/norm
        };
    }

static    Model::float3 get_normal(const Model::Triangle& dev_tri)
    {
//      return normalized(cross(minus(dev_tri.v1, dev_tri.v0), minus(dev_tri.v1, dev_tri.v0)));

      // no need for normalizing?
      return (cross(minus(dev_tri.v1, dev_tri.v0), minus(dev_tri.v2, dev_tri.v0)));
    }
};

static void rasterization(const Model::Triangle& dev_tri, Model::float3& last_row,
                                        float* depth_entry, size_t width, size_t height){
//    cv::Mat depth = cv::Mat(height, width, CV_32FC1, depth_entry);
    // refer to tiny renderer
    // https://github.com/ssloy/tinyrenderer/blob/master/our_gl.cpp
    float pts2[3][2];

    // viewport transform(0, 0, width, height)
    pts2[0][0] = dev_tri.v0.x/last_row.x*width/2.0f+width/2.0f; pts2[0][1] = dev_tri.v0.y/last_row.x*height/2.0f+height/2.0f;
    pts2[1][0] = dev_tri.v1.x/last_row.y*width/2.0f+width/2.0f; pts2[1][1] = dev_tri.v1.y/last_row.y*height/2.0f+height/2.0f;
    pts2[2][0] = dev_tri.v2.x/last_row.z*width/2.0f+width/2.0f; pts2[2][1] = dev_tri.v2.y/last_row.z*height/2.0f+height/2.0f;

//    std::cout << "\n------------------------------" << std::endl;
//    std::cout << last_row << std::endl;
//    std::cout << dev_tri << std::endl;

//    std::cout << pts2[0][0] << "\t" << pts2[0][1] << std::endl;
//    std::cout << pts2[1][0] << "\t" << pts2[1][1] << std::endl;
//    std::cout << pts2[2][0] << "\t" << pts2[2][1] << std::endl;
//    std::cout << "------------------------------\n" << std::endl;

    // for test
//    Model::Triangle local_tri;
//    local_tri.v0.x = pts2[0][0]; local_tri.v0.y = pts2[0][1]; local_tri.v0.z = 0;
//    local_tri.v1.x = pts2[1][0]; local_tri.v1.y = pts2[1][1]; local_tri.v1.z = 0;
//    local_tri.v2.x = pts2[2][0]; local_tri.v2.y = pts2[2][1]; local_tri.v2.z = 0;

    float bboxmin[2] = {FLT_MAX,  FLT_MAX};
    float bboxmax[2] = {-FLT_MAX, -FLT_MAX};
    float clamp[2] = {float(width-1), float(height-1)};
    for (int i=0; i<3; i++) {
        for (int j=0; j<2; j++) {
            bboxmin[j] = std::max(0.f,      std::min(bboxmin[j], pts2[i][j]));
            bboxmax[j] = std::min(clamp[j], std::max(bboxmax[j], pts2[i][j]));
        }
    }

    float P[2];

    // there will be small holes if +1; Model may be not so good?
    for(P[1] = int(bboxmin[1]); P[1]<=bboxmax[1]; P[1] += 1.0f){
        for(P[0] = int(bboxmin[0]); P[0]<=bboxmax[0]; P[0] += 1.0f){

            Model::float3 bc_screen  = barycentric(pts2[0], pts2[1], pts2[2], P);

//            const bool test_whitch = P[0]==(640-361) && P[1]==(480-121);
//            if(test_whitch){
//                std::cout << "\nbarycentric:" << std::endl;
//                std::cout << bc_screen;
//                std::cout << "\ntriangle:" << std::endl;
//                std::cout << local_tri;
//                std::cout << "--------------" << std::endl;
//            }

            // out of triangle
            if (bc_screen.x<-0.0f || bc_screen.y<-0.0f || bc_screen.z<-0.0f ||
                    bc_screen.x>1.0f || bc_screen.y>1.0f || bc_screen.z>1.0f ) continue;

//            if(test_whitch){
//                std::cerr << "can't be here" << std::endl;
//            }

            Model::float3 bc_over_z = {bc_screen.x/last_row.x, bc_screen.y/last_row.y, bc_screen.z/last_row.z};

            // refer to https://en.wikibooks.org/wiki/Cg_Programming/Rasterization, Perspectively Correct Interpolation
            float frag_depth = -(dev_tri.v0.z*bc_over_z.x + dev_tri.v1.z*bc_over_z.y + dev_tri.v2.z*bc_over_z.z)
                    /(bc_over_z.x + bc_over_z.y + bc_over_z.z);

            auto& depth_to_write = depth_entry[(width - int(P[0]+0.5f))+(height - int(P[1]+0.5f))*width];
            if(frag_depth < depth_to_write){
                depth_to_write = frag_depth;

//                cv::imshow("test", depth != FLT_MAX);
//                cv::waitKey(0);
            }
        }
    }

//    cv::imshow("test", depth != FLT_MAX);
//    cv::waitKey(0);
}

std::vector<float> cuda_renderer::render_cpu(const std::vector<cuda_renderer::Model::Triangle> &tris,
                                           const std::vector<cuda_renderer::Model::mat4x4> &poses,
                                           size_t width, size_t height, const cuda_renderer::Model::mat4x4 &proj_mat)
{
    std::vector<float> depth(poses.size()*width*height, FLT_MAX);

    size_t i = 0;
    for(const auto& pose: poses){
        for(const auto& tri: tris){
            // model transform
            Model::Triangle local_tri = transform_triangle(tri, pose);
//            if(normal_functor::get_normal(local_tri).z < 0) continue;

            // assume last column of projection matrix is  0 0 -1 0
            Model::float3 last_row = {
                -local_tri.v0.z,
                -local_tri.v1.z,
                -local_tri.v2.z
            };
            // projection transform
            local_tri = transform_triangle(local_tri, proj_mat);
            rasterization(local_tri, last_row, depth.data()+i*width*height, width, height);
        }
        i++;
    }

    for(auto& d: depth){
        if(d==FLT_MAX) d=0;
    }

    return depth;
}
