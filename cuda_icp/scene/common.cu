#include "common.h"

device_vector_v3f_holder::~device_vector_v3f_holder(){
    __free();
}

void device_vector_v3f_holder::__free(){
    if(valid){
        cudaFree(__gpu_memory);
        valid = false;
        __size = 0;
    }
}

device_vector_v3f_holder::device_vector_v3f_holder(size_t size_, Vec3f init)
{
    __malloc(size_);
    thrust::fill(begin_thr(), end_thr(), init);
}

void device_vector_v3f_holder::__malloc(size_t size_){
    if(valid) __free();
    cudaMalloc((void**)&__gpu_memory, size_ * sizeof(Vec3f));
    __size = size_;
    valid = true;
}

device_vector_v3f_holder::device_vector_v3f_holder(size_t size_){
    __malloc(size_);
}

