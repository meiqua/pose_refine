
# opencv
find_package(OpenCV 3 REQUIRED)
list(APPEND renderer_inc ${OpenCV_INCLUDE_DIRS})
list(APPEND renderer_lib ${OpenCV_LIBS})


# cuda
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O3")
list(APPEND renderer_inc ${CUDA_INCLUDE_DIRS})
list(APPEND renderer_lib ${CUDA_LIBRARIES})


# assimp, load ply
find_package(ASSIMP REQUIRED)
LINK_DIRECTORIES(/usr/local/lib)
list(APPEND renderer_inc ${ASSIMP_INCLUDE_DIRS})
list(APPEND renderer_lib ${ASSIMP_LIBRARIES})


# src
SET(renderer_cuda renderer.cu)
SET(renderer_srcs renderer.cpp)
CUDA_COMPILE(renderer_cuda_objs ${renderer_cuda})


# lib & test exe
add_library(cuda_renderer
               ${renderer_srcs}
               ${renderer_cuda} 
               ${renderer_cuda_objs} 
)
target_include_directories(cuda_renderer PUBLIC ${renderer_inc})
target_link_libraries(cuda_renderer PUBLIC ${renderer_lib})


add_executable(renderer_test test.cpp)
target_link_libraries(renderer_test cuda_renderer)
