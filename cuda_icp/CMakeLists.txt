
# opencv
find_package(OpenCV 3 REQUIRED)
list(APPEND icp_inc ${OpenCV_INCLUDE_DIRS})
list(APPEND icp_lib ${OpenCV_LIBS})
include_directories(${INCLUDE_DIRECTORIES} ${OpenCV_INCLUDE_DIRS})

if(USE_CUDA)
# cuda
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O3 --default-stream per-thread -Xcompiler -fopenmp")
list(APPEND icp_inc ${CUDA_INCLUDE_DIRS})
list(APPEND icp_lib ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endif()


# eigen
find_package(Eigen3 REQUIRED)
include_directories(${INCLUDE_DIRECTORIES} ${EIGEN3_INCLUDE_DIR})


# src
SET(icp_cuda_srcs  icp.cu scene/common.cu scene/depth_scene/depth_scene.cu scene/pcd_scene/pcd_scene.cu)
SET(icp_srcs  icp.cpp scene/common.cpp scene/depth_scene/depth_scene.cpp scene/pcd_scene/pcd_scene.cpp)


if(USE_CUDA)
CUDA_COMPILE(icp_cuda_objs ${icp_cuda_srcs})
endif()

# lib & test exe
add_library(cuda_icp
               ${icp_srcs}
               ${icp_cuda_srcs}
               ${icp_cuda_objs} 
)
target_include_directories(cuda_icp PUBLIC ${icp_inc})
target_link_libraries(cuda_icp PUBLIC ${icp_lib})

add_executable(icp_test test.cpp)
target_link_libraries(icp_test cuda_icp)
