
# opencv
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-8.0)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /home/meiqua/anaconda3/envs/furnace)
find_package(OpenCV 3 REQUIRED)
list(APPEND icp_inc ${OpenCV_INCLUDE_DIRS})
list(APPEND icp_lib ${OpenCV_LIBS})


# cuda
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O3 --default-stream per-thread")
list(APPEND icp_inc ${CUDA_INCLUDE_DIRS})
list(APPEND icp_lib ${CUDA_LIBRARIES})


# src
SET(icp_cuda_srcs icp.cu)
SET(icp_srcs icp.cpp)
CUDA_COMPILE(icp_cuda_objs ${icp_cuda_srcs})


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
