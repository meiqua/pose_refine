cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 14)
project(pose_refine)

option(USE_CUDA "use cuda or not" ON)

#SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_BUILD_TYPE "Debug")

SET(CMAKE_CXX_FLAGS "-fopenmp -Wall -Wno-sign-compare $ENV{CXXFLAGS}")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(USE_CUDA)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.0)
add_definitions(-DCUDA_ON)
endif()

add_subdirectory(cuda_renderer)
add_subdirectory(cuda_icp)
list(APPEND pose_refine_lib cuda_renderer cuda_icp)

# opencv
find_package(OpenCV 3 REQUIRED)
list(APPEND pose_refine_inc ${OpenCV_INCLUDE_DIRS})
list(APPEND pose_refine_lib ${OpenCV_LIBS})


# open3d
find_package(Open3D REQUIRED)
if (Open3D_FOUND)
    message(STATUS "Found Open3D ${Open3D_VERSION}")

    include_directories(${Open3D_INCLUDE_DIRS})
    link_directories(${Open3D_LIBRARY_DIRS})

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Open3D_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")
else ()
    message(SEND_ERROR "Open3D not found")
endif ()
list(APPEND pose_refine_lib ${Open3D_LIBRARIES})


# src
SET(pose_refine_srcs pose_refine.cpp)


# lib & test exe
add_library(pose_refine ${pose_refine_srcs})
target_include_directories(pose_refine PUBLIC ${pose_refine_inc})
target_link_libraries(pose_refine PUBLIC ${pose_refine_lib})

add_executable(pose_refine_test test.cpp)
target_link_libraries(pose_refine_test pose_refine)
