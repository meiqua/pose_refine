# pose_refine
for 6D pose estimation we need a quick icp.

## introduction
cuda_renderer: render handreds poes of a ply model at once.  

cuda_icp: cuda version of point to plane icp, implement projective & nearest neibor association  

key ideas:  

1. build & search a kdtree in non-recursion, also stackless way, because on gpu our stack is small for each thread, and we also don't want dynamic malloc to mimic a stack.

2. regard icp as a huge transform_reduce process, one thrust call is enough, save much time compared to global memory version  

3. use cuda per-thread stream, we can handle multiple icps at once to make full use of gpu

For pcd scene, kdtree are built on cpu then transfered to gpu. [this branch](https://github.com/meiqua/pose_refine/tree/cuda_build_kdtree)
may bring you some inspirations about how to build kdtree on GPU. Also the normals should be calculated on GPU if we want it faster.  

See test.cpp on the outmost and in the cuda_renderer folder to learn how to use it.

[Chinese blog](https://zhuanlan.zhihu.com/p/58757649)
