# pose_refine
for 6D pose estimation we need a quick icp.

## introduction
cuda_renderer: render handreds poes of a ply model at once.  

cuda_icp: cuda version of point to plane icp, implement projective & nearest neibor association  

key ideas:  

1. build & search a kdtree in non-iterative, non-stack way, because on gpu our stack is small for each thread, and we also don't want dynamic malloc to mimic a stack.

2. regard icp as a huge transform_reduce process, one thrust call is enough, save much time compared to global memory version  

3. use cuda per-thread stream, we can handle multiple icps at once to make full use of gpu

todo:  core functions are done, more test, api and documents in the future  

[Chinese blog](https://zhuanlan.zhihu.com/p/58757649)
