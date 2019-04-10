#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "np2mat/ndarray_converter.h"
#include "pose_refine.h"
namespace py = pybind11;

PYBIND11_MODULE(linemodLevelup_pybind, m) {
    NDArrayConverter::init_numpy();

    py::class_<Mat4x4f>(m, "Mat4x4f", py::buffer_protocol())
       .def_buffer([](Mat4x4f &m) -> py::buffer_info {
            return py::buffer_info(
                &m[0][0],                               /* Pointer to buffer */
                sizeof(float),                          /* Size of one scalar */
                py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                2,                                      /* Number of dimensions */
                { 4, 4 },                 /* Buffer dimensions */
                { sizeof(float) * 4,             /* Strides (in bytes) for each index */
                  sizeof(float) }
            );
        });

    py::class_<cuda_icp::RegistrationResult>(m,"RegistrationResult")
            .def(py::init<>())
            .def_readwrite("fitness_", &cuda_icp::RegistrationResult::fitness_)
            .def_readwrite("inlier_rmse_", &cuda_icp::RegistrationResult::inlier_rmse_)
            .def_readwrite("transformation_", &cuda_icp::RegistrationResult::transformation_);

    py::class_<PoseRefine>(m, "PoseRefine")
            .def(py::init<cv::Mat, cv::Mat, std::string>())
            .def("process_batch", &PoseRefine::process_batch, py::arg("init_poses"),
                  py::arg("down_sample") = 2, py::arg("depth_aligned") = false)
            .def("poses_extend", &PoseRefine::poses_extend, py::arg("init_poses"),
                  py::arg("degree_var") = CV_PI/18)
            .def("results_filter", &PoseRefine::results_filter, py::arg("results"),
                  py::arg("edge_hit_rate_thresh") = 0.7f, py::arg("fitness_thresh") = 0.7f,
                  py::arg("rmse_thresh") = 0.07f);
}
