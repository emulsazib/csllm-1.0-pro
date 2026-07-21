//
// pybind11 surface for the CSLLM C++ engine.
//
// Every compute entry point releases the GIL (py::gil_scoped_release). This is
// what allows the FastAPI gateway to serve concurrent token streams by
// dispatching per-token work through asyncio.to_thread — without it the event
// loop would stall on the first request. See memory-bank/rules.md.
//
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "csllm/arena.hpp"
#include "csllm/common.hpp"
#include "csllm/gemm.hpp"
#include "csllm/model.hpp"
#include "csllm/optim.hpp"
#include "csllm/sampler.hpp"
#include "csllm/threadpool.hpp"

namespace py = pybind11;
using namespace csllm;

namespace {

// Dense row-major C = A @ B. Exists so the build can be verified end-to-end
// against NumPy before any model math is written — it proves the Accelerate
// link is real, not just that CMake found the framework.
template <typename T>
py::array_t<T> py_matmul(py::array_t<T, py::array::c_style | py::array::forcecast> a,
                         py::array_t<T, py::array::c_style | py::array::forcecast> b) {
  auto ab = a.request();
  auto bb = b.request();
  if (ab.ndim != 2 || bb.ndim != 2) throw Error("matmul expects 2-D arrays");
  if (ab.shape[1] != bb.shape[0]) {
    throw Error("inner dimensions do not match: " + std::to_string(ab.shape[1]) + " vs " +
                std::to_string(bb.shape[0]));
  }

  const i64 m = ab.shape[0], k = ab.shape[1], n = bb.shape[1];
  py::array_t<T> out({m, n});
  auto ob = out.request();

  const T* ap = static_cast<const T*>(ab.ptr);
  const T* bp = static_cast<const T*>(bb.ptr);
  T* op = static_cast<T*>(ob.ptr);

  {
    py::gil_scoped_release release;
    gemm_nn<T>(m, n, k, ap, bp, op);
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(_csllm_core, m) {
  m.doc() = "CSLLM C++ engine — tensors, hand-written autograd, and inference";
  m.attr("__version__") = CSLLM_VERSION;

  py::register_exception<Error>(m, "CSLLMError", PyExc_RuntimeError);

  py::class_<BuildInfo>(m, "BuildInfo")
      .def_readonly("version", &BuildInfo::version)
      .def_readonly("blas_backend", &BuildInfo::blas_backend)
      .def_readonly("accelerate_enabled", &BuildInfo::accelerate_enabled)
      .def_readonly("fast_math", &BuildInfo::fast_math)
      .def_readonly("hardware_threads", &BuildInfo::hardware_threads)
      .def_readonly("cxx_standard", &BuildInfo::cxx_standard)
      .def("__repr__", [](const BuildInfo& b) {
        return "<BuildInfo version=" + b.version + " blas=" + b.blas_backend +
               " threads=" + std::to_string(b.hardware_threads) + ">";
      });

  m.def("build_info", &build_info, "Compiler, BLAS backend, and threading details");

  py::class_<ModelConfig>(m, "ModelConfig")
      .def(py::init<>())
      .def_readwrite("vocab_size", &ModelConfig::vocab_size)
      .def_readwrite("n_layer", &ModelConfig::n_layer)
      .def_readwrite("n_head", &ModelConfig::n_head)
      .def_readwrite("n_embd", &ModelConfig::n_embd)
      .def_readwrite("block_size", &ModelConfig::block_size)
      .def_readwrite("ffn_hidden", &ModelConfig::ffn_hidden)
      .def_readwrite("rope_theta", &ModelConfig::rope_theta)
      .def_readwrite("norm_eps", &ModelConfig::norm_eps)
      .def_property_readonly("head_dim", &ModelConfig::head_dim)
      .def("num_params", &ModelConfig::num_params)
      .def("validate", &ModelConfig::validate)
      .def("to_json", &ModelConfig::to_json)
      .def("__repr__", [](const ModelConfig& c) {
        return "<ModelConfig L=" + std::to_string(c.n_layer) + " H=" + std::to_string(c.n_head) +
               " D=" + std::to_string(c.n_embd) + " V=" + std::to_string(c.vocab_size) +
               " params=" + std::to_string(c.num_params()) + ">";
      });

  py::class_<SamplingParams>(m, "SamplingParams")
      .def(py::init<>())
      .def_readwrite("temperature", &SamplingParams::temperature)
      .def_readwrite("top_k", &SamplingParams::top_k)
      .def_readwrite("top_p", &SamplingParams::top_p)
      .def_readwrite("repetition_penalty", &SamplingParams::repetition_penalty);

  py::class_<AdamWConfig>(m, "AdamWConfig")
      .def(py::init<>())
      .def_readwrite("lr", &AdamWConfig::lr)
      .def_readwrite("beta1", &AdamWConfig::beta1)
      .def_readwrite("beta2", &AdamWConfig::beta2)
      .def_readwrite("eps", &AdamWConfig::eps)
      .def_readwrite("weight_decay", &AdamWConfig::weight_decay);

  m.def("cosine_lr", &cosine_lr, py::arg("step"), py::arg("warmup_steps"), py::arg("max_steps"),
        py::arg("lr_max"), py::arg("lr_min"),
        "Cosine decay with linear warmup");

  m.def("matmul_f32", &py_matmul<f32>, py::arg("a"), py::arg("b"),
        "Row-major C = A @ B in float32, via the configured BLAS backend");
  m.def("matmul_f64", &py_matmul<f64>, py::arg("a"), py::arg("b"),
        "Row-major C = A @ B in float64, via the configured BLAS backend");

  m.def("thread_pool_size", [] { return ThreadPool::global().size(); },
        "Worker count of the process-wide thread pool");

  // Arena is exposed mainly so tests can assert that exhaustion throws rather
  // than silently falling back to malloc.
  py::class_<Arena>(m, "Arena")
      .def(py::init<std::size_t>(), py::arg("capacity_bytes"))
      .def("allocate",
           [](Arena& a, std::size_t n) {
             a.allocate(n);
             return a.used();
           })
      .def("reset", &Arena::reset)
      .def_property_readonly("used", &Arena::used)
      .def_property_readonly("capacity", &Arena::capacity)
      .def_property_readonly("high_water", &Arena::high_water);
}
