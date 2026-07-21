//
// pybind11 surface for the CSLLM C++ engine.
//
// Every compute entry point releases the GIL (py::gil_scoped_release). This is
// what allows the FastAPI gateway to serve concurrent token streams by
// dispatching per-token work through asyncio.to_thread — without it the event
// loop would stall on the first request. See memory-bank/rules.md.
//
// The op_* functions exist for testing: each runs one op and, when given a
// grad_output, returns the analytic input gradients. Registered for both f32
// and f64 so gradchecks can run in double precision.
//
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <optional>
#include <vector>

#include "csllm/arena.hpp"
#include "csllm/attention.hpp"
#include "csllm/autograd.hpp"
#include "csllm/common.hpp"
#include "csllm/gemm.hpp"
#include "csllm/model.hpp"
#include "csllm/ops.hpp"
#include "csllm/optim.hpp"
#include "csllm/sampler.hpp"
#include "csllm/serialize.hpp"
#include "csllm/threadpool.hpp"

namespace py = pybind11;
using namespace csllm;

namespace {

template <typename T>
using NpArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

// Scratch arena for the op harness. Test tensors are small; 128 MiB is ample.
constexpr std::size_t kTestArenaBytes = 1u << 27;

Shape shape_of(const py::buffer_info& b) {
  CSLLM_CHECK(b.ndim >= 1 && b.ndim <= kMaxDims, "unsupported array rank");
  Shape s;
  s.ndim = static_cast<int>(b.ndim);
  for (int i = 0; i < s.ndim; ++i) s.dims[i] = static_cast<i64>(b.shape[i]);
  return s;
}

// Owns an arena and copies numpy inputs into it as differentiable tensors.
template <typename T>
class OpCtx {
 public:
  OpCtx() : arena_(kTestArenaBytes) { Tape<T>::active().clear(); }

  Tensor<T> in(const NpArray<T>& arr, bool requires_grad = true) {
    auto b = arr.request();
    Tensor<T> t = make_tensor<T>(arena_, shape_of(b), requires_grad);
    std::memcpy(t.data, b.ptr, static_cast<std::size_t>(t.numel()) * sizeof(T));
    return t;
  }

  Arena& arena() { return arena_; }

  static py::array_t<T> to_numpy(const Tensor<T>& t, const T* src) {
    std::vector<py::ssize_t> dims(static_cast<std::size_t>(t.shape.ndim));
    for (int i = 0; i < t.shape.ndim; ++i) dims[static_cast<std::size_t>(i)] = t.shape[i];
    py::array_t<T> out(dims);
    std::memcpy(out.request().ptr, src, static_cast<std::size_t>(t.numel()) * sizeof(T));
    return out;
  }

  py::array_t<T> value(const Tensor<T>& t) const { return to_numpy(t, t.data); }
  py::array_t<T> grad(const Tensor<T>& t) const { return to_numpy(t, t.grad); }

  // Seeds the output gradient and walks the tape.
  void run_backward(const Tensor<T>& out, const NpArray<T>& grad_out) {
    auto b = grad_out.request();
    CSLLM_CHECK(static_cast<i64>(b.size) == out.numel(),
                "grad_output size does not match the op output");
    std::memcpy(out.grad, b.ptr, static_cast<std::size_t>(out.numel()) * sizeof(T));
    CSLLM_CHECK(out.node_id >= 0, "op produced no graph node");
    Tape<T>::active().backward(out.node_id);
  }

 private:
  Arena arena_;
};

// ── op harness ──────────────────────────────────────────────────────────────
// Each returns the forward value when grad_output is None, else a tuple of
// input gradients in declaration order.

template <typename T>
py::object op_binary(const NpArray<T>& a, const NpArray<T>& b,
                     std::optional<NpArray<T>> grad_output,
                     Tensor<T> (*fn)(Arena&, const Tensor<T>&, const Tensor<T>&)) {
  OpCtx<T> ctx;
  Tensor<T> A = ctx.in(a), B = ctx.in(b);
  Tensor<T> out = fn(ctx.arena(), A, B);
  if (!grad_output) return ctx.value(out);
  ctx.run_backward(out, *grad_output);
  return py::make_tuple(ctx.grad(A), ctx.grad(B));
}

template <typename T>
py::object op_matmul(const NpArray<T>& a, const NpArray<T>& b, std::optional<NpArray<T>> g) {
  return op_binary<T>(a, b, std::move(g), &matmul<T>);
}
template <typename T>
py::object op_matmul_bt(const NpArray<T>& a, const NpArray<T>& b, std::optional<NpArray<T>> g) {
  return op_binary<T>(a, b, std::move(g), &matmul_bt<T>);
}
template <typename T>
py::object op_add(const NpArray<T>& a, const NpArray<T>& b, std::optional<NpArray<T>> g) {
  return op_binary<T>(a, b, std::move(g), &add<T>);
}

template <typename T>
py::object op_rmsnorm(const NpArray<T>& x, const NpArray<T>& gain, double eps,
                      std::optional<NpArray<T>> grad_output) {
  OpCtx<T> ctx;
  Tensor<T> X = ctx.in(x), G = ctx.in(gain);
  Tensor<T> out = rmsnorm<T>(ctx.arena(), X, G, static_cast<T>(eps));
  if (!grad_output) return ctx.value(out);
  ctx.run_backward(out, *grad_output);
  return py::make_tuple(ctx.grad(X), ctx.grad(G));
}

template <typename T>
py::object op_rope(const NpArray<T>& x, i64 pos_offset, double theta,
                   std::optional<NpArray<T>> grad_output) {
  OpCtx<T> ctx;
  Tensor<T> X = ctx.in(x);
  Tensor<T> out = rope<T>(ctx.arena(), X, pos_offset, static_cast<T>(theta));
  if (!grad_output) return ctx.value(out);
  ctx.run_backward(out, *grad_output);
  return py::make_tuple(ctx.grad(X));
}

template <typename T>
py::object op_softmax_causal(const NpArray<T>& s, std::optional<NpArray<T>> grad_output) {
  OpCtx<T> ctx;
  Tensor<T> S = ctx.in(s);
  Tensor<T> out = softmax_causal<T>(ctx.arena(), S);
  if (!grad_output) return ctx.value(out);
  ctx.run_backward(out, *grad_output);
  return py::make_tuple(ctx.grad(S));
}

template <typename T>
py::object op_silu(const NpArray<T>& x, std::optional<NpArray<T>> grad_output) {
  OpCtx<T> ctx;
  Tensor<T> X = ctx.in(x);
  Tensor<T> out = silu<T>(ctx.arena(), X);
  if (!grad_output) return ctx.value(out);
  ctx.run_backward(out, *grad_output);
  return py::make_tuple(ctx.grad(X));
}

template <typename T>
py::object op_swiglu(const NpArray<T>& x, const NpArray<T>& wg, const NpArray<T>& wu,
                     const NpArray<T>& wd, std::optional<NpArray<T>> grad_output) {
  OpCtx<T> ctx;
  Tensor<T> X = ctx.in(x), G = ctx.in(wg), U = ctx.in(wu), D = ctx.in(wd);
  Tensor<T> out = swiglu<T>(ctx.arena(), X, G, U, D);
  if (!grad_output) return ctx.value(out);
  ctx.run_backward(out, *grad_output);
  return py::make_tuple(ctx.grad(X), ctx.grad(G), ctx.grad(U), ctx.grad(D));
}

template <typename T>
py::object op_embedding(const NpArray<T>& table, const NpArray<i32>& ids,
                        std::optional<NpArray<T>> grad_output) {
  OpCtx<T> ctx;
  Tensor<T> W = ctx.in(table);
  auto ib = ids.request();
  Tensor<T> out = embedding<T>(ctx.arena(), W, static_cast<const i32*>(ib.ptr),
                               static_cast<i64>(ib.size));
  if (!grad_output) return ctx.value(out);
  ctx.run_backward(out, *grad_output);
  return py::make_tuple(ctx.grad(W));
}

template <typename T>
py::object op_cross_entropy(const NpArray<T>& logits, const NpArray<i32>& targets,
                            std::optional<NpArray<T>> grad_output) {
  OpCtx<T> ctx;
  Tensor<T> L = ctx.in(logits);
  auto tb = targets.request();
  Tensor<T> out = cross_entropy<T>(ctx.arena(), L, static_cast<const i32*>(tb.ptr),
                                   static_cast<i64>(tb.size));
  if (!grad_output) return py::cast(out.data[0]);
  ctx.run_backward(out, *grad_output);
  return py::make_tuple(ctx.grad(L));
}

template <typename T>
py::object op_attention(const NpArray<T>& x, const NpArray<T>& wq, const NpArray<T>& wk,
                        const NpArray<T>& wv, const NpArray<T>& wo, i64 n_head, double theta,
                        std::optional<NpArray<T>> grad_output) {
  OpCtx<T> ctx;
  Tensor<T> X = ctx.in(x), Q = ctx.in(wq), K = ctx.in(wk), V = ctx.in(wv), O = ctx.in(wo);
  AttentionParams p;
  p.n_head = n_head;
  p.head_dim = X.shape[2] / n_head;
  p.rope_theta = theta;
  Tensor<T> out = attention<T>(ctx.arena(), X, Q, K, V, O, p);
  if (!grad_output) return ctx.value(out);
  ctx.run_backward(out, *grad_output);
  return py::make_tuple(ctx.grad(X), ctx.grad(Q), ctx.grad(K), ctx.grad(V), ctx.grad(O));
}

template <typename T>
py::array_t<T> py_matmul_raw(NpArray<T> a, NpArray<T> b) {
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

// Registers the whole op harness for one scalar type under the given suffix.
template <typename T>
void register_ops(py::module_& m, const char* suffix) {
  const auto name = [&](const char* base) { return std::string(base) + suffix; };
  using OG = std::optional<NpArray<T>>;

  m.def(name("matmul").c_str(), &op_matmul<T>, py::arg("x"), py::arg("w"),
        py::arg("grad_output") = OG{});
  m.def(name("matmul_bt").c_str(), &op_matmul_bt<T>, py::arg("x"), py::arg("w"),
        py::arg("grad_output") = OG{});
  m.def(name("add").c_str(), &op_add<T>, py::arg("x"), py::arg("y"),
        py::arg("grad_output") = OG{});
  m.def(name("rmsnorm").c_str(), &op_rmsnorm<T>, py::arg("x"), py::arg("gain"),
        py::arg("eps") = 1e-5, py::arg("grad_output") = OG{});
  m.def(name("rope").c_str(), &op_rope<T>, py::arg("x"), py::arg("pos_offset") = 0,
        py::arg("theta") = 10000.0, py::arg("grad_output") = OG{});
  m.def(name("softmax_causal").c_str(), &op_softmax_causal<T>, py::arg("scores"),
        py::arg("grad_output") = OG{});
  m.def(name("silu").c_str(), &op_silu<T>, py::arg("x"), py::arg("grad_output") = OG{});
  m.def(name("swiglu").c_str(), &op_swiglu<T>, py::arg("x"), py::arg("wg"), py::arg("wu"),
        py::arg("wd"), py::arg("grad_output") = OG{});
  m.def(name("embedding").c_str(), &op_embedding<T>, py::arg("table"), py::arg("ids"),
        py::arg("grad_output") = OG{});
  m.def(name("cross_entropy").c_str(), &op_cross_entropy<T>, py::arg("logits"),
        py::arg("targets"), py::arg("grad_output") = OG{});
  m.def(name("attention").c_str(), &op_attention<T>, py::arg("x"), py::arg("wq"), py::arg("wk"),
        py::arg("wv"), py::arg("wo"), py::arg("n_head"), py::arg("theta") = 10000.0,
        py::arg("grad_output") = OG{});
}

// Zero-copy view of a parameter buffer. `base` keeps the owning Model alive for
// as long as the array exists.
template <typename T>
py::array_t<T> param_view(T* ptr, const Shape& shape, py::object base) {
  std::vector<py::ssize_t> dims(static_cast<std::size_t>(shape.ndim));
  std::vector<py::ssize_t> strides(static_cast<std::size_t>(shape.ndim));
  py::ssize_t stride = sizeof(T);
  for (int i = shape.ndim - 1; i >= 0; --i) {
    dims[static_cast<std::size_t>(i)] = shape[i];
    strides[static_cast<std::size_t>(i)] = stride;
    stride *= static_cast<py::ssize_t>(shape[i]);
  }
  return py::array_t<T>(dims, strides, ptr, std::move(base));
}

// Registers Model<T>. The f64 instantiation exists so the ENTIRE network —
// including weight tying, where the embedding gradient must accumulate from two
// separate paths — can be gradchecked in double precision. That end-to-end check
// is the one thing per-op tests cannot cover.
template <typename T>
py::class_<Model<T>> register_model(py::module_& m, const char* name) {
  return py::class_<Model<T>>(m, name)
      .def(py::init<const ModelConfig&, u64>(), py::arg("config"), py::arg("seed") = 1337)
      .def_static("load", &Model<T>::load, py::arg("path"))
      .def("save", &Model<T>::save, py::arg("path"))
      .def_property_readonly("config", &Model<T>::config)
      .def("num_params", [](Model<T>& s) { return s.params().num_params(); })
      .def("param_names", [](Model<T>& s) { return s.params().names(); })
      .def("activation_high_water", &Model<T>::activation_high_water)
      .def("zero_grad", &Model<T>::zero_grad)
      .def(
          "get_param",
          [](py::object self, const std::string& n) {
            auto& model = self.cast<Model<T>&>();
            Tensor<T>& t = model.params().get(n);
            return param_view<T>(t.data, t.shape, self);
          },
          py::arg("name"), "Zero-copy view of a parameter buffer")
      .def(
          "get_grad",
          [](py::object self, const std::string& n) {
            auto& model = self.cast<Model<T>&>();
            Tensor<T>& t = model.params().get(n);
            return param_view<T>(t.grad, t.shape, self);
          },
          py::arg("name"), "Zero-copy view of a parameter's gradient buffer")
      .def(
          "forward_loss",
          [](Model<T>& s, NpArray<i32> ids, NpArray<i32> targets) {
            auto ib = ids.request();
            auto tb = targets.request();
            CSLLM_CHECK(ib.ndim == 2, "ids must be [B,T]");
            CSLLM_CHECK(ib.size == tb.size, "ids and targets must have the same size");
            const i64 B = ib.shape[0], T_len = ib.shape[1];
            const auto* ip = static_cast<const i32*>(ib.ptr);
            const auto* tp = static_cast<const i32*>(tb.ptr);
            py::gil_scoped_release release;
            return s.forward_loss(ip, tp, B, T_len);
          },
          py::arg("ids"), py::arg("targets"), "Mean cross-entropy loss; builds the tape")
      .def(
          "backward",
          [](Model<T>& s) {
            py::gil_scoped_release release;
            s.backward();
          },
          "Seed dL/dL = 1 and walk the tape in reverse")
      .def(
          "forward_logits",
          [](Model<T>& s, NpArray<i32> ids) {
            auto ib = ids.request();
            CSLLM_CHECK(ib.ndim == 2, "ids must be [B,T]");
            const i64 B = ib.shape[0], T_len = ib.shape[1];
            const i64 V = s.config().vocab_size;
            const auto* ip = static_cast<const i32*>(ib.ptr);
            py::array_t<T> out({B * T_len, V});
            auto ob = out.request();
            {
              py::gil_scoped_release release;
              const T* logits = s.forward_logits(ip, B, T_len);
              std::memcpy(ob.ptr, logits, static_cast<std::size_t>(B * T_len * V) * sizeof(T));
            }
            return out;
          },
          py::arg("ids"), "No-grad forward; returns logits [B*T, vocab_size]");
}

}  // namespace

PYBIND11_MODULE(_csllm_core, m) {
  m.doc() = "CSLLM C++ engine — tensors, hand-written autograd, and inference";
  m.attr("__version__") = CSLLM_VERSION;

  py::register_exception<Error>(m, "CSLLMError", PyExc_RuntimeError);

  // ── build introspection ───────────────────────────────────────────────────
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
  m.def("thread_pool_size", [] { return ThreadPool::global().size(); });

  // ── config ────────────────────────────────────────────────────────────────
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
      .def_static("from_json", &ModelConfig::from_json)
      .def("__repr__", [](const ModelConfig& c) {
        return "<ModelConfig L=" + std::to_string(c.n_layer) + " H=" + std::to_string(c.n_head) +
               " D=" + std::to_string(c.n_embd) + " V=" + std::to_string(c.vocab_size) +
               " params=" + std::to_string(c.num_params()) + ">";
      });

  m.def("estimate_activation_bytes",
        [](const ModelConfig& c, i64 B, i64 T) {
          return estimate_activation_bytes(c, B, T, sizeof(f32));
        },
        py::arg("config"), py::arg("batch_size"), py::arg("seq_len"));

  // ── sampling / optimizer config ───────────────────────────────────────────
  py::class_<SamplingParams>(m, "SamplingParams")
      .def(py::init<>())
      .def(py::init([](f32 temperature, i32 top_k, f32 top_p) {
             SamplingParams p;
             p.temperature = temperature;
             p.top_k = top_k;
             p.top_p = top_p;
             return p;
           }),
           py::arg("temperature") = 1.0f, py::arg("top_k") = 0, py::arg("top_p") = 1.0f)
      .def_readwrite("temperature", &SamplingParams::temperature)
      .def_readwrite("top_k", &SamplingParams::top_k)
      .def_readwrite("top_p", &SamplingParams::top_p);

  py::class_<AdamWConfig>(m, "AdamWConfig")
      .def(py::init<>())
      .def_readwrite("lr", &AdamWConfig::lr)
      .def_readwrite("beta1", &AdamWConfig::beta1)
      .def_readwrite("beta2", &AdamWConfig::beta2)
      .def_readwrite("eps", &AdamWConfig::eps)
      .def_readwrite("weight_decay", &AdamWConfig::weight_decay);

  m.def("cosine_lr", &cosine_lr, py::arg("step"), py::arg("warmup_steps"), py::arg("max_steps"),
        py::arg("lr_max"), py::arg("lr_min"), "Cosine decay with linear warmup");

  // Standalone sampler, so its statistics can be tested without a model.
  m.def(
      "sample_logits",
      [](NpArray<f32> logits, const SamplingParams& p, u64 seed, i64 draws) {
        auto b = logits.request();
        const auto n = static_cast<i64>(b.size);
        const auto* src = static_cast<const f32*>(b.ptr);
        py::array_t<i32> out(draws);
        auto ob = out.request();
        auto* dst = static_cast<i32*>(ob.ptr);
        {
          py::gil_scoped_release release;
          Sampler sampler(seed);
          std::vector<f32> scratch(static_cast<std::size_t>(n));
          for (i64 d = 0; d < draws; ++d) {
            // sample() scales logits in place, so each draw needs a fresh copy.
            std::memcpy(scratch.data(), src, static_cast<std::size_t>(n) * sizeof(f32));
            dst[d] = sampler.sample(scratch.data(), n, p);
          }
        }
        return out;
      },
      py::arg("logits"), py::arg("params"), py::arg("seed") = 0, py::arg("draws") = 1,
      "Draw token ids from logits using the configured sampling strategy");

  // ── raw GEMM (build verification) ─────────────────────────────────────────
  m.def("matmul_f32", &py_matmul_raw<f32>, py::arg("a"), py::arg("b"));
  m.def("matmul_f64", &py_matmul_raw<f64>, py::arg("a"), py::arg("b"));

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

  // ── differentiable ops (testing harness) ──────────────────────────────────
  register_ops<f32>(m, "_f32");
  register_ops<f64>(m, "_f64");

  // ── model ─────────────────────────────────────────────────────────────────
  register_model<f32>(m, "Model");
  register_model<f64>(m, "ModelF64");

  // ── optimizer ─────────────────────────────────────────────────────────────
  py::class_<AdamW<f32>>(m, "AdamW")
      .def(py::init([](Model<f32>& model, const AdamWConfig& cfg) {
             return std::make_unique<AdamW<f32>>(model.params().parameter_list(), cfg);
           }),
           py::arg("model"), py::arg("config"), py::keep_alive<1, 2>())
      .def(
          "clip_grad_norm",
          [](AdamW<f32>& o, f32 max_norm) {
            py::gil_scoped_release release;
            return o.clip_grad_norm(max_norm);
          },
          py::arg("max_norm"), "Rescale all grads to a global norm; returns the PRE-clip norm")
      .def(
          "step",
          [](AdamW<f32>& o, f32 lr) {
            py::gil_scoped_release release;
            o.step(lr);
          },
          py::arg("lr"))
      .def("zero_grad", &AdamW<f32>::zero_grad)
      .def_property_readonly("step_count", &AdamW<f32>::step_count);

  // ── generation ────────────────────────────────────────────────────────────
  py::class_<GenerationSession>(m, "GenerationSession")
      .def(py::init<const Model<f32>&, u64>(), py::arg("model"), py::arg("seed") = 0,
           py::keep_alive<1, 2>())
      .def(
          "prefill",
          [](GenerationSession& s, NpArray<i32> ids, i64 vocab_size) {
            auto ib = ids.request();
            py::array_t<f32> out(vocab_size);
            auto ob = out.request();
            {
              py::gil_scoped_release release;
              const f32* logits = s.prefill(static_cast<const i32*>(ib.ptr),
                                            static_cast<i64>(ib.size));
              std::memcpy(ob.ptr, logits, static_cast<std::size_t>(vocab_size) * sizeof(f32));
            }
            return out;
          },
          py::arg("ids"), py::arg("vocab_size"))
      .def(
          "step",
          [](GenerationSession& s, i32 token, const SamplingParams& p) {
            py::gil_scoped_release release;
            return s.step(token, p);
          },
          py::arg("token"), py::arg("params"), "Decode one token and sample the next id")
      .def(
          "sample_last",
          [](GenerationSession& s, const SamplingParams& p) {
            py::gil_scoped_release release;
            return s.sample_last(p);
          },
          py::arg("params"),
          "Sample from the most recent logits without advancing. Use this for the "
          "FIRST generated token after prefill() — step(prompt[-1]) would feed the "
          "prompt's last token twice.")
      .def("reset", &GenerationSession::reset)
      .def("reseed", &GenerationSession::reseed, py::arg("seed"))
      .def_property_readonly("position", &GenerationSession::position)
      .def_property_readonly("cache_bytes", &GenerationSession::cache_bytes);
}
