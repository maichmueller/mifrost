#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/hgraph_stream_encoder.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

NB_MODULE(_core, m) {
  m.doc() = "Mifrost: High-performance C++ Stream Encoding for PyG";

  nb::class_<BatchBuilder>(m, "BatchBuilder")
      .def(nb::init<>())
      .def(
          "add_node_features",
          [](BatchBuilder &b, const std::string &type, const std::string &attr,
             nb::ndarray<float, nb::c_contig> data) {
            // Helper lambda to dispatch ndarray to span
            // Assuming data is [N, F]
            int feature_dim = data.shape(1);
            size_t size = data.size();
            std::span<const float> span(data.data(), size);
            b.add_node_features(type, attr, span, feature_dim);
          },
          "node_type"_a, "attr_name"_a, "data"_a)
      .def(
          "add_edges",
          [](BatchBuilder &b, const std::string &src_type,
             const std::string &rel_type, const std::string &dst_type,
             nb::ndarray<int64_t, nb::c_contig> src,
             nb::ndarray<int64_t, nb::c_contig> dst) {
            std::span<const int64_t> src_span(src.data(), src.size());
            std::span<const int64_t> dst_span(dst.data(), dst.size());
            b.add_edges(src_type, rel_type, dst_type, src_span, dst_span);
          },
          "src_type"_a, "rel_type"_a, "dst_type"_a, "src_indices"_a,
          "dst_indices"_a)
      .def("next_graph", &BatchBuilder::next_graph)
      .def("build", &BatchBuilder::build);

  nb::class_<HGraphStreamEncoder>(m, "HGraphStreamEncoder")
      .def(nb::init<mimir::formalism::Domain>())
      .def("encode_state",
           [](HGraphStreamEncoder &e, const mimir::search::State &state) {
             BatchBuilder builder;
             e.encode_state(state, builder);
             return builder.build();
           });
}

} // namespace mifrost
