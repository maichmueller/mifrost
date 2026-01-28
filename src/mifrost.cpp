#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <filesystem>
#include <mimir/formalism/problem.hpp>
#include <mimir/search/axiom_evaluators/grounded/grounded.hpp>
#include <mimir/search/axiom_evaluators/interface.hpp>
#include <mimir/search/grounders/lifted.hpp>
#include <mimir/search/state_repository.hpp>
#include <new>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/hgraph_stream_encoder.hpp"
#include "mifrost/core/nanobind_unordered_dense.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {
namespace {

}  // namespace

class PyStreamEncoder: public StreamEncoderInterface {
  public:
   NB_TRAMPOLINE(StreamEncoderInterface, 1);

   void encode_state(const mimir::search::State& state, BatchBuilder& builder) override
   {
      NB_OVERRIDE_PURE(encode_state, state, builder);
   }
};

NB_MODULE(_core, m)
{
   m.doc() = "Mifrost: High-performance C++ Stream Encoding for PyG";

   // Low-level C++ interfaces (typed boundary; Python wrapper assembles PyG).
   nb::class_< StreamEncoderInterface, PyStreamEncoder >(m, "StreamEncoder")
      .def(nb::init<>())
      .def("encode_state_into", &StreamEncoderInterface::encode_state)
      .def("encode_state", [](StreamEncoderInterface& e, const mimir::search::State& state) {
         BatchBuilder builder;
         e.encode_state(state, builder);
         return builder.build();
      });

   nb::class_< BatchBuilder >(m, "BatchBuilder")
      .def(nb::init<>())
      .def(
         "add_node_features",
         [](BatchBuilder& b,
            const std::string& type,
            const std::string& attr,
            nb::ndarray< float, nb::c_contig > data) {
            // Helper lambda to dispatch ndarray to span
            // Assuming data is [N, F]
            int feature_dim = data.shape(1);
            size_t size = data.size();
            std::span< const float > span(data.data(), size);
            b.add_node_features(type, attr, span, feature_dim);
         },
         "node_type"_a,
         "attr_name"_a,
         "data"_a
      )
      .def(
         "add_edges",
         [](BatchBuilder& b,
            const std::string& src_type,
            const std::string& rel_type,
            const std::string& dst_type,
            nb::ndarray< int64_t, nb::c_contig > src,
            nb::ndarray< int64_t, nb::c_contig > dst) {
            std::span< const int64_t > src_span(src.data(), src.size());
            std::span< const int64_t > dst_span(dst.data(), dst.size());
            b.add_edges(src_type, rel_type, dst_type, src_span, dst_span);
         },
         "src_type"_a,
         "rel_type"_a,
         "dst_type"_a,
         "src_indices"_a,
         "dst_indices"_a
      )
      .def("next_graph", &BatchBuilder::next_graph)
      .def("build", &BatchBuilder::build)
      .def("build_parts", &BatchBuilder::build_parts);

   // Encoder-agnostic goal container; variant order biases fluent literals first.
   nb::class_< GoalInputs >(m, "GoalInputs")
      .def(nb::init<>())
      .def(nb::init< const std::vector< GoalInputs::AnyGoalLiteral >& >(), "goals"_a)
      .def(
         nb::init< const std::vector< GoalInputs::AnyGoalLiteral >&, int >(), "goals"_a, "level"_a
      )
      .def_rw("static_goals", &GoalInputs::static_goals)
      .def_rw("fluent_goals", &GoalInputs::fluent_goals)
      .def_rw("derived_goals", &GoalInputs::derived_goals)
      .def_rw("static_goal_levels", &GoalInputs::static_goal_levels)
      .def_rw("fluent_goal_levels", &GoalInputs::fluent_goal_levels)
      .def_rw("derived_goal_levels", &GoalInputs::derived_goal_levels);

   nb::class_< HGraphEncoderEngine, StreamEncoderInterface >(m, "HGraphEncoderEngine")
      .def(
         "__init__",
         [](HGraphEncoderEngine* self,
            mimir::formalism::Domain domain,
            const std::string& symbol_type_id,
            bool ignore_actions,
            bool add_nullary_predicates,
            bool include_lgan_edges,
            bool include_static,
            int max_goal_level,
            bool support_literals,
            const std::string& nullary_object_name,
            const std::string& lgan_nn_edge_pos) {
            HGraphEncoderEngine::Config config;
            config.symbol_type_id = symbol_type_id;
            config.ignore_actions = ignore_actions;
            config.add_nullary_predicates = add_nullary_predicates;
            config.include_lgan_edges = include_lgan_edges;
            config.include_static = include_static;
            config.max_goal_level = max_goal_level;
            config.support_literals = support_literals;
            config.nullary_object_name = nullary_object_name;
            config.lgan_nn_edge_pos = lgan_nn_edge_pos;
            // Placement-new constructs in the already-allocated Python instance.
            new(self) HGraphEncoderEngine(std::move(domain), config);
         },
         "domain"_a,
         "symbol_type_id"_a = "_symbol_",
         "ignore_actions"_a = true,
         "add_nullary_predicates"_a = false,
         "include_lgan_edges"_a = false,
         "include_static"_a = true,
         "max_goal_level"_a = 0,
         "support_literals"_a = false,
         "nullary_object_name"_a = RelationFormatter::kDefaultNullarySymbolName,
         "lgan_nn_edge_pos"_a = "lgan_nn"
      )
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder, const mimir::search::State& state) {
            BatchBuilder builder;
            encoder.encode(state, builder);
            return builder.build_parts();
         },
         "state"_a
      )
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions) {
            BatchBuilder builder;
            encoder.encode(state, goals, actions, builder);
            return builder.build_parts();
         },
         "state"_a,
         "goals"_a,
         "actions"_a
      )
      // Overloads that append into a caller-provided builder for streaming.
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder,
            const mimir::search::State& state,
            BatchBuilder& builder) { encoder.encode(state, builder); },
         "state"_a,
         "builder"_a
      )
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions,
            BatchBuilder& builder) { encoder.encode(state, goals, actions, builder); },
         "state"_a,
         "goals"_a,
         "actions"_a,
         "builder"_a
      );
}

}  // namespace mifrost
