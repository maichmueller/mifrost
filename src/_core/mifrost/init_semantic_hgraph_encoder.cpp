#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mifrost/binding_kwargs.hpp"
#include "mifrost/bindings.hpp"
#include "mifrost/capsule_bridge.hpp"
#include "mifrost/core/encoders/hetero/semantic_hgraph_encoder.hpp"
#include "mifrost/core/encoders/hetero/semantic_successor_hgraph_encoder.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {
void init_semantic_hgraph_encoder(nb::module_& m)
{
   nb::class_< SemanticHGraphEncoderConfig >(m, "SemanticHGraphEncoderConfig")
      .def(nb::init<>())
      .def(
         "__init__",
         [](SemanticHGraphEncoderConfig* self, const nb::kwargs& kwargs) {
            new(self) SemanticHGraphEncoderConfig();
            apply_config_kwargs(*self, kwargs, "SemanticHGraphEncoderConfig");
         }
      )
      .def_rw("symbol_type_id", &SemanticHGraphEncoderConfig::symbol_type_id)
      .def_rw("target_symbol_prefix", &SemanticHGraphEncoderConfig::target_symbol_prefix)
      .def_rw("nullary_object_name", &SemanticHGraphEncoderConfig::nullary_object_name)
      .def_rw("lgan_tn_edge_pos", &SemanticHGraphEncoderConfig::lgan_tn_edge_pos)
      .def_rw("lgan_nn_edge_pos", &SemanticHGraphEncoderConfig::lgan_nn_edge_pos)
      .def_rw("lgan_rr_edge_pos", &SemanticHGraphEncoderConfig::lgan_rr_edge_pos)
      .def_rw("history_link_relation", &SemanticHGraphEncoderConfig::history_link_relation)
      .def_rw("max_goal_level", &SemanticHGraphEncoderConfig::max_goal_level)
      .def_rw("support_literals", &SemanticHGraphEncoderConfig::support_literals)
      .def_rw("add_nullary_predicates", &SemanticHGraphEncoderConfig::add_nullary_predicates)
      .def_rw("ignore_actions", &SemanticHGraphEncoderConfig::ignore_actions)
      .def_rw("include_lgan_edges", &SemanticHGraphEncoderConfig::include_lgan_edges)
      .def_rw("include_static", &SemanticHGraphEncoderConfig::include_static)
      .def_rw("include_empty_edge_types", &SemanticHGraphEncoderConfig::include_empty_edge_types)
      .def_rw("export_node_names", &SemanticHGraphEncoderConfig::export_node_names)
      .def_rw("lgan_anchor_sources", &SemanticHGraphEncoderConfig::lgan_anchor_sources)
      .def_rw("target_sources", &SemanticHGraphEncoderConfig::target_sources)
      .def_rw("goal_derivations", &SemanticHGraphEncoderConfig::goal_derivations);

   nb::class_< SemanticHGraphEncoderEngine >(m, "SemanticHGraphEncoderEngine")
      .def(
         nb::init<
            std::vector< SemanticPredicateSpec >,
            std::vector< SemanticActionSpec >,
            SemanticHGraphEncoderConfig >(),
         "predicates"_a,
         "actions"_a,
         "config"_a = SemanticHGraphEncoderConfig{}
      )
      .def_prop_ro(
         "config", &SemanticHGraphEncoderEngine::get_config, nb::rv_policy::reference_internal
      )
      .def_prop_ro("predicates", &SemanticHGraphEncoderEngine::get_predicates)
      .def_prop_ro("actions", &SemanticHGraphEncoderEngine::get_actions)
      .def_prop_ro("relation_arities", &SemanticHGraphEncoderEngine::get_relation_arities)
      .def(
         "encode",
         nb::overload_cast< const SemanticFlatRelationInput& >(
            &SemanticHGraphEncoderEngine::encode, nb::const_
         ),
         "input"_a
      )
      .def(
         "encode",
         nb::overload_cast< const SemanticFlatRelationInput&, BatchBuilder& >(
            &SemanticHGraphEncoderEngine::encode, nb::const_
         ),
         "input"_a,
         "builder"_a
      )
      .def("encode_batch", &SemanticHGraphEncoderEngine::encode_batch, "inputs"_a);

   nb::enum_< SemanticSuccessorMode >(m, "SemanticSuccessorEncoderMode")
      .value("full", SemanticSuccessorMode::full)
      .value("delta", SemanticSuccessorMode::delta);

   nb::class_< SemanticSuccessorHGraphEncoderConfig, SemanticHGraphEncoderConfig >(
      m, "SemanticSuccessorHGraphEncoderConfig"
   )
      .def(nb::init<>())
      .def(
         "__init__",
         [](SemanticSuccessorHGraphEncoderConfig* self, const nb::kwargs& kwargs) {
            new(self) SemanticSuccessorHGraphEncoderConfig();
            apply_config_kwargs(*self, kwargs, "SemanticSuccessorHGraphEncoderConfig");
         }
      )
      .def_rw("successor_mode", &SemanticSuccessorHGraphEncoderConfig::successor_mode)
      .def_rw("successor_suffix", &SemanticSuccessorHGraphEncoderConfig::successor_suffix)
      .def_rw(
         "include_successor_goal_satisfaction",
         &SemanticSuccessorHGraphEncoderConfig::include_successor_goal_satisfaction
      );

   nb::class_< SemanticSuccessorHGraphEncoderEngine >(m, "SemanticSuccessorHGraphEncoderEngine")
      .def(
         nb::init<
            std::vector< SemanticPredicateSpec >,
            std::vector< SemanticActionSpec >,
            SemanticSuccessorHGraphEncoderConfig >(),
         "predicates"_a,
         "actions"_a,
         "config"_a = SemanticSuccessorHGraphEncoderConfig{}
      )
      .def_prop_ro(
         "config",
         &SemanticSuccessorHGraphEncoderEngine::get_config,
         nb::rv_policy::reference_internal
      )
      .def_prop_ro("predicates", &SemanticSuccessorHGraphEncoderEngine::get_predicates)
      .def_prop_ro("actions", &SemanticSuccessorHGraphEncoderEngine::get_actions)
      .def_prop_ro("relation_arities", &SemanticSuccessorHGraphEncoderEngine::get_relation_arities)
      .def(
         "encode",
         nb::overload_cast< const SemanticFlatRelationInput&, const SemanticFlatRelationInput& >(
            &SemanticSuccessorHGraphEncoderEngine::encode, nb::const_
         ),
         "current"_a,
         "successor"_a
      )
      .def(
         "encode",
         nb::overload_cast<
            const SemanticFlatRelationInput&,
            const SemanticFlatRelationInput&,
            BatchBuilder& >(&SemanticSuccessorHGraphEncoderEngine::encode, nb::const_),
         "current"_a,
         "successor"_a,
         "builder"_a
      )
      .def(
         "encode_batch",
         &SemanticSuccessorHGraphEncoderEngine::encode_batch,
         "currents"_a,
         "successors"_a
      );

   m.def(
      "_semantic_hgraph_config_capsule",
      [](const SemanticHGraphEncoderConfig& config) {
         auto* capsule = capsule_bridge::make_owned(
            SemanticHGraphEncoderConfig(config), capsule_bridge::hgraph_config_name
         );
         if(capsule == nullptr) {
            throw nb::python_error();
         }
         return nb::steal< nb::object >(capsule);
      },
      "config"_a
   );
   m.def(
      "_consume_semantic_hgraph_engine_capsule",
      [](nb::handle capsule) {
         return capsule_bridge::take< SemanticHGraphEncoderEngine >(
            capsule.ptr(), capsule_bridge::hgraph_engine_name
         );
      },
      "capsule"_a
   );
}

}  // namespace mifrost
