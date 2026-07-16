#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mifrost/binding_kwargs.hpp"
#include "mifrost/bindings.hpp"
#include "mifrost/capsule_bridge.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_horizon_encoder.hpp"
#include "mifrost/core/encoders/hetero/semantic_hgraph_encoder.hpp"
#include "mifrost/core/encoders/hetero/semantic_horizon_hgraph_encoder.hpp"
#include "mifrost/core/encoders/hetero/semantic_successor_hgraph_encoder.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {
void init_semantic_hgraph_encoder(nb::module_& m)
{
   nb::enum_< RootPolicy >(m, "RootPolicy")
      .value("include", RootPolicy::include)
      .value("encode_only", RootPolicy::encode_only)
      .value("exclude", RootPolicy::exclude);

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
      .def("update_relations", &SemanticHGraphEncoderEngine::update_relations, "relations"_a)
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

   nb::enum_< SemanticHorizonMode >(m, "SemanticHorizonEncoderMode")
      .value("full", SemanticHorizonMode::full)
      .value("delta", SemanticHorizonMode::delta)
      .value("action", SemanticHorizonMode::action);

   nb::class_< SemanticHorizonHGraphEncoderConfig, SemanticHGraphEncoderConfig >(
      m, "SemanticHorizonHGraphEncoderConfig"
   )
      .def(nb::init<>())
      .def(
         "__init__",
         [](SemanticHorizonHGraphEncoderConfig* self, const nb::kwargs& kwargs) {
            new(self) SemanticHorizonHGraphEncoderConfig();
            apply_config_kwargs(*self, kwargs, "SemanticHorizonHGraphEncoderConfig");
         }
      )
      .def_rw("transition_mode", &SemanticHorizonHGraphEncoderConfig::transition_mode)
      .def_rw("parent_relation", &SemanticHorizonHGraphEncoderConfig::parent_relation)
      .def_rw("sibling_relation", &SemanticHorizonHGraphEncoderConfig::sibling_relation)
      .def_rw("cousin_relation", &SemanticHorizonHGraphEncoderConfig::cousin_relation)
      .def_rw("enable_parent_relation", &SemanticHorizonHGraphEncoderConfig::enable_parent_relation)
      .def_rw(
         "enable_sibling_relation", &SemanticHorizonHGraphEncoderConfig::enable_sibling_relation
      )
      .def_rw("enable_cousin_relation", &SemanticHorizonHGraphEncoderConfig::enable_cousin_relation)
      .def_rw("root_policy", &SemanticHorizonHGraphEncoderConfig::root_policy);

   nb::class_< SemanticHorizonHGraphEncoderEngine >(m, "SemanticHorizonHGraphEncoderEngine")
      .def(
         nb::init<
            std::vector< SemanticPredicateSpec >,
            std::vector< SemanticActionSpec >,
            SemanticHorizonHGraphEncoderConfig >(),
         "predicates"_a,
         "actions"_a,
         "config"_a = SemanticHorizonHGraphEncoderConfig{}
      )
      .def_prop_ro(
         "config",
         &SemanticHorizonHGraphEncoderEngine::get_config,
         nb::rv_policy::reference_internal
      )
      .def_prop_ro("predicates", &SemanticHorizonHGraphEncoderEngine::get_predicates)
      .def_prop_ro("actions", &SemanticHorizonHGraphEncoderEngine::get_actions)
      .def_prop_ro("relation_arities", &SemanticHorizonHGraphEncoderEngine::get_relation_arities)
      .def("update_relations", &SemanticHorizonHGraphEncoderEngine::update_relations, "relations"_a)
      .def(
         "encode",
         nb::overload_cast< const SemanticTransitionDAG& >(
            &SemanticHorizonHGraphEncoderEngine::encode, nb::const_
         ),
         "dag"_a
      )
      .def(
         "encode",
         nb::overload_cast< const SemanticTransitionDAG&, BatchBuilder& >(
            &SemanticHorizonHGraphEncoderEngine::encode, nb::const_
         ),
         "dag"_a,
         "builder"_a
      )
      .def("encode_batch", &SemanticHorizonHGraphEncoderEngine::encode_batch, "dags"_a);

   nb::class_< SemanticFlatHorizonEncoderConfig, FlatRelationEncoderConfig >(
      m, "SemanticFlatHorizonEncoderConfig"
   )
      .def(nb::init<>())
      .def(
         "__init__",
         [](SemanticFlatHorizonEncoderConfig* self, const nb::kwargs& kwargs) {
            new(self) SemanticFlatHorizonEncoderConfig();
            apply_config_kwargs(*self, kwargs, "SemanticFlatHorizonEncoderConfig");
         }
      )
      .def_rw("ignore_actions", &SemanticFlatHorizonEncoderConfig::ignore_actions)
      .def_rw("transition_mode", &SemanticFlatHorizonEncoderConfig::transition_mode)
      .def_rw("parent_relation", &SemanticFlatHorizonEncoderConfig::parent_relation)
      .def_rw("sibling_relation", &SemanticFlatHorizonEncoderConfig::sibling_relation)
      .def_rw("cousin_relation", &SemanticFlatHorizonEncoderConfig::cousin_relation)
      .def_rw("enable_parent_relation", &SemanticFlatHorizonEncoderConfig::enable_parent_relation)
      .def_rw("enable_sibling_relation", &SemanticFlatHorizonEncoderConfig::enable_sibling_relation)
      .def_rw("enable_cousin_relation", &SemanticFlatHorizonEncoderConfig::enable_cousin_relation)
      .def_rw("root_policy", &SemanticFlatHorizonEncoderConfig::root_policy);

   nb::class_< SemanticFlatHorizonEncoderEngine >(m, "SemanticFlatHorizonEncoderEngine")
      .def(
         nb::init<
            std::vector< SemanticPredicateSpec >,
            std::vector< SemanticActionSpec >,
            SemanticFlatHorizonEncoderConfig >(),
         "predicates"_a,
         "actions"_a,
         "config"_a = SemanticFlatHorizonEncoderConfig{}
      )
      .def_prop_ro(
         "config", &SemanticFlatHorizonEncoderEngine::get_config, nb::rv_policy::reference_internal
      )
      .def_prop_ro("predicates", &SemanticFlatHorizonEncoderEngine::get_predicates)
      .def_prop_ro("actions", &SemanticFlatHorizonEncoderEngine::get_actions)
      .def_prop_ro("relation_names", &SemanticFlatHorizonEncoderEngine::get_relation_names)
      .def_prop_ro("relation_arities", &SemanticFlatHorizonEncoderEngine::get_relation_arities)
      .def_prop_ro("relation_sources", &SemanticFlatHorizonEncoderEngine::get_relation_sources)
      .def_prop_ro(
         "relation_logical_arities", &SemanticFlatHorizonEncoderEngine::get_relation_logical_arities
      )
      .def_prop_ro(
         "relation_encoded_arities", &SemanticFlatHorizonEncoderEngine::get_relation_encoded_arities
      )
      .def_prop_ro(
         "relation_slot_roles", &SemanticFlatHorizonEncoderEngine::get_relation_slot_roles
      )
      .def_prop_ro(
         "relation_slot_role_offsets",
         &SemanticFlatHorizonEncoderEngine::get_relation_slot_role_offsets
      )
      .def_prop_ro("slot_role_names", &SemanticFlatHorizonEncoderEngine::get_slot_role_names)
      .def(
         "encode",
         nb::overload_cast< const SemanticTransitionDAG& >(
            &SemanticFlatHorizonEncoderEngine::encode, nb::const_
         ),
         "dag"_a
      )
      .def(
         "encode",
         nb::overload_cast< const SemanticTransitionDAG&, BatchBuilder& >(
            &SemanticFlatHorizonEncoderEngine::encode, nb::const_
         ),
         "dag"_a,
         "builder"_a
      )
      .def("encode_batch", &SemanticFlatHorizonEncoderEngine::encode_batch, "dags"_a)
      .def(
         "finalize_batch_encoding",
         &SemanticFlatHorizonEncoderEngine::finalize_batch_encoding,
         "encoding"_a
      );

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
         "update_relations", &SemanticSuccessorHGraphEncoderEngine::update_relations, "relations"_a
      )
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
   m.def(
      "_semantic_successor_hgraph_config_capsule",
      [](const SemanticSuccessorHGraphEncoderConfig& config) {
         auto* capsule = capsule_bridge::make_owned(
            SemanticSuccessorHGraphEncoderConfig(config),
            capsule_bridge::successor_hgraph_config_name
         );
         if(capsule == nullptr) {
            throw nb::python_error();
         }
         return nb::steal< nb::object >(capsule);
      },
      "config"_a
   );
   m.def(
      "_consume_semantic_successor_hgraph_engine_capsule",
      [](nb::handle capsule) {
         return capsule_bridge::take< SemanticSuccessorHGraphEncoderEngine >(
            capsule.ptr(), capsule_bridge::successor_hgraph_engine_name
         );
      },
      "capsule"_a
   );
   m.def(
      "_semantic_horizon_hgraph_config_capsule",
      [](const SemanticHorizonHGraphEncoderConfig& config) {
         auto* capsule = capsule_bridge::make_owned(
            SemanticHorizonHGraphEncoderConfig(config), capsule_bridge::horizon_hgraph_config_name
         );
         if(capsule == nullptr) {
            throw nb::python_error();
         }
         return nb::steal< nb::object >(capsule);
      },
      "config"_a
   );
   m.def(
      "_consume_semantic_horizon_hgraph_engine_capsule",
      [](nb::handle capsule) {
         return capsule_bridge::take< SemanticHorizonHGraphEncoderEngine >(
            capsule.ptr(), capsule_bridge::horizon_hgraph_engine_name
         );
      },
      "capsule"_a
   );
   m.def(
      "_semantic_flat_horizon_config_capsule",
      [](const SemanticFlatHorizonEncoderConfig& config) {
         auto* capsule = capsule_bridge::make_owned(
            SemanticFlatHorizonEncoderConfig(config), capsule_bridge::flat_horizon_config_name
         );
         if(capsule == nullptr) {
            throw nb::python_error();
         }
         return nb::steal< nb::object >(capsule);
      },
      "config"_a
   );
   m.def(
      "_consume_semantic_flat_horizon_engine_capsule",
      [](nb::handle capsule) {
         return capsule_bridge::take< SemanticFlatHorizonEncoderEngine >(
            capsule.ptr(), capsule_bridge::flat_horizon_engine_name
         );
      },
      "capsule"_a
   );
}

}  // namespace mifrost
