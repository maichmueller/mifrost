#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
#include <string_view>

#include "mifrost/bindings.hpp"
#include "mifrost/core/encoders/hetero/semantic_hgraph_encoder.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {
namespace {

void apply_config(SemanticHGraphEncoderConfig& config, const nb::kwargs& kwargs)
{
   for(const auto& [key_handle, value] : kwargs) {
      const auto key_string = nb::str(key_handle);
      const std::string_view key(key_string.c_str());
      if(key == "symbol_type_id") {
         config.symbol_type_id = nb::cast< std::string >(value);
      } else if(key == "target_symbol_prefix") {
         config.target_symbol_prefix = nb::cast< std::string >(value);
      } else if(key == "nullary_object_name") {
         config.nullary_object_name = nb::cast< std::string >(value);
      } else if(key == "lgan_tn_edge_pos") {
         config.lgan_tn_edge_pos = nb::cast< std::string >(value);
      } else if(key == "lgan_nn_edge_pos") {
         config.lgan_nn_edge_pos = nb::cast< std::string >(value);
      } else if(key == "lgan_rr_edge_pos") {
         config.lgan_rr_edge_pos = nb::cast< std::string >(value);
      } else if(key == "history_link_relation") {
         config.history_link_relation = nb::cast< std::string >(value);
      } else if(key == "max_goal_level") {
         config.max_goal_level = nb::cast< size_t >(value);
      } else if(key == "support_literals") {
         config.support_literals = nb::cast< bool >(value);
      } else if(key == "add_nullary_predicates") {
         config.add_nullary_predicates = nb::cast< bool >(value);
      } else if(key == "ignore_actions") {
         config.ignore_actions = nb::cast< bool >(value);
      } else if(key == "include_lgan_edges") {
         config.include_lgan_edges = nb::cast< bool >(value);
      } else if(key == "include_static") {
         config.include_static = nb::cast< bool >(value);
      } else if(key == "include_empty_edge_types") {
         config.include_empty_edge_types = nb::cast< bool >(value);
      } else if(key == "export_node_names") {
         config.export_node_names = nb::cast< bool >(value);
      } else if(key == "lgan_anchor_sources") {
         config.lgan_anchor_sources = nb::cast< std::set< TargetSource > >(value);
      } else if(key == "target_sources") {
         config.target_sources = nb::cast< std::set< TargetSource > >(value);
      } else if(key == "goal_derivations") {
         config.goal_derivations = nb::cast< std::set< GoalDerivation > >(value);
      } else {
         throw std::invalid_argument(
            "Unknown SemanticHGraphEncoderConfig kwarg '" + std::string(key) + "'"
         );
      }
   }
}

}  // namespace

void init_semantic_hgraph_encoder(nb::module_& m)
{
   nb::class_< SemanticHGraphEncoderConfig >(m, "SemanticHGraphEncoderConfig")
      .def(nb::init<>())
      .def(
         "__init__",
         [](SemanticHGraphEncoderConfig* self, const nb::kwargs& kwargs) {
            new(self) SemanticHGraphEncoderConfig();
            apply_config(*self, kwargs);
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
}

}  // namespace mifrost
