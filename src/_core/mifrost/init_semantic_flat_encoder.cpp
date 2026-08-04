#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <utility>

#include "mifrost/binding_kwargs.hpp"
#include "mifrost/bindings.hpp"
#include "mifrost/capsule_bridge.hpp"
#include "mifrost/core/encoders/flat/flat_composition.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {
namespace {

using CompactSemanticAtom = std::pair< int64_t, std::vector< int64_t > >;
using CompactSemanticLiteral = std::tuple< int64_t, std::vector< int64_t >, bool >;

SemanticAtom expand_compact_atom(CompactSemanticAtom value)
{
   return SemanticAtom{
      .predicate = value.first,
      .arguments = SemanticArguments(std::move(value.second)),
   };
}

SemanticLiteral expand_compact_literal(CompactSemanticLiteral value)
{
   auto [predicate, arguments, positive] = std::move(value);
   return SemanticLiteral{
      .atom =
         SemanticAtom{
            .predicate = predicate,
            .arguments = SemanticArguments(std::move(arguments)),
         },
      .positive = positive,
   };
}

SemanticFlatRelationInput make_compact_semantic_input(
   std::vector< std::string > objects,
   std::vector< CompactSemanticAtom > state_facts,
   std::vector< CompactSemanticLiteral > goals,
   std::vector< CompactSemanticAtom > actions,
   std::vector< std::vector< CompactSemanticLiteral > > subgoal_layers,
   std::vector< std::pair< int64_t, std::vector< CompactSemanticLiteral > > > history,
   std::optional< int64_t > history_max_steps
)
{
   SemanticFlatRelationInput result;
   result.objects = std::move(objects);
   result.state_facts.reserve(state_facts.size());
   for(auto& fact : state_facts) {
      result.state_facts.push_back(expand_compact_atom(std::move(fact)));
   }
   result.goals.reserve(goals.size());
   for(auto& goal : goals) {
      result.goals.push_back(expand_compact_literal(std::move(goal)));
   }
   result.actions.reserve(actions.size());
   for(auto& action : actions) {
      auto expanded = expand_compact_atom(std::move(action));
      result.actions.push_back(
         SemanticGroundAction{
            .action = expanded.predicate,
            .arguments = std::move(expanded.arguments),
         }
      );
   }
   result.subgoal_layers.reserve(subgoal_layers.size());
   for(auto& compact_layer : subgoal_layers) {
      std::vector< SemanticLiteral > layer;
      layer.reserve(compact_layer.size());
      for(auto& literal : compact_layer) {
         layer.push_back(expand_compact_literal(std::move(literal)));
      }
      result.subgoal_layers.push_back(std::move(layer));
   }
   result.history.reserve(history.size());
   for(auto& [dt, compact_literals] : history) {
      std::vector< SemanticLiteral > literals;
      literals.reserve(compact_literals.size());
      for(auto& literal : compact_literals) {
         literals.push_back(expand_compact_literal(std::move(literal)));
      }
      result.history.push_back(
         SemanticHistoryEntry{
            .dt = dt,
            .literals = std::move(literals),
         }
      );
   }
   result.history_max_steps = history_max_steps;
   return result;
}

}  // namespace

void init_semantic_flat_encoder(nb::module_& m)
{
   nb::enum_< FlatExternalComponent >(m, "FlatExternalComponent")
      .value("state_facts", FlatExternalComponent::state_facts)
      .value("goal_facts", FlatExternalComponent::goal_facts)
      .value("ground_actions", FlatExternalComponent::ground_actions)
      .value("transition_effects", FlatExternalComponent::transition_effects)
      .value("parent_relations", FlatExternalComponent::parent_relations)
      .value("root_action_nodes", FlatExternalComponent::root_action_nodes)
      .value("shared_state", FlatExternalComponent::shared_state);

   nb::enum_< FlatExternalMode >(m, "FlatExternalMode")
      .value("concurrent_internal", FlatExternalMode::concurrent_internal)
      .value("concurrent_internal_tree", FlatExternalMode::concurrent_internal_tree)
      .value("concurrent_internal_tree_rooted", FlatExternalMode::concurrent_internal_tree_rooted)
      .value(
         "concurrent_internal_comparison_tree",
         FlatExternalMode::concurrent_internal_comparison_tree
      )
      .value("concurrent_internal_action_tree", FlatExternalMode::concurrent_internal_action_tree)
      .value(
         "concurrent_internal_action_hybrid_tree",
         FlatExternalMode::concurrent_internal_action_hybrid_tree
      );

   nb::class_< FlatExternalModeContract >(m, "FlatExternalModeContract")
      .def_ro("mode", &FlatExternalModeContract::mode)
      .def_prop_ro(
         "name", [](const FlatExternalModeContract& contract) { return std::string(contract.name); }
      )
      .def_ro("required_components", &FlatExternalModeContract::required_components);
   m.def("flat_external_mode_contract", &flat_external_mode_contract, "mode"_a);
   m.def("flat_external_mode_contracts", [] {
      const auto contracts = flat_external_mode_contracts();
      return std::vector< FlatExternalModeContract >(contracts.begin(), contracts.end());
   });
   m.def(
      "flat_external_mode_satisfied",
      &flat_external_mode_satisfied,
      "mode"_a,
      "available_components"_a
   );
   m.def(
      "flat_external_mode_missing_components",
      &flat_external_mode_missing_components,
      "mode"_a,
      "available_components"_a
   );

   nb::enum_< GoalDerivation >(m, "GoalDerivation")
      .value("plain", GoalDerivation::plain)
      .value("satisfied", GoalDerivation::satisfied)
      .value("unsatisfied", GoalDerivation::unsatisfied)
      .value("added_satisfied", GoalDerivation::added_satisfied)
      .value("added_unsatisfied", GoalDerivation::added_unsatisfied);

   nb::enum_< TargetSource >(m, "TargetSource")
      .value("actions", TargetSource::actions)
      .value("goals", TargetSource::goals)
      .value("subgoals", TargetSource::subgoals)
      .value("states", TargetSource::states)
      .value("history", TargetSource::history);

   nb::class_< FlatRelationEncoderConfig >(m, "FlatRelationEncoderConfig")
      .def(nb::init<>())
      .def(
         "__init__",
         [](FlatRelationEncoderConfig* self, const nb::kwargs& kwargs) {
            new(self) FlatRelationEncoderConfig();
            apply_config_kwargs(*self, kwargs, "FlatRelationEncoderConfig");
         }
      )
      .def_rw("max_goal_level", &FlatRelationEncoderConfig::max_goal_level)
      .def_rw("support_literals", &FlatRelationEncoderConfig::support_literals)
      .def_rw("include_static", &FlatRelationEncoderConfig::include_static)
      .def_rw("export_node_names", &FlatRelationEncoderConfig::export_node_names)
      .def_rw(
         "ignore_zero_arity_relations", &FlatRelationEncoderConfig::ignore_zero_arity_relations
      )
      .def_rw(
         "use_predicate_virtual_nodes", &FlatRelationEncoderConfig::use_predicate_virtual_nodes
      )
      .def_rw("include_lgan_edges", &FlatRelationEncoderConfig::include_lgan_edges)
      .def_rw("lgan_anchor_sources", &FlatRelationEncoderConfig::lgan_anchor_sources)
      .def_rw("target_sources", &FlatRelationEncoderConfig::target_sources)
      .def_rw("target_symbol_prefix", &FlatRelationEncoderConfig::target_symbol_prefix)
      .def_rw("lgan_tn_edge_pos", &FlatRelationEncoderConfig::lgan_tn_edge_pos)
      .def_rw("lgan_nn_edge_pos", &FlatRelationEncoderConfig::lgan_nn_edge_pos)
      .def_rw("lgan_rr_edge_pos", &FlatRelationEncoderConfig::lgan_rr_edge_pos)
      .def_rw(
         "pack_relation_args_relation_major",
         &FlatRelationEncoderConfig::pack_relation_args_relation_major
      )
      .def_rw("goal_derivations", &FlatRelationEncoderConfig::goal_derivations);

   nb::enum_< SemanticPredicateCategory >(m, "SemanticPredicateCategory")
      .value("static", SemanticPredicateCategory::static_predicate)
      .value("fluent", SemanticPredicateCategory::fluent)
      .value("derived", SemanticPredicateCategory::derived);

   nb::class_< SemanticPredicateSpec >(m, "SemanticPredicateSpec")
      .def(
         nb::init< SemanticPredicateCategory, std::string, int64_t >(),
         "category"_a,
         "name"_a,
         "arity"_a
      )
      .def_rw("category", &SemanticPredicateSpec::category)
      .def_rw("name", &SemanticPredicateSpec::name)
      .def_rw("arity", &SemanticPredicateSpec::arity);

   nb::class_< SemanticActionSpec >(m, "SemanticActionSpec")
      .def(nb::init< std::string, int64_t >(), "name"_a, "arity"_a)
      .def_rw("name", &SemanticActionSpec::name)
      .def_rw("arity", &SemanticActionSpec::arity);

   nb::class_< SemanticAtom >(m, "SemanticAtom")
      .def(
         "__init__",
         [](SemanticAtom* self, int64_t predicate, std::vector< int64_t > arguments) {
            new(self) SemanticAtom{predicate, SemanticArguments(std::move(arguments))};
         },
         "predicate"_a,
         "arguments"_a
      )
      .def_rw("predicate", &SemanticAtom::predicate)
      .def_prop_rw(
         "arguments",
         [](const SemanticAtom& self) {
            return std::vector< int64_t >(self.arguments.begin(), self.arguments.end());
         },
         [](SemanticAtom& self, std::vector< int64_t > arguments) {
            self.arguments = SemanticArguments(std::move(arguments));
         }
      );

   nb::class_< SemanticLiteral >(m, "SemanticLiteral")
      .def(nb::init< SemanticAtom, bool >(), "atom"_a, "positive"_a = true)
      .def_rw("atom", &SemanticLiteral::atom)
      .def_rw("positive", &SemanticLiteral::positive);

   nb::class_< SemanticGroundAction >(m, "SemanticGroundAction")
      .def(
         "__init__",
         [](SemanticGroundAction* self, int64_t action, std::vector< int64_t > arguments) {
            new(self) SemanticGroundAction{action, SemanticArguments(std::move(arguments))};
         },
         "action"_a,
         "arguments"_a
      )
      .def_rw("action", &SemanticGroundAction::action)
      .def_prop_rw(
         "arguments",
         [](const SemanticGroundAction& self) {
            return std::vector< int64_t >(self.arguments.begin(), self.arguments.end());
         },
         [](SemanticGroundAction& self, std::vector< int64_t > arguments) {
            self.arguments = SemanticArguments(std::move(arguments));
         }
      );

   nb::class_< SemanticHistoryEntry >(m, "SemanticHistoryEntry")
      .def(nb::init< int64_t, std::vector< SemanticLiteral > >(), "dt"_a, "literals"_a)
      .def_rw("dt", &SemanticHistoryEntry::dt)
      .def_rw("literals", &SemanticHistoryEntry::literals);

   nb::class_< SemanticFlatRelationInput >(m, "SemanticFlatRelationInput")
      .def(nb::init<>())
      .def_static(
         "from_compact",
         &make_compact_semantic_input,
         "objects"_a,
         "state_facts"_a,
         "goals"_a,
         "actions"_a,
         "subgoal_layers"_a,
         "history"_a,
         "history_max_steps"_a.none() = nb::none()
      )
      .def_rw("objects", &SemanticFlatRelationInput::objects)
      .def_rw("state_facts", &SemanticFlatRelationInput::state_facts)
      .def_rw("goals", &SemanticFlatRelationInput::goals)
      .def_rw("actions", &SemanticFlatRelationInput::actions)
      .def_rw("subgoal_layers", &SemanticFlatRelationInput::subgoal_layers)
      .def_rw("history", &SemanticFlatRelationInput::history)
      .def_prop_rw(
         "history_max_steps",
         [](const SemanticFlatRelationInput& self) { return self.history_max_steps; },
         [](SemanticFlatRelationInput& self, const std::optional< int64_t >& value) {
            self.history_max_steps = value;
         },
         nb::arg().none()
      );

   nb::class_< SemanticFlatRelationEncoderEngine >(m, "SemanticFlatRelationEncoderEngine")
      .def(
         nb::init<
            std::vector< SemanticPredicateSpec >,
            std::vector< SemanticActionSpec >,
            FlatRelationEncoderConfig >(),
         "predicates"_a,
         "actions"_a,
         "config"_a = FlatRelationEncoderConfig{}
      )
      .def_prop_ro(
         "config", &SemanticFlatRelationEncoderEngine::get_config, nb::rv_policy::reference_internal
      )
      .def_prop_ro("predicates", &SemanticFlatRelationEncoderEngine::get_predicates)
      .def_prop_ro("actions", &SemanticFlatRelationEncoderEngine::get_actions)
      .def_prop_ro("relation_names", &SemanticFlatRelationEncoderEngine::get_relation_names)
      .def_prop_ro("relation_arities", &SemanticFlatRelationEncoderEngine::get_relation_arities)
      .def_prop_ro("relation_sources", &SemanticFlatRelationEncoderEngine::get_relation_sources)
      .def_prop_ro(
         "relation_logical_arities",
         &SemanticFlatRelationEncoderEngine::get_relation_logical_arities
      )
      .def_prop_ro(
         "relation_encoded_arities",
         &SemanticFlatRelationEncoderEngine::get_relation_encoded_arities
      )
      .def_prop_ro(
         "relation_slot_roles", &SemanticFlatRelationEncoderEngine::get_relation_slot_roles
      )
      .def_prop_ro(
         "relation_slot_role_offsets",
         &SemanticFlatRelationEncoderEngine::get_relation_slot_role_offsets
      )
      .def_prop_ro("slot_role_names", &SemanticFlatRelationEncoderEngine::get_slot_role_names)
      .def(
         "encode",
         nb::overload_cast< const SemanticFlatRelationInput& >(
            &SemanticFlatRelationEncoderEngine::encode, nb::const_
         ),
         "input"_a
      )
      .def(
         "encode",
         nb::overload_cast< const SemanticFlatRelationInput&, BatchBuilder& >(
            &SemanticFlatRelationEncoderEngine::encode, nb::const_
         ),
         "input"_a,
         "builder"_a
      )
      .def(
         "finalize_batch_encoding",
         &SemanticFlatRelationEncoderEngine::finalize_batch_encoding,
         "encoding"_a
      )
      .def("encode_batch", &SemanticFlatRelationEncoderEngine::encode_batch, "inputs"_a);

   m.def(
      "_flat_relation_config_capsule",
      [](const FlatRelationEncoderConfig& config) {
         auto* capsule = capsule_bridge::make_owned(
            FlatRelationEncoderConfig(config), capsule_bridge::config_name
         );
         if(capsule == nullptr) {
            throw nb::python_error();
         }
         return nb::steal< nb::object >(capsule);
      },
      "config"_a
   );
   m.def(
      "_consume_semantic_flat_input_capsule",
      [](nb::handle capsule) {
         return capsule_bridge::take< SemanticFlatRelationInput >(
            capsule.ptr(), capsule_bridge::input_name
         );
      },
      "capsule"_a
   );
   m.def(
      "_consume_semantic_flat_inputs_capsule",
      [](nb::handle capsule) {
         return capsule_bridge::take< std::vector< SemanticFlatRelationInput > >(
            capsule.ptr(), capsule_bridge::inputs_name
         );
      },
      "capsule"_a
   );
   m.def(
      "_consume_semantic_flat_engine_capsule",
      [](nb::handle capsule) {
         return capsule_bridge::take< SemanticFlatRelationEncoderEngine >(
            capsule.ptr(), capsule_bridge::engine_name
         );
      },
      "capsule"_a
   );
}

}  // namespace mifrost
