#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "mifrost/backends/pytyr/semantic_flat_encoder.hpp"
#include "mifrost/capsule_bridge.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_horizon_encoder.hpp"
#include "mifrost/core/encoders/hetero/semantic_hgraph_encoder.hpp"
#include "mifrost/core/encoders/hetero/semantic_horizon_hgraph_encoder.hpp"
#include "mifrost/core/encoders/hetero/semantic_successor_hgraph_encoder.hpp"
#include "mifrost/core/encoders/homo/semantic_color_encoder.hpp"
#include "mifrost/core/semantic/views.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost::pytyr {
namespace {

using CompactLiteral = std::tuple< int64_t, std::vector< int64_t >, bool >;
using RawLiteral = std::tuple< int64_t, int64_t, std::vector< int64_t >, bool >;
using AdapterLiteral = std::variant< CompactLiteral, RawLiteral >;
using CompactHistoryEntry = std::tuple< int64_t, std::vector< AdapterLiteral > >;

SemanticLiteral expand_literal(const CompactLiteral& value)
{
   const auto& [predicate, arguments, positive] = value;
   return SemanticLiteral{
      SemanticAtom{predicate, SemanticArguments(arguments)},
      positive,
   };
}

SemanticLiteral
expand_literal(const SemanticPlanningTaskAdapter& adapter, const AdapterLiteral& value)
{
   if(const auto* compact = std::get_if< CompactLiteral >(&value)) {
      return expand_literal(*compact);
   }
   const auto& [category, predicate, objects, positive] = std::get< RawLiteral >(value);
   return adapter.make_raw_literal(category, predicate, objects, positive);
}

std::vector< SemanticLiteral > expand_literals(
   const SemanticPlanningTaskAdapter& adapter,
   const std::vector< AdapterLiteral >& values
)
{
   std::vector< SemanticLiteral > result;
   result.reserve(values.size());
   for(const auto& value : values) {
      result.push_back(expand_literal(adapter, value));
   }
   return result;
}

void apply_optional_lanes(
   const SemanticPlanningTaskAdapter& adapter,
   SemanticFlatRelationInput& input,
   const std::optional< std::vector< AdapterLiteral > >& goals,
   const std::vector< std::vector< AdapterLiteral > >& subgoal_layers,
   const std::vector< CompactHistoryEntry >& history,
   std::optional< int64_t > history_max_steps
)
{
   if(goals) {
      input.goals = expand_literals(adapter, *goals);
      input.use_default_goals = false;
   }
   input.subgoal_layers.reserve(subgoal_layers.size());
   for(const auto& layer : subgoal_layers) {
      input.subgoal_layers.push_back(expand_literals(adapter, layer));
   }
   input.history.reserve(history.size());
   for(const auto& [dt, literals] : history) {
      input.history.push_back(SemanticHistoryEntry{dt, expand_literals(adapter, literals)});
   }
   input.history_max_steps = history_max_steps;
}

template < typename State >
std::vector< SemanticFlatRelationInput > make_inputs(
   const SemanticPlanningTaskAdapter& adapter,
   const std::vector< State >& states,
   const std::vector< std::vector< tyr::formalism::planning::GroundActionView > >& actions,
   const std::vector< std::optional< std::vector< AdapterLiteral > > >& goals,
   const std::vector< std::vector< std::vector< AdapterLiteral > > >& subgoal_layers,
   const std::vector< std::vector< CompactHistoryEntry > >& history,
   std::optional< int64_t > history_max_steps
)
{
   const auto validate_size = [&](size_t size, std::string_view lane) {
      if(size != 0 and size != states.size()) {
         throw std::invalid_argument(
            "PyTyr semantic flat " + std::string(lane) + " batch length must match states"
         );
      }
   };
   validate_size(actions.size(), "actions");
   validate_size(goals.size(), "goals");
   validate_size(subgoal_layers.size(), "subgoal_layers");
   validate_size(history.size(), "history");

   std::vector< SemanticFlatRelationInput > inputs;
   inputs.reserve(states.size());
   const std::vector< tyr::formalism::planning::GroundActionView > no_actions;
   const std::optional< std::vector< AdapterLiteral > > no_goals;
   const std::vector< std::vector< AdapterLiteral > > no_subgoals;
   const std::vector< CompactHistoryEntry > no_history;
   for(size_t index = 0; index < states.size(); ++index) {
      auto input = adapter.make_input(states[index], actions.empty() ? no_actions : actions[index]);
      apply_optional_lanes(
         adapter,
         input,
         goals.empty() ? no_goals : goals[index],
         subgoal_layers.empty() ? no_subgoals : subgoal_layers[index],
         history.empty() ? no_history : history[index],
         history_max_steps
      );
      inputs.push_back(std::move(input));
   }
   return inputs;
}

/**
 * A canonical family engine that lives on the PyTyr side of the ABI boundary.
 *
 * The neutral engines are plain C++ and both modules link the same neutral
 * library, so an engine can be constructed and *used* here. That is what makes
 * the direct-View path possible for PyTyr: the state and its actions never have
 * to become an owning `SemanticFlatRelationInput` just to reach the algorithm,
 * because the algorithm runs in the module that can see the Tyr types. Only the
 * finished, planner-neutral `BatchEncoding` crosses to the core module, and it
 * crosses as a capsule because PyTyr and Pymimir ship incompatible nanobind ABI
 * generations.
 *
 * The goal, subgoal and history lanes are still expanded into owned
 * `SemanticLiteral` records: those literals arrive from Python as compact
 * tuples, so there is no native value to borrow from in the first place.
 */
template < typename Engine >
class DirectEncoder {
  public:
   DirectEncoder(const SemanticPlanningTaskAdapter& adapter, typename Engine::Config config)
       : adapter_(&adapter), engine_(adapter.get_task_context(), std::move(config))
   {
   }

   [[nodiscard]] const Engine& engine() const noexcept { return engine_; }

   /** Encode one state through granular Views over the caller's own values. */
   template < typename State >
   [[nodiscard]] BatchBuilder::BatchEncoding encode(
      const State& state,
      const std::vector< tyr::formalism::planning::GroundActionView >& actions,
      const std::optional< std::vector< AdapterLiteral > >& goals,
      const std::vector< std::vector< AdapterLiteral > >& subgoal_layers,
      const std::vector< CompactHistoryEntry >& history,
      std::optional< int64_t > history_max_steps
   ) const
   {
      const auto state_view = adapter_->make_view(state);
      const auto action_views = adapter_->make_action_views(actions);

      // Owned only where the input itself is already owned: literals handed in
      // from Python. Each vector is borrowed by a semantic View below.
      const auto goal_records = goals ? expand_literals(*adapter_, *goals)
                                      : std::vector< SemanticLiteral >{};
      std::vector< std::vector< SemanticLiteral > > layer_records;
      layer_records.reserve(subgoal_layers.size());
      for(const auto& layer : subgoal_layers) {
         layer_records.push_back(expand_literals(*adapter_, layer));
      }
      std::vector< SemanticHistoryEntry > history_records;
      history_records.reserve(history.size());
      for(const auto& [dt, literals] : history) {
         history_records.push_back(SemanticHistoryEntry{dt, expand_literals(*adapter_, literals)});
      }

      const semantic::LiteralsView goals_view{
         std::span{goals ? goal_records : adapter_->get_task_context()->default_goals}
      };
      const semantic::SubgoalLayersView layers_view{std::span{layer_records}};
      const semantic::HistoryView history_view{std::span{history_records}};

      // No explicit goals and no other optional lane: let the engine apply its
      // own default-goal handling rather than restating it here.
      if(not goals and layer_records.empty() and history_records.empty()) {
         return engine_.encode(state_view, action_views);
      }
      // Color has no history lane at all; every other family takes one.
      if constexpr(requires {
                      engine_.encode(
                         state_view,
                         goals_view,
                         layers_view,
                         action_views,
                         history_view,
                         history_max_steps
                      );
                   }) {
         return engine_.encode(
            state_view, goals_view, layers_view, action_views, history_view, history_max_steps
         );
      } else {
         if(not history_records.empty()) {
            throw std::invalid_argument("this PyTyr encoder family does not accept a history lane");
         }
         (void) history_view;
         (void) history_max_steps;
         return engine_.encode(state_view, goals_view, layers_view, action_views);
      }
   }

  private:
   const SemanticPlanningTaskAdapter* adapter_;
   Engine engine_;
};

}  // namespace

NB_MODULE(_pytyr_adapter, m)
{
#ifdef NDEBUG
   nb::set_leak_warnings(false);
#endif
   // PyTyr and Pymimir currently ship different nanobind ABI generations.
   // This module uses PyTyr's runtime and transfers neutral C++ values through
   // ownership-safe capsules instead of crossing incompatible registries.
   nb::module_::import_("pytyr._pytyr");

   using Adapter = SemanticPlanningTaskAdapter;
   using PlanningTask = tyr::formalism::planning::PlanningTask;
   using GroundAction = tyr::formalism::planning::GroundActionView;
   using LiftedState = tyr::planning::StateView< tyr::planning::LiftedTag >;
   using GroundState = tyr::planning::StateView< tyr::planning::GroundTag >;

   const auto owned_capsule = [](auto value, const char* name) {
      auto* capsule = capsule_bridge::make_owned(std::move(value), name);
      if(capsule == nullptr) {
         throw nb::python_error();
      }
      return nb::steal< nb::object >(capsule);
   };

   nb::class_< Adapter >(m, "_NativeSemanticPlanningTaskAdapter")
      .def(
         "__init__",
         [](Adapter* self, const PlanningTask& task) { new(self) Adapter(task); },
         "task"_a
      )
      .def(
         "_make_input_capsule",
         [owned_capsule](
            const Adapter& self,
            const LiftedState& state,
            const std::vector< GroundAction >& actions,
            const std::optional< std::vector< AdapterLiteral > >& goals,
            const std::vector< std::vector< AdapterLiteral > >& subgoal_layers,
            const std::vector< CompactHistoryEntry >& history,
            std::optional< int64_t > history_max_steps
         ) {
            auto input = self.make_input(state, actions);
            apply_optional_lanes(self, input, goals, subgoal_layers, history, history_max_steps);
            return owned_capsule(std::move(input), capsule_bridge::input_name);
         },
         "state"_a,
         "actions"_a = std::vector< GroundAction >{},
         "goals"_a = nb::none(),
         "subgoal_layers"_a = std::vector< std::vector< AdapterLiteral > >{},
         "history"_a = std::vector< CompactHistoryEntry >{},
         "history_max_steps"_a = nb::none()
      )
      .def(
         "_make_input_capsule",
         [owned_capsule](
            const Adapter& self,
            const GroundState& state,
            const std::vector< GroundAction >& actions,
            const std::optional< std::vector< AdapterLiteral > >& goals,
            const std::vector< std::vector< AdapterLiteral > >& subgoal_layers,
            const std::vector< CompactHistoryEntry >& history,
            std::optional< int64_t > history_max_steps
         ) {
            auto input = self.make_input(state, actions);
            apply_optional_lanes(self, input, goals, subgoal_layers, history, history_max_steps);
            return owned_capsule(std::move(input), capsule_bridge::input_name);
         },
         "state"_a,
         "actions"_a = std::vector< GroundAction >{},
         "goals"_a = nb::none(),
         "subgoal_layers"_a = std::vector< std::vector< AdapterLiteral > >{},
         "history"_a = std::vector< CompactHistoryEntry >{},
         "history_max_steps"_a = nb::none()
      )
      .def(
         "_make_inputs_capsule",
         [owned_capsule](
            const Adapter& self,
            const std::vector< LiftedState >& states,
            const std::vector< std::vector< GroundAction > >& actions,
            const std::vector< std::optional< std::vector< AdapterLiteral > > >& goals,
            const std::vector< std::vector< std::vector< AdapterLiteral > > >& subgoal_layers,
            const std::vector< std::vector< CompactHistoryEntry > >& history,
            std::optional< int64_t > history_max_steps
         ) {
            return owned_capsule(
               make_inputs(
                  self, states, actions, goals, subgoal_layers, history, history_max_steps
               ),
               capsule_bridge::inputs_name
            );
         },
         "states"_a,
         "actions"_a = std::vector< std::vector< GroundAction > >{},
         "goals"_a = std::vector< std::optional< std::vector< AdapterLiteral > > >{},
         "subgoal_layers"_a = std::vector< std::vector< std::vector< AdapterLiteral > > >{},
         "history"_a = std::vector< std::vector< CompactHistoryEntry > >{},
         "history_max_steps"_a = nb::none()
      )
      .def(
         "_make_inputs_capsule",
         [owned_capsule](
            const Adapter& self,
            const std::vector< GroundState >& states,
            const std::vector< std::vector< GroundAction > >& actions,
            const std::vector< std::optional< std::vector< AdapterLiteral > > >& goals,
            const std::vector< std::vector< std::vector< AdapterLiteral > > >& subgoal_layers,
            const std::vector< std::vector< CompactHistoryEntry > >& history,
            std::optional< int64_t > history_max_steps
         ) {
            return owned_capsule(
               make_inputs(
                  self, states, actions, goals, subgoal_layers, history, history_max_steps
               ),
               capsule_bridge::inputs_name
            );
         },
         "states"_a,
         "actions"_a = std::vector< std::vector< GroundAction > >{},
         "goals"_a = std::vector< std::optional< std::vector< AdapterLiteral > > >{},
         "subgoal_layers"_a = std::vector< std::vector< std::vector< AdapterLiteral > > >{},
         "history"_a = std::vector< std::vector< CompactHistoryEntry > >{},
         "history_max_steps"_a = nb::none()
      )
      .def(
         "_make_engine_capsule",
         [owned_capsule](const Adapter& self, nb::handle config_capsule) {
            const auto* config = capsule_bridge::get< FlatRelationEncoderConfig >(
               config_capsule.ptr(), capsule_bridge::config_name
            );
            if(config == nullptr) {
               throw nb::python_error();
            }
            return owned_capsule(
               SemanticFlatRelationEncoderEngine(self.get_task_context(), *config),
               capsule_bridge::engine_name
            );
         },
         "config_capsule"_a
      )
      .def(
         "_make_color_engine_capsule",
         [owned_capsule](const Adapter& self, nb::handle config_capsule) {
            const auto* config = capsule_bridge::get< SemanticColorEncoderConfig >(
               config_capsule.ptr(), capsule_bridge::color_config_name
            );
            if(config == nullptr) {
               throw nb::python_error();
            }
            return owned_capsule(
               SemanticColorEncoderEngine(self.get_task_context(), *config),
               capsule_bridge::color_engine_name
            );
         },
         "config_capsule"_a
      )
      .def(
         "_make_hgraph_engine_capsule",
         [owned_capsule](const Adapter& self, nb::handle config_capsule) {
            const auto* config = capsule_bridge::get< SemanticHGraphEncoderConfig >(
               config_capsule.ptr(), capsule_bridge::hgraph_config_name
            );
            if(config == nullptr) {
               throw nb::python_error();
            }
            return owned_capsule(
               SemanticHGraphEncoderEngine(self.get_task_context(), *config),
               capsule_bridge::hgraph_engine_name
            );
         },
         "config_capsule"_a
      )
      .def(
         "_make_successor_hgraph_engine_capsule",
         [owned_capsule](const Adapter& self, nb::handle config_capsule) {
            const auto* config = capsule_bridge::get< SemanticSuccessorHGraphEncoderConfig >(
               config_capsule.ptr(), capsule_bridge::successor_hgraph_config_name
            );
            if(config == nullptr) {
               throw nb::python_error();
            }
            return owned_capsule(
               SemanticSuccessorHGraphEncoderEngine(self.get_task_context(), *config),
               capsule_bridge::successor_hgraph_engine_name
            );
         },
         "config_capsule"_a
      )
      .def(
         "_make_horizon_hgraph_engine_capsule",
         [owned_capsule](const Adapter& self, nb::handle config_capsule) {
            const auto* config = capsule_bridge::get< SemanticHorizonHGraphEncoderConfig >(
               config_capsule.ptr(), capsule_bridge::horizon_hgraph_config_name
            );
            if(config == nullptr) {
               throw nb::python_error();
            }
            return owned_capsule(
               SemanticHorizonHGraphEncoderEngine(self.get_task_context(), *config),
               capsule_bridge::horizon_hgraph_engine_name
            );
         },
         "config_capsule"_a
      )
      .def(
         "_make_flat_horizon_engine_capsule",
         [owned_capsule](const Adapter& self, nb::handle config_capsule) {
            const auto* config = capsule_bridge::get< SemanticFlatHorizonEncoderConfig >(
               config_capsule.ptr(), capsule_bridge::flat_horizon_config_name
            );
            if(config == nullptr) {
               throw nb::python_error();
            }
            return owned_capsule(
               SemanticFlatHorizonEncoderEngine(self.get_task_context(), *config),
               capsule_bridge::flat_horizon_engine_name
            );
         },
         "config_capsule"_a
      );

   // Direct-View encoders. These are the normal PyTyr encode path: the engine
   // runs here, next to the Tyr types, so a state and its actions reach the
   // canonical algorithm as granular Views instead of as an owning semantic
   // input. The `_make_*_capsule` entry points above remain the explicit
   // compatibility route for callers that want the owned records themselves.
   const auto bind_direct_encoder =
      [&m, owned_capsule]< typename Engine >(const char* name, const char* config_capsule_name) {
         using Encoder = DirectEncoder< Engine >;
         nb::class_< Encoder >(m, name)
            .def(
               "__init__",
               [config_capsule_name](
                  Encoder* self, const Adapter& adapter, nb::handle config_capsule
               ) {
                  const auto* config = capsule_bridge::get< typename Engine::Config >(
                     config_capsule.ptr(), config_capsule_name
                  );
                  if(config == nullptr) {
                     throw nb::python_error();
                  }
                  new(self) Encoder(adapter, *config);
               },
               "adapter"_a,
               "config_capsule"_a,
               // The encoder borrows the adapter's View context for its lifetime.
               nb::keep_alive< 1, 2 >()
            )
            .def(
               "_encode_capsule",
               [owned_capsule](
                  const Encoder& self,
                  const LiftedState& state,
                  const std::vector< GroundAction >& actions,
                  const std::optional< std::vector< AdapterLiteral > >& goals,
                  const std::vector< std::vector< AdapterLiteral > >& subgoal_layers,
                  const std::vector< CompactHistoryEntry >& history,
                  std::optional< int64_t > history_max_steps
               ) {
                  return owned_capsule(
                     self.encode(state, actions, goals, subgoal_layers, history, history_max_steps),
                     capsule_bridge::batch_encoding_name
                  );
               },
               "state"_a,
               "actions"_a = std::vector< GroundAction >{},
               "goals"_a = nb::none(),
               "subgoal_layers"_a = std::vector< std::vector< AdapterLiteral > >{},
               "history"_a = std::vector< CompactHistoryEntry >{},
               "history_max_steps"_a = nb::none()
            )
            .def(
               "_encode_capsule",
               [owned_capsule](
                  const Encoder& self,
                  const GroundState& state,
                  const std::vector< GroundAction >& actions,
                  const std::optional< std::vector< AdapterLiteral > >& goals,
                  const std::vector< std::vector< AdapterLiteral > >& subgoal_layers,
                  const std::vector< CompactHistoryEntry >& history,
                  std::optional< int64_t > history_max_steps
               ) {
                  return owned_capsule(
                     self.encode(state, actions, goals, subgoal_layers, history, history_max_steps),
                     capsule_bridge::batch_encoding_name
                  );
               },
               "state"_a,
               "actions"_a = std::vector< GroundAction >{},
               "goals"_a = nb::none(),
               "subgoal_layers"_a = std::vector< std::vector< AdapterLiteral > >{},
               "history"_a = std::vector< CompactHistoryEntry >{},
               "history_max_steps"_a = nb::none()
            );
      };

   bind_direct_encoder.template operator()< SemanticFlatRelationEncoderEngine >(
      "_NativeDirectFlatEncoder", capsule_bridge::config_name
   );
   bind_direct_encoder.template operator()< SemanticColorEncoderEngine >(
      "_NativeDirectColorEncoder", capsule_bridge::color_config_name
   );
   bind_direct_encoder.template operator()< SemanticHGraphEncoderEngine >(
      "_NativeDirectHGraphEncoder", capsule_bridge::hgraph_config_name
   );
}

}  // namespace mifrost::pytyr
