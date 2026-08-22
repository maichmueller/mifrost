#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <map>
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
#include "mifrost/core/encoders/homo/semantic_derived_graph_encoder.hpp"
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

/** An optional batch lane is either absent entirely or aligned with the states. */
void validate_batch_lanes(
   size_t states,
   size_t actions,
   size_t goals,
   size_t subgoal_layers,
   size_t history
)
{
   const auto validate_size = [states](size_t size, std::string_view lane) {
      if(size != 0 and size != states) {
         throw std::invalid_argument(
            "PyTyr semantic flat " + std::string(lane) + " batch length must match states"
         );
      }
   };
   validate_size(actions, "actions");
   validate_size(goals, "goals");
   validate_size(subgoal_layers, "subgoal_layers");
   validate_size(history, "history");
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
   validate_batch_lanes(
      states.size(), actions.size(), goals.size(), subgoal_layers.size(), history.size()
   );

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
 * The optional lanes, owned for exactly as long as a View borrows them.
 *
 * These literals arrive from Python as compact tuples, so there is no native
 * value to borrow from in the first place; the state and its actions are the
 * lanes that stay borrowed.
 */
struct OwnedLanes {
   std::vector< SemanticLiteral > goals;
   std::vector< std::vector< SemanticLiteral > > layers;
   std::vector< SemanticHistoryEntry > history;
   bool explicit_goals = false;

   [[nodiscard]] bool only_defaults() const noexcept
   {
      return not explicit_goals and layers.empty() and history.empty();
   }
};

OwnedLanes expand_lanes(
   const SemanticPlanningTaskAdapter& adapter,
   const std::optional< std::vector< AdapterLiteral > >& goals,
   const std::vector< std::vector< AdapterLiteral > >& subgoal_layers,
   const std::vector< CompactHistoryEntry >& history
)
{
   OwnedLanes lanes;
   lanes.explicit_goals = goals.has_value();
   if(goals) {
      lanes.goals = expand_literals(adapter, *goals);
   }
   lanes.layers.reserve(subgoal_layers.size());
   for(const auto& layer : subgoal_layers) {
      lanes.layers.push_back(expand_literals(adapter, layer));
   }
   lanes.history.reserve(history.size());
   for(const auto& [dt, literals] : history) {
      lanes.history.push_back(SemanticHistoryEntry{dt, expand_literals(adapter, literals)});
   }
   return lanes;
}

/**
 * Borrow every preparation capsule in a Python sequence without consuming it.
 *
 * A stream flushes the same preparations repeatedly, so these capsules are not
 * take-once transfers: the payload stays owned by the capsule and the batch
 * reads it in place. Only the pointer vector is built here.
 */
std::vector< const canonical::detail::ViewPreparation* > borrow_preparations(
   const std::vector< nb::object >& capsules
)
{
   std::vector< const canonical::detail::ViewPreparation* > result;
   result.reserve(capsules.size());
   for(const auto& capsule : capsules) {
      const auto* preparation = capsule_bridge::get< canonical::detail::ViewPreparation >(
         capsule.ptr(), capsule_bridge::view_preparation_name
      );
      if(preparation == nullptr) {
         throw nb::python_error();
      }
      result.push_back(preparation);
   }
   return result;
}

/** Address every element of a natively built batch, without copying any of it. */
std::vector< const canonical::detail::ViewPreparation* > address_preparations(
   const std::vector< canonical::detail::ViewPreparation >& preparations
)
{
   std::vector< const canonical::detail::ViewPreparation* > result;
   result.reserve(preparations.size());
   for(const auto& preparation : preparations) {
      result.push_back(&preparation);
   }
   return result;
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
   using Preparation = canonical::detail::ViewPreparation;

   /**
    * The engine is built from the adapter's *schema*, not its problem, so it is
    * reusable across every task of that domain. This facade still holds one
    * adapter, and therefore still encodes one task; what it no longer does is
    * bake that task into the engine.
    */
   DirectEncoder(const SemanticPlanningTaskAdapter& adapter, typename Engine::Config config)
       : adapter_(&adapter), engine_(adapter.get_schema_context(), std::move(config))
   {
   }

   [[nodiscard]] const Engine& engine() const noexcept { return engine_; }

   /**
    * Replace the relation arity table.
    *
    * The direct encoder owns its own engine instance, so a runtime that also
    * keeps a compatibility engine has to update both; otherwise the direct
    * path keeps encoding against the table it was constructed with.
    */
   void update_relations(std::map< std::string, int > relations)
      requires requires(Engine& engine) { engine.update_relations(std::move(relations)); }
   {
      engine_.update_relations(std::move(relations));
   }

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
      return dispatch(
         state,
         actions,
         goals,
         subgoal_layers,
         history,
         history_max_steps,
         [this](const auto&... args) -> decltype(engine_.encode(args...)) {
            return engine_.encode(args...);
         }
      );
   }

   /**
    * Prepare one state without encoding it.
    *
    * The result owns its compact pools and borrows nothing from `state`, so a
    * batch or a stream can hold it after the Tyr state has gone away.
    */
   template < typename State >
   [[nodiscard]] Preparation prepare(
      const State& state,
      const std::vector< tyr::formalism::planning::GroundActionView >& actions,
      const std::optional< std::vector< AdapterLiteral > >& goals,
      const std::vector< std::vector< AdapterLiteral > >& subgoal_layers,
      const std::vector< CompactHistoryEntry >& history,
      std::optional< int64_t > history_max_steps
   ) const
   {
      return dispatch(
         state,
         actions,
         goals,
         subgoal_layers,
         history,
         history_max_steps,
         [this](const auto&... args) -> decltype(engine_.prepare(args...)) {
            return engine_.prepare(args...);
         }
      );
   }

   [[nodiscard]] BatchBuilder::BatchEncoding encode_prepared(
      std::span< const Preparation* const > preparations
   ) const
   {
      return engine_.encode_batch(preparations);
   }

   /**
    * Prepare and encode a whole state batch in one crossing.
    *
    * The preparations stay in one native vector, so the common batch call never
    * pays for a capsule per graph. Streams, which must hold graphs between
    * calls, use `prepare` plus `encode_prepared` instead.
    */
   template < typename State >
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      const std::vector< State >& states,
      const std::vector< std::vector< tyr::formalism::planning::GroundActionView > >& actions,
      const std::vector< std::optional< std::vector< AdapterLiteral > > >& goals,
      const std::vector< std::vector< std::vector< AdapterLiteral > > >& subgoal_layers,
      const std::vector< std::vector< CompactHistoryEntry > >& history,
      std::optional< int64_t > history_max_steps
   ) const
   {
      validate_batch_lanes(
         states.size(), actions.size(), goals.size(), subgoal_layers.size(), history.size()
      );
      const std::vector< tyr::formalism::planning::GroundActionView > no_actions;
      const std::optional< std::vector< AdapterLiteral > > no_goals;
      const std::vector< std::vector< AdapterLiteral > > no_subgoals;
      const std::vector< CompactHistoryEntry > no_history;

      std::vector< Preparation > prepared;
      prepared.reserve(states.size());
      for(size_t index = 0; index < states.size(); ++index) {
         prepared.push_back(prepare(
            states[index],
            actions.empty() ? no_actions : actions[index],
            goals.empty() ? no_goals : goals[index],
            subgoal_layers.empty() ? no_subgoals : subgoal_layers[index],
            history.empty() ? no_history : history[index],
            history_max_steps
         ));
      }
      return encode_prepared(address_preparations(prepared));
   }

  private:
   /**
    * Pick the family's lane arity once, for both `encode` and `prepare`.
    *
    * `call` receives exactly the Views the family accepts, so the two entry
    * points cannot drift into choosing different overloads.
    */
   template < typename State, typename Call >
   [[nodiscard]] auto dispatch(
      const State& state,
      const std::vector< tyr::formalism::planning::GroundActionView >& actions,
      const std::optional< std::vector< AdapterLiteral > >& goals,
      const std::vector< std::vector< AdapterLiteral > >& subgoal_layers,
      const std::vector< CompactHistoryEntry >& history,
      std::optional< int64_t > history_max_steps,
      Call&& call
   ) const
   {
      const auto state_view = adapter_->make_view(state);
      const auto action_views = adapter_->make_action_views(actions);
      const auto lanes = expand_lanes(*adapter_, goals, subgoal_layers, history);

      const semantic::LiteralsView goals_view{std::span{
         lanes.explicit_goals ? lanes.goals : adapter_->get_problem_context()->default_goals
      }};
      const semantic::SubgoalLayersView layers_view{std::span{lanes.layers}};
      const semantic::HistoryView history_view{std::span{lanes.history}};

      // No explicit goals and no other optional lane: let the engine apply its
      // own default-goal handling rather than restating it here.
      const auto problem_context = adapter_->get_problem_context();
      if(lanes.only_defaults()) {
         return call(problem_context, state_view, action_views);
      }
      // Color has no history lane at all; every other family takes one.
      if constexpr(requires {
                      call(
                         problem_context,
                         state_view,
                         goals_view,
                         layers_view,
                         action_views,
                         history_view,
                         history_max_steps
                      );
                   }) {
         return call(
            problem_context,
            state_view,
            goals_view,
            layers_view,
            action_views,
            history_view,
            history_max_steps
         );
      } else {
         if(not lanes.history.empty()) {
            throw std::invalid_argument("this PyTyr encoder family does not accept a history lane");
         }
         (void) history_view;
         (void) history_max_steps;
         return call(problem_context, state_view, goals_view, layers_view, action_views);
      }
   }

   const SemanticPlanningTaskAdapter* adapter_;
   Engine engine_;
};

/**
 * The successor family's direct-View encoder.
 *
 * It is not an instance of `DirectEncoder` because a transition has two state
 * lanes with deliberately different preparations: the successor side reads only
 * the object table and the successor state facts.
 */
class DirectSuccessorEncoder {
  public:
   using Engine = SemanticSuccessorHGraphEncoderEngine;
   using Preparation = canonical::detail::ViewPreparation;

   DirectSuccessorEncoder(
      const SemanticPlanningTaskAdapter& adapter,
      SemanticSuccessorHGraphEncoderConfig config
   )
       : adapter_(&adapter), engine_(adapter.get_schema_context(), std::move(config))
   {
   }

   [[nodiscard]] const Engine& engine() const noexcept { return engine_; }

   /** See `DirectEncoder::update_relations`: this encoder owns its own engine. */
   void update_relations(std::map< std::string, int > relations)
   {
      engine_.update_relations(std::move(relations));
   }

   template < typename CurrentState, typename SuccessorState >
   [[nodiscard]] BatchBuilder::BatchEncoding encode(
      const CurrentState& current,
      const SuccessorState& successor,
      const std::optional< std::vector< AdapterLiteral > >& goals,
      const std::vector< std::vector< AdapterLiteral > >& subgoal_layers
   ) const
   {
      const auto current_view = adapter_->make_view(current);
      const auto successor_view = adapter_->make_view(successor);
      const auto lanes = expand_lanes(*adapter_, goals, subgoal_layers, {});
      const std::vector< tyr::formalism::planning::GroundActionView > no_actions;
      const auto action_views = adapter_->make_action_views(no_actions);

      const auto problem_context = adapter_->get_problem_context();
      if(lanes.only_defaults()) {
         return engine_.encode(
            problem_context, current_view, action_views, successor_view, action_views
         );
      }
      const semantic::LiteralsView goals_view{std::span{
         lanes.explicit_goals ? lanes.goals : adapter_->get_problem_context()->default_goals
      }};
      const semantic::SubgoalLayersView layers_view{std::span{lanes.layers}};
      return engine_.encode(
         problem_context,
         current_view,
         goals_view,
         layers_view,
         action_views,
         successor_view,
         action_views
      );
   }

   template < typename State >
   [[nodiscard]] Preparation prepare_current(
      const State& state,
      const std::optional< std::vector< AdapterLiteral > >& goals,
      const std::vector< std::vector< AdapterLiteral > >& subgoal_layers
   ) const
   {
      const auto state_view = adapter_->make_view(state);
      const auto lanes = expand_lanes(*adapter_, goals, subgoal_layers, {});
      const std::vector< tyr::formalism::planning::GroundActionView > no_actions;
      const auto action_views = adapter_->make_action_views(no_actions);

      const auto problem_context = adapter_->get_problem_context();
      if(lanes.only_defaults()) {
         return engine_.prepare_current(problem_context, state_view, action_views);
      }
      const semantic::LiteralsView goals_view{std::span{
         lanes.explicit_goals ? lanes.goals : adapter_->get_problem_context()->default_goals
      }};
      const semantic::SubgoalLayersView layers_view{std::span{lanes.layers}};
      return engine_.prepare_current(
         problem_context, state_view, goals_view, layers_view, action_views
      );
   }

   template < typename State >
   [[nodiscard]] Preparation prepare_successor(const State& state) const
   {
      return engine_.prepare_successor(adapter_->get_problem_context(), adapter_->make_view(state));
   }

   [[nodiscard]] BatchBuilder::BatchEncoding encode_prepared(
      std::span< const Preparation* const > currents,
      std::span< const Preparation* const > successors
   ) const
   {
      return engine_.encode_batch(currents, successors);
   }

   /** Prepare and encode a whole transition batch in one crossing. */
   template < typename CurrentState, typename SuccessorState >
   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      const std::vector< CurrentState >& currents,
      const std::vector< SuccessorState >& successors,
      const std::vector< std::optional< std::vector< AdapterLiteral > > >& goals,
      const std::vector< std::vector< std::vector< AdapterLiteral > > >& subgoal_layers
   ) const
   {
      if(currents.size() != successors.size()) {
         throw std::invalid_argument("PyTyr successor batch length must match states");
      }
      validate_batch_lanes(currents.size(), 0, goals.size(), subgoal_layers.size(), 0);
      const std::optional< std::vector< AdapterLiteral > > no_goals;
      const std::vector< std::vector< AdapterLiteral > > no_subgoals;

      std::vector< Preparation > current_graphs;
      std::vector< Preparation > successor_graphs;
      current_graphs.reserve(currents.size());
      successor_graphs.reserve(successors.size());
      for(size_t index = 0; index < currents.size(); ++index) {
         current_graphs.push_back(prepare_current(
            currents[index],
            goals.empty() ? no_goals : goals[index],
            subgoal_layers.empty() ? no_subgoals : subgoal_layers[index]
         ));
         successor_graphs.push_back(prepare_successor(successors[index]));
      }
      return encode_prepared(
         address_preparations(current_graphs), address_preparations(successor_graphs)
      );
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
   using LiftedState = tyr::planning::StateView< tyr::LiftedTag >;
   using GroundState = tyr::planning::StateView< tyr::GroundTag >;

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
               SemanticFlatRelationEncoderEngine(self.get_schema_context(), *config),
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
               SemanticColorEncoderEngine(self.get_schema_context(), *config),
               capsule_bridge::color_engine_name
            );
         },
         "config_capsule"_a
      )
      .def(
         "_make_derived_engine_capsule",
         [owned_capsule](const Adapter& self, nb::handle config_capsule) {
            const auto* config = capsule_bridge::get< SemanticDerivedGraphEncoderConfig >(
               config_capsule.ptr(), capsule_bridge::derived_config_name
            );
            if(config == nullptr) {
               throw nb::python_error();
            }
            return owned_capsule(
               SemanticDerivedGraphEncoderEngine(self.get_schema_context(), *config),
               capsule_bridge::derived_engine_name
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
               SemanticHGraphEncoderEngine(self.get_schema_context(), *config),
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
               SemanticSuccessorHGraphEncoderEngine(self.get_schema_context(), *config),
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
               SemanticHorizonHGraphEncoderEngine(self.get_schema_context(), *config),
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
               SemanticFlatHorizonEncoderEngine(self.get_schema_context(), *config),
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
   const auto bind_direct_encoder = [&m, owned_capsule]< typename Engine >(
                                       const char* name, const char* config_capsule_name
                                    ) {
      using Encoder = DirectEncoder< Engine >;
      auto cls = nb::class_< Encoder >(m, name);
      cls.def(
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
         )
         // A stream must hold graphs between calls, so it prepares each state
         // as its own capsule and flushes the collection. The payload is
         // borrowed, never taken: the same capsule survives repeated flushes.
         .def(
            "_prepare_capsule",
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
                  self.prepare(state, actions, goals, subgoal_layers, history, history_max_steps),
                  capsule_bridge::view_preparation_name
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
            "_prepare_capsule",
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
                  self.prepare(state, actions, goals, subgoal_layers, history, history_max_steps),
                  capsule_bridge::view_preparation_name
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
            "_encode_prepared_capsule",
            [owned_capsule](const Encoder& self, const std::vector< nb::object >& prepared) {
               const auto graphs = borrow_preparations(prepared);
               return owned_capsule(
                  self.encode_prepared(std::span{graphs}), capsule_bridge::batch_encoding_name
               );
            },
            "prepared"_a
         )
         // The ordinary batch never needs a capsule per graph: it prepares
         // and encodes inside one crossing.
         .def(
            "_encode_batch_capsule",
            [owned_capsule](
               const Encoder& self,
               const std::vector< LiftedState >& states,
               const std::vector< std::vector< GroundAction > >& actions,
               const std::vector< std::optional< std::vector< AdapterLiteral > > >& goals,
               const std::vector< std::vector< std::vector< AdapterLiteral > > >& subgoal_layers,
               const std::vector< std::vector< CompactHistoryEntry > >& history,
               std::optional< int64_t > history_max_steps
            ) {
               return owned_capsule(
                  self.encode_batch(
                     states, actions, goals, subgoal_layers, history, history_max_steps
                  ),
                  capsule_bridge::batch_encoding_name
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
            "_encode_batch_capsule",
            [owned_capsule](
               const Encoder& self,
               const std::vector< GroundState >& states,
               const std::vector< std::vector< GroundAction > >& actions,
               const std::vector< std::optional< std::vector< AdapterLiteral > > >& goals,
               const std::vector< std::vector< std::vector< AdapterLiteral > > >& subgoal_layers,
               const std::vector< std::vector< CompactHistoryEntry > >& history,
               std::optional< int64_t > history_max_steps
            ) {
               return owned_capsule(
                  self.encode_batch(
                     states, actions, goals, subgoal_layers, history, history_max_steps
                  ),
                  capsule_bridge::batch_encoding_name
               );
            },
            "states"_a,
            "actions"_a = std::vector< std::vector< GroundAction > >{},
            "goals"_a = std::vector< std::optional< std::vector< AdapterLiteral > > >{},
            "subgoal_layers"_a = std::vector< std::vector< std::vector< AdapterLiteral > > >{},
            "history"_a = std::vector< std::vector< CompactHistoryEntry > >{},
            "history_max_steps"_a = nb::none()
         );
      // Only the relation-based families keep an arity table a caller can
      // replace after construction.
      if constexpr(requires(Encoder& encoder) {
                      encoder.update_relations(std::map< std::string, int >{});
                   }) {
         cls.def("_update_relations", &Encoder::update_relations, "relations"_a);
      }
   };

   bind_direct_encoder.template operator()< SemanticFlatRelationEncoderEngine >(
      "_NativeDirectFlatEncoder", capsule_bridge::config_name
   );
   bind_direct_encoder.template operator()< SemanticColorEncoderEngine >(
      "_NativeDirectColorEncoder", capsule_bridge::color_config_name
   );
   bind_direct_encoder.template operator()< SemanticDerivedGraphEncoderEngine >(
      "_NativeDirectDerivedEncoder", capsule_bridge::derived_config_name
   );
   bind_direct_encoder.template operator()< SemanticHGraphEncoderEngine >(
      "_NativeDirectHGraphEncoder", capsule_bridge::hgraph_config_name
   );

   // The successor family binds separately: a transition carries two state
   // lanes and the successor lane is prepared with state facts only.
   const auto bind_successor_states =
      [owned_capsule]< typename CurrentState, typename SuccessorState >(auto& cls) {
         cls.def(
               "_encode_capsule",
               [owned_capsule](
                  const DirectSuccessorEncoder& self,
                  const CurrentState& current,
                  const SuccessorState& successor,
                  const std::optional< std::vector< AdapterLiteral > >& goals,
                  const std::vector< std::vector< AdapterLiteral > >& subgoal_layers
               ) {
                  return owned_capsule(
                     self.encode(current, successor, goals, subgoal_layers),
                     capsule_bridge::batch_encoding_name
                  );
               },
               "current"_a,
               "successor"_a,
               "goals"_a = nb::none(),
               "subgoal_layers"_a = std::vector< std::vector< AdapterLiteral > >{}
         )
            .def(
               "_prepare_capsules",
               [owned_capsule](
                  const DirectSuccessorEncoder& self,
                  const CurrentState& current,
                  const SuccessorState& successor,
                  const std::optional< std::vector< AdapterLiteral > >& goals,
                  const std::vector< std::vector< AdapterLiteral > >& subgoal_layers
               ) {
                  auto current_graph = owned_capsule(
                     self.prepare_current(current, goals, subgoal_layers),
                     capsule_bridge::view_preparation_name
                  );
                  auto successor_graph = owned_capsule(
                     self.prepare_successor(successor), capsule_bridge::view_preparation_name
                  );
                  return std::pair{std::move(current_graph), std::move(successor_graph)};
               },
               "current"_a,
               "successor"_a,
               "goals"_a = nb::none(),
               "subgoal_layers"_a = std::vector< std::vector< AdapterLiteral > >{}
            )
            .def(
               "_encode_batch_capsule",
               [owned_capsule](
                  const DirectSuccessorEncoder& self,
                  const std::vector< CurrentState >& currents,
                  const std::vector< SuccessorState >& successors,
                  const std::vector< std::optional< std::vector< AdapterLiteral > > >& goals,
                  const std::vector< std::vector< std::vector< AdapterLiteral > > >& subgoal_layers
               ) {
                  return owned_capsule(
                     self.encode_batch(currents, successors, goals, subgoal_layers),
                     capsule_bridge::batch_encoding_name
                  );
               },
               "currents"_a,
               "successors"_a,
               "goals"_a = std::vector< std::optional< std::vector< AdapterLiteral > > >{},
               "subgoal_layers"_a = std::vector< std::vector< std::vector< AdapterLiteral > > >{}
            );
      };

   auto successor_encoder = nb::class_< DirectSuccessorEncoder >(
      m, "_NativeDirectSuccessorEncoder"
   );
   successor_encoder
      .def("_update_relations", &DirectSuccessorEncoder::update_relations, "relations"_a)
      .def(
         "__init__",
         [](DirectSuccessorEncoder* self, const Adapter& adapter, nb::handle config_capsule) {
            const auto* config = capsule_bridge::get< SemanticSuccessorHGraphEncoderConfig >(
               config_capsule.ptr(), capsule_bridge::successor_hgraph_config_name
            );
            if(config == nullptr) {
               throw nb::python_error();
            }
            new(self) DirectSuccessorEncoder(adapter, *config);
         },
         "adapter"_a,
         "config_capsule"_a,
         // The encoder borrows the adapter's View context for its lifetime.
         nb::keep_alive< 1, 2 >()
      )
      .def(
         "_encode_prepared_capsule",
         [owned_capsule](
            const DirectSuccessorEncoder& self,
            const std::vector< nb::object >& currents,
            const std::vector< nb::object >& successors
         ) {
            const auto current_graphs = borrow_preparations(currents);
            const auto successor_graphs = borrow_preparations(successors);
            return owned_capsule(
               self.encode_prepared(std::span{current_graphs}, std::span{successor_graphs}),
               capsule_bridge::batch_encoding_name
            );
         },
         "currents"_a,
         "successors"_a
      );
   // A transition may mix lifted and ground states, so bind every combination.
   bind_successor_states.template operator()< LiftedState, LiftedState >(successor_encoder);
   bind_successor_states.template operator()< LiftedState, GroundState >(successor_encoder);
   bind_successor_states.template operator()< GroundState, LiftedState >(successor_encoder);
   bind_successor_states.template operator()< GroundState, GroundState >(successor_encoder);
}

}  // namespace mifrost::pytyr
