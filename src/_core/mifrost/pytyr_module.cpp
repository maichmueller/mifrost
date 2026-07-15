#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "mifrost/backends/pytyr/semantic_flat_encoder.hpp"
#include "mifrost/capsule_bridge.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost::pytyr {
namespace {

using CompactLiteral = std::tuple< int64_t, std::vector< int64_t >, bool >;
using CompactHistoryEntry = std::tuple< int64_t, std::vector< CompactLiteral > >;

SemanticLiteral expand_literal(const CompactLiteral& value)
{
   const auto& [predicate, arguments, positive] = value;
   return SemanticLiteral{SemanticAtom{predicate, arguments}, positive};
}

std::vector< SemanticLiteral > expand_literals(const std::vector< CompactLiteral >& values)
{
   std::vector< SemanticLiteral > result;
   result.reserve(values.size());
   for(const auto& value : values) {
      result.push_back(expand_literal(value));
   }
   return result;
}

void apply_optional_lanes(
   SemanticFlatRelationInput& input,
   const std::optional< std::vector< CompactLiteral > >& goals,
   const std::vector< std::vector< CompactLiteral > >& subgoal_layers,
   const std::vector< CompactHistoryEntry >& history,
   std::optional< int64_t > history_max_steps
)
{
   if(goals) {
      input.goals = expand_literals(*goals);
   }
   input.subgoal_layers.reserve(subgoal_layers.size());
   for(const auto& layer : subgoal_layers) {
      input.subgoal_layers.push_back(expand_literals(layer));
   }
   input.history.reserve(history.size());
   for(const auto& [dt, literals] : history) {
      input.history.push_back(SemanticHistoryEntry{dt, expand_literals(literals)});
   }
   input.history_max_steps = history_max_steps;
}

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

   using Encoder = SemanticFlatRelationEncoder;
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

   nb::class_< Encoder >(m, "_NativeSemanticFlatRelationEncoder")
      .def(
         "__init__",
         [](Encoder* self, const PlanningTask& task, nb::handle config_capsule) {
            const auto* config = capsule_bridge::get< Encoder::Config >(
               config_capsule.ptr(), capsule_bridge::config_name
            );
            if(config == nullptr) {
               throw nb::python_error();
            }
            new(self) Encoder(task, *config);
         },
         "task"_a,
         "config_capsule"_a
      )
      .def(
         "_make_input_capsule",
         [owned_capsule](
            const Encoder& self,
            const LiftedState& state,
            const std::vector< GroundAction >& actions,
            const std::optional< std::vector< CompactLiteral > >& goals,
            const std::vector< std::vector< CompactLiteral > >& subgoal_layers,
            const std::vector< CompactHistoryEntry >& history,
            std::optional< int64_t > history_max_steps
         ) {
            auto input = self.make_input(state, actions);
            apply_optional_lanes(input, goals, subgoal_layers, history, history_max_steps);
            return owned_capsule(std::move(input), capsule_bridge::input_name);
         },
         "state"_a,
         "actions"_a = std::vector< GroundAction >{},
         "goals"_a = nb::none(),
         "subgoal_layers"_a = std::vector< std::vector< CompactLiteral > >{},
         "history"_a = std::vector< CompactHistoryEntry >{},
         "history_max_steps"_a = nb::none()
      )
      .def(
         "_make_input_capsule",
         [owned_capsule](
            const Encoder& self,
            const GroundState& state,
            const std::vector< GroundAction >& actions,
            const std::optional< std::vector< CompactLiteral > >& goals,
            const std::vector< std::vector< CompactLiteral > >& subgoal_layers,
            const std::vector< CompactHistoryEntry >& history,
            std::optional< int64_t > history_max_steps
         ) {
            auto input = self.make_input(state, actions);
            apply_optional_lanes(input, goals, subgoal_layers, history, history_max_steps);
            return owned_capsule(std::move(input), capsule_bridge::input_name);
         },
         "state"_a,
         "actions"_a = std::vector< GroundAction >{},
         "goals"_a = nb::none(),
         "subgoal_layers"_a = std::vector< std::vector< CompactLiteral > >{},
         "history"_a = std::vector< CompactHistoryEntry >{},
         "history_max_steps"_a = nb::none()
      )
      .def("_make_engine_capsule", [owned_capsule](const Encoder& self) {
         const auto& engine = self.get_engine();
         return owned_capsule(
            SemanticFlatRelationEncoderEngine(
               engine.get_predicates(), engine.get_actions(), engine.get_config()
            ),
            capsule_bridge::engine_name
         );
      });
}

}  // namespace mifrost::pytyr
