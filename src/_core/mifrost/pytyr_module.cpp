#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <utility>

#include "mifrost/backends/pytyr/semantic_flat_encoder.hpp"
#include "mifrost/capsule_bridge.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost::pytyr {

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
            const std::vector< GroundAction >& actions
         ) { return owned_capsule(self.make_input(state, actions), capsule_bridge::input_name); },
         "state"_a,
         "actions"_a = std::vector< GroundAction >{}
      )
      .def(
         "_make_input_capsule",
         [owned_capsule](
            const Encoder& self,
            const GroundState& state,
            const std::vector< GroundAction >& actions
         ) { return owned_capsule(self.make_input(state, actions), capsule_bridge::input_name); },
         "state"_a,
         "actions"_a = std::vector< GroundAction >{}
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
