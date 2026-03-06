#include <nanobind/nanobind.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mifrost/binding_kwargs.hpp"
#include "mifrost/bindings.hpp"
#include "mifrost/core/flat_relation_encoder.hpp"
#include "mifrost/input_handling/batch_input_parser.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

namespace {

void apply_flat_config_kwargs(FlatRelationEncoderEngine::Config& config, const nb::kwargs& kwargs)
{
   apply_config_kwargs(config, kwargs, "FlatRelationEncoderConfig");
}

}  // namespace

void init_flat_encoder(nb::module_& m)
{
   nb::class_< FlatRelationEncoderEngine::Config >(m, "FlatRelationEncoderConfig")
      .def(nb::init<>())
      .def(
         "__init__",
         [](FlatRelationEncoderEngine::Config* self, const nb::kwargs& kwargs) {
            new(self) FlatRelationEncoderEngine::Config();
            apply_flat_config_kwargs(*self, kwargs);
         }
      )
      .def_rw("max_goal_level", &FlatRelationEncoderEngine::Config::max_goal_level)
      .def_rw("support_literals", &FlatRelationEncoderEngine::Config::support_literals)
      .def_rw("include_static", &FlatRelationEncoderEngine::Config::include_static)
      .def_rw("export_node_names", &FlatRelationEncoderEngine::Config::export_node_names)
      .def_rw(
         "ignore_zero_arity_relations",
         &FlatRelationEncoderEngine::Config::ignore_zero_arity_relations
      )
      .def_rw(
         "goal_satisfaction_derivations",
         &FlatRelationEncoderEngine::Config::goal_satisfaction_derivations
      );

   nb::class_< FlatRelationEncoderEngine >(m, "FlatRelationEncoderEngine")
      .def(nb::init< const mimir::formalism::DomainImpl& >())
      .def(nb::init< const mimir::formalism::DomainImpl&, FlatRelationEncoderEngine::Config >())
      .def(nb::init< mimir::formalism::Domain >())
      .def(nb::init< mimir::formalism::Domain, FlatRelationEncoderEngine::Config >())
      .def_prop_ro(
         "config", &FlatRelationEncoderEngine::get_config, nb::rv_policy::reference_internal
      )
      .def_prop_ro(
         "relation_dict",
         &FlatRelationEncoderEngine::get_relation_dict,
         nb::rv_policy::reference_internal
      )
      .def_prop_ro("relation_names", &FlatRelationEncoderEngine::get_relation_names)
      .def_prop_ro("relation_arities", &FlatRelationEncoderEngine::get_relation_arities)
      .def_prop_ro("relation_sources", &FlatRelationEncoderEngine::get_relation_sources)
      .def(
         "encode",
         [](FlatRelationEncoderEngine& encoder, const mimir::search::State& state) {
            BatchBuilder builder;
            builder.set_graph_kind("homo");
            encoder.encode(state, builder);
            builder.next_graph();
            return builder.build();
         },
         "state"_a
      )
      .def(
         "encode",
         [](FlatRelationEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals) {
            BatchBuilder builder;
            builder.set_graph_kind("homo");
            encoder.encode(state, goals, builder);
            builder.next_graph();
            return builder.build();
         },
         "state"_a,
         "goals"_a
      )
      .def(
         "encode",
         [](FlatRelationEncoderEngine& encoder,
            const mimir::search::State& state,
            BatchBuilder& builder) { encoder.encode(state, builder); },
         "state"_a,
         "builder"_a
      )
      .def(
         "encode",
         [](FlatRelationEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            BatchBuilder& builder) { encoder.encode(state, goals, builder); },
         "state"_a,
         "goals"_a,
         "builder"_a
      )
      .def(
         "encode_batch",
         [](FlatRelationEncoderEngine& encoder,
            nb::object states,
            nb::object goals,
            nb::object actions,
            nb::object subgoal_layers) {
            auto parsed = batch_input::parse_color_batch_inputs(
               states, goals, actions, subgoal_layers
            );
            return encoder.encode_batch(parsed);
         },
         "states"_a,
         "goals"_a = nb::none(),
         "actions"_a = nb::none(),
         "subgoal_layers"_a = nb::none()
      );
}

}  // namespace mifrost
