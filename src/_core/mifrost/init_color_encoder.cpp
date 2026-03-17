#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <mimir/formalism/problem.hpp>
#include <mimir/search/state.hpp>

#include "mifrost/bindings.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/encoders/common/goal_inputs.hpp"
#include "mifrost/core/encoders/homo/color_encoder.hpp"
#include "mifrost/input_handling/batch_input_parser.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

namespace {

void apply_color_config_kwargs(ColorEncoderEngine::Config& config, const nb::kwargs& kwargs)
{
   for(const auto& [key_handle, value_handle] : kwargs) {
      const auto key = nb::str(key_handle);
      auto key_view = std::string_view{key.c_str()};
      if(key_view == "edge_features") {
         config.edge_features = nb::cast< bool >(value_handle);
      } else if(key_view == "enable_global_predicate_nodes") {
         config.enable_global_predicate_nodes = nb::cast< bool >(value_handle);
      } else {
         throw std::invalid_argument(
            fmt::format("Unknown ColorEncoderConfig kwarg '{}'", key_view)
         );
      }
   }
}

}  // namespace

void init_color_encoder(nb::module_& m)
{
   nb::class_< ColorEncoderEngine::Config >(m, "ColorEncoderConfig")
      .def(nb::init<>())
      .def(
         "__init__",
         [](ColorEncoderEngine::Config* self, nb::kwargs kwargs) {
            new(self) ColorEncoderEngine::Config();
            apply_color_config_kwargs(*self, kwargs);
         }
      )
      .def_rw("edge_features", &ColorEncoderEngine::Config::edge_features)
      .def_rw(
         "enable_global_predicate_nodes", &ColorEncoderEngine::Config::enable_global_predicate_nodes
      );

   nb::class_< ColorEncoderEngine >(m, "ColorEncoderEngine")
      .def(nb::init< const mimir::formalism::DomainImpl& >())
      .def(nb::init< const mimir::formalism::DomainImpl&, ColorEncoderEngine::Config >())
      .def(nb::init< mimir::formalism::Domain >())
      .def(nb::init< mimir::formalism::Domain, ColorEncoderEngine::Config >())
      .def(
         "encode",
         [](ColorEncoderEngine& encoder, const mimir::search::State& state) {
            BatchBuilder builder;
            builder.set_graph_kind("homo");
            encoder.encode(state, builder);
            return builder.build();
         },
         "state"_a
      )
      .def(
         "encode",
         [](ColorEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals) {
            BatchBuilder builder;
            builder.set_graph_kind("homo");
            encoder.encode(state, goals, builder);
            return builder.build();
         },
         "state"_a,
         "goals"_a
      )
      .def(
         "encode",
         [](ColorEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions) {
            BatchBuilder builder;
            builder.set_graph_kind("homo");
            encoder.encode(state, goals, actions, builder);
            return builder.build();
         },
         "state"_a,
         "goals"_a,
         "actions"_a
      )
      .def(
         "encode",
         [](ColorEncoderEngine& encoder, const mimir::search::State& state, BatchBuilder& builder) {
            encoder.encode(state, builder);
         },
         "state"_a,
         "builder"_a
      )
      .def(
         "encode",
         [](ColorEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            BatchBuilder& builder) { encoder.encode(state, goals, builder); },
         "state"_a,
         "goals"_a,
         "builder"_a
      )
      .def(
         "encode",
         [](ColorEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions,
            BatchBuilder& builder) { encoder.encode(state, goals, actions, builder); },
         "state"_a,
         "goals"_a,
         "actions"_a,
         "builder"_a
      )
      .def(
         "encode_batch",
         [](ColorEncoderEngine& encoder,
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

   nb::class_< ColorStreamEncoder >(m, "ColorStreamEncoder")
      .def(nb::init< ColorEncoderEngine& >(), nb::keep_alive< 1, 2 >())
      .def(
         "append",
         nb::overload_cast< const mimir::search::State& >(&ColorStreamEncoder::append),
         "state"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast< const mimir::search::State&, const GoalInputs& >(
            &ColorStreamEncoder::append
         ),
         "state"_a,
         "goals"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "update",
         nb::overload_cast< int64_t, const mimir::search::State& >(&ColorStreamEncoder::update),
         "id"_a,
         "state"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "update",
         nb::overload_cast< int64_t, const mimir::search::State&, const GoalInputs& >(
            &ColorStreamEncoder::update
         ),
         "id"_a,
         "state"_a,
         "goals"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def("remove", &ColorStreamEncoder::remove, "id"_a)
      .def("set_reuse_removed", &ColorStreamEncoder::set_reuse_removed, "value"_a)
      .def("flush", &ColorStreamEncoder::flush)
      .def("flush_pyg", &ColorStreamEncoder::flush_pyg)
      .def("reset", &ColorStreamEncoder::reset);
}

}  // namespace mifrost
