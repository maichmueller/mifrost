#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <mimir/formalism/problem.hpp>
#include <mimir/search/state.hpp>

#include "mifrost/bindings.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/color_encoder.hpp"
#include "mifrost/core/goal_inputs.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

void init_color_encoder(nb::module_& m)
{
   auto add_color_extension = [](nb::dict& parts) {
      if(not parts.contains("schema")) {
         return;
      }
      auto schema = nb::cast< nb::dict >(parts["schema"]);
      bool edge_features = false;
      if(schema.contains("flags")) {
         auto flags = nb::cast< nb::dict >(schema["flags"]);
         if(flags.contains("edge_features")) {
            edge_features = nb::cast< bool >(flags["edge_features"]);
         }
      }
      nb::dict extensions;
      if(schema.contains("extensions")) {
         extensions = nb::cast< nb::dict >(schema["extensions"]);
      }
      extensions["color_encoding"] = nb::str(edge_features ? "edge" : "node");
      schema["extensions"] = extensions;
   };

   nb::class_< ColorEncoderEngine::Config >(m, "ColorEncoderConfig")
      .def(nb::init<>())
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
         [&add_color_extension](ColorEncoderEngine& encoder, const mimir::search::State& state) {
            BatchBuilder builder;
            builder.set_graph_kind("homo");
            encoder.encode(state, builder);
            auto parts = builder.build_batch_encoding_py();
            add_color_extension(parts);
            return parts;
         },
         "state"_a
      )
      .def(
         "encode",
         [&add_color_extension](
            ColorEncoderEngine& encoder, const mimir::search::State& state, const GoalInputs& goals
         ) {
            BatchBuilder builder;
            builder.set_graph_kind("homo");
            encoder.encode(state, goals, builder);
            auto parts = builder.build_batch_encoding_py();
            add_color_extension(parts);
            return parts;
         },
         "state"_a,
         "goals"_a
      )
      .def(
         "encode",
         [&add_color_extension](
            ColorEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions
         ) {
            BatchBuilder builder;
            builder.set_graph_kind("homo");
            encoder.encode(state, goals, actions, builder);
            auto parts = builder.build_batch_encoding_py();
            add_color_extension(parts);
            return parts;
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
      .def("flush_batch_encoding_py", &ColorStreamEncoder::flush_batch_encoding_py)
      .def("flush", &ColorStreamEncoder::flush)
      .def("reset", &ColorStreamEncoder::reset);
}

}  // namespace mifrost
