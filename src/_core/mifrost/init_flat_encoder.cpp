#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mifrost/backends/pymimir/encoders/flat/flat_relation_encoder.hpp"
#include "mifrost/bindings.hpp"
#include "mifrost/init_batch_encoding.hpp"
#include "mifrost/input_handling/batch_input_parser.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

void init_flat_encoder(nb::module_& m)
{
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
      .def_prop_ro(
         "relation_logical_arities", &FlatRelationEncoderEngine::get_relation_logical_arities
      )
      .def_prop_ro(
         "relation_encoded_arities", &FlatRelationEncoderEngine::get_relation_encoded_arities
      )
      .def_prop_ro("relation_slot_roles", &FlatRelationEncoderEngine::get_relation_slot_roles)
      .def_prop_ro(
         "relation_slot_role_offsets", &FlatRelationEncoderEngine::get_relation_slot_role_offsets
      )
      .def_prop_ro("slot_role_names", &FlatRelationEncoderEngine::get_slot_role_names)
      .def(
         "finalize_batch_encoding",
         &FlatRelationEncoderEngine::finalize_batch_encoding,
         "encoding"_a
      )
      .def(
         "encode",
         [](FlatRelationEncoderEngine& encoder, const mimir::search::State& state) {
            BatchBuilder builder;
            builder.set_graph_kind("flat");
            encoder.encode(state, builder);
            builder.next_graph();
            auto encoding = builder.build();
            encoder.finalize_batch_encoding(encoding);
            return encoding;
         },
         "state"_a
      )
      .def(
         "encode",
         [](FlatRelationEncoderEngine& encoder,
            const mimir::search::State& state,
            const std::vector< mimir::formalism::GroundAction >& actions) {
            BatchBuilder builder;
            builder.set_graph_kind("flat");
            encoder.encode(state, actions, builder);
            builder.next_graph();
            auto encoding = builder.build();
            encoder.finalize_batch_encoding(encoding);
            return encoding;
         },
         "state"_a,
         "actions"_a
      )
      .def(
         "encode",
         [](FlatRelationEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions,
            const std::vector< FlatRelationEncoderEngine::HistorySubgoal >& history_subgoals,
            std::optional< int > history_max_steps) {
            BatchBuilder builder;
            builder.set_graph_kind("flat");
            encoder.encode(state, goals, actions, history_subgoals, history_max_steps, builder);
            builder.next_graph();
            auto encoding = builder.build();
            encoder.finalize_batch_encoding(encoding);
            return encoding;
         },
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt
      )
      .def(
         "encode",
         [](FlatRelationEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions) {
            BatchBuilder builder;
            builder.set_graph_kind("flat");
            encoder.encode(state, goals, actions, builder);
            builder.next_graph();
            auto encoding = builder.build();
            encoder.finalize_batch_encoding(encoding);
            return encoding;
         },
         "state"_a,
         "goals"_a,
         "actions"_a
      )
      .def(
         "encode",
         [](FlatRelationEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals) {
            BatchBuilder builder;
            builder.set_graph_kind("flat");
            encoder.encode(state, goals, builder);
            builder.next_graph();
            auto encoding = builder.build();
            encoder.finalize_batch_encoding(encoding);
            return encoding;
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
            const std::vector< mimir::formalism::GroundAction >& actions,
            BatchBuilder& builder) { encoder.encode(state, actions, builder); },
         "state"_a,
         "actions"_a,
         "builder"_a
      )
      .def(
         "encode",
         [](FlatRelationEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions,
            const std::vector< FlatRelationEncoderEngine::HistorySubgoal >& history_subgoals,
            BatchBuilder& builder) {
            encoder.encode(state, goals, actions, history_subgoals, std::nullopt, builder);
         },
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "builder"_a
      )
      .def(
         "encode",
         [](FlatRelationEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions,
            const std::vector< FlatRelationEncoderEngine::HistorySubgoal >& history_subgoals,
            std::optional< int > history_max_steps,
            BatchBuilder& builder) {
            encoder.encode(state, goals, actions, history_subgoals, history_max_steps, builder);
         },
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a,
         "builder"_a
      )
      .def(
         "encode",
         [](FlatRelationEncoderEngine& encoder,
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
            nb::object subgoal_layers,
            nb::object history_subgoals,
            std::optional< int > history_max_steps) {
            auto parsed = batch_input::parse_flat_batch_inputs(
               states, goals, actions, subgoal_layers, history_subgoals
            );
            return encoder.encode_batch(parsed, history_max_steps);
         },
         "states"_a,
         "goals"_a = nb::none(),
         "actions"_a = nb::none(),
         "subgoal_layers"_a = nb::none(),
         "history_subgoals"_a = nb::none(),
         "history_max_steps"_a = std::nullopt
      );

   nb::class_< FlatRelationMutableStreamEncoder >(m, "FlatRelationMutableStreamEncoder")
      .def(nb::init< FlatRelationEncoderEngine& >(), nb::keep_alive< 1, 2 >())
      .def(
         "append",
         nb::overload_cast< const mimir::search::State& >(
            &FlatRelationMutableStreamEncoder::append
         ),
         "state"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast<
            const mimir::search::State&,
            const std::vector< mimir::formalism::GroundAction >& >(
            &FlatRelationMutableStreamEncoder::append
         ),
         "state"_a,
         "actions"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast< const mimir::search::State&, const GoalInputs& >(
            &FlatRelationMutableStreamEncoder::append
         ),
         "state"_a,
         "goals"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast<
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >& >(
            &FlatRelationMutableStreamEncoder::append
         ),
         "state"_a,
         "goals"_a,
         "actions"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast<
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >&,
            const std::vector< FlatRelationEncoderEngine::HistorySubgoal >&,
            std::optional< int > >(&FlatRelationMutableStreamEncoder::append),
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "update",
         nb::overload_cast< int64_t, const mimir::search::State& >(
            &FlatRelationMutableStreamEncoder::update
         ),
         "id"_a,
         "state"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "update",
         nb::overload_cast<
            int64_t,
            const mimir::search::State&,
            const std::vector< mimir::formalism::GroundAction >& >(
            &FlatRelationMutableStreamEncoder::update
         ),
         "id"_a,
         "state"_a,
         "actions"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "update",
         nb::overload_cast< int64_t, const mimir::search::State&, const GoalInputs& >(
            &FlatRelationMutableStreamEncoder::update
         ),
         "id"_a,
         "state"_a,
         "goals"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "update",
         nb::overload_cast<
            int64_t,
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >& >(
            &FlatRelationMutableStreamEncoder::update
         ),
         "id"_a,
         "state"_a,
         "goals"_a,
         "actions"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "update",
         nb::overload_cast<
            int64_t,
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >&,
            const std::vector< FlatRelationEncoderEngine::HistorySubgoal >&,
            std::optional< int > >(&FlatRelationMutableStreamEncoder::update),
         "id"_a,
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def("remove", &FlatRelationMutableStreamEncoder::remove, "id"_a)
      .def("set_reuse_removed", &FlatRelationMutableStreamEncoder::set_reuse_removed, "value"_a)
      .def("flush", &FlatRelationMutableStreamEncoder::flush)
      .def(
         "flush_pyg",
         [](FlatRelationMutableStreamEncoder& encoder) {
            auto encoding = encoder.flush();
            return batch_encoding_as_pyg(encoding, true);
         }
      )
      .def("reset", &FlatRelationMutableStreamEncoder::reset);

   nb::class_< FlatRelationStreamEncoder >(m, "FlatRelationStreamEncoder")
      .def(nb::init< FlatRelationEncoderEngine& >(), nb::keep_alive< 1, 2 >())
      .def(
         "append",
         nb::overload_cast< const mimir::search::State& >(&FlatRelationStreamEncoder::append),
         "state"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast<
            const mimir::search::State&,
            const std::vector< mimir::formalism::GroundAction >& >(
            &FlatRelationStreamEncoder::append
         ),
         "state"_a,
         "actions"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast< const mimir::search::State&, const GoalInputs& >(
            &FlatRelationStreamEncoder::append
         ),
         "state"_a,
         "goals"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast<
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >& >(
            &FlatRelationStreamEncoder::append
         ),
         "state"_a,
         "goals"_a,
         "actions"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast<
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >&,
            const std::vector< FlatRelationEncoderEngine::HistorySubgoal >&,
            std::optional< int > >(&FlatRelationStreamEncoder::append),
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def("flush", &FlatRelationStreamEncoder::flush)
      .def(
         "flush_pyg",
         [](FlatRelationStreamEncoder& encoder) {
            auto encoding = encoder.flush();
            return batch_encoding_as_pyg(encoding, true);
         }
      )
      .def("reset", &FlatRelationStreamEncoder::reset);
}

}  // namespace mifrost
