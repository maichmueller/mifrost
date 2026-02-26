#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <optional>

#include "mifrost/binding_kwargs.hpp"
#include "mifrost/bindings.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/default_relations.hpp"
#include "mifrost/core/goal_inputs.hpp"
#include "mifrost/core/hgraph_stream_encoder.hpp"
#include "mifrost/input_handling/batch_input_parser.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

namespace {

void apply_hgraph_config_kwargs(HGraphEncoderEngine::Config& config, const nb::kwargs& kwargs)
{
   apply_config_kwargs(config, kwargs, "HGraphEncoderConfig");
}

}  // namespace

void init_hgraph_encoder(nb::module_& m)
{
   nb::class_< HGraphEncoderEngine::Config >(m, "HGraphEncoderConfig")
      .def(nb::init<>())
      .def(
         "__init__",
         [](HGraphEncoderEngine::Config* self, const nb::kwargs& kwargs) {
            new(self) HGraphEncoderEngine::Config();
            apply_hgraph_config_kwargs(*self, kwargs);
         }
      )
      .def_rw("symbol_type_id", &HGraphEncoderEngine::Config::symbol_type_id)
      .def_rw("nullary_object_name", &HGraphEncoderEngine::Config::nullary_object_name)
      .def_rw("max_goal_level", &HGraphEncoderEngine::Config::max_goal_level)
      .def_rw("support_literals", &HGraphEncoderEngine::Config::support_literals)
      .def_rw(
         "goal_satisfaction_derivations",
         &HGraphEncoderEngine::Config::goal_satisfaction_derivations
      )
      .def_rw("add_nullary_predicates", &HGraphEncoderEngine::Config::add_nullary_predicates)
      .def_rw("ignore_actions", &HGraphEncoderEngine::Config::ignore_actions)
      .def_rw("include_static", &HGraphEncoderEngine::Config::include_static)
      .def_rw("include_lgan_edges", &HGraphEncoderEngine::Config::include_lgan_edges)
      .def_rw("include_empty_edge_types", &HGraphEncoderEngine::Config::include_empty_edge_types)
      .def_rw("export_node_names", &HGraphEncoderEngine::Config::export_node_names)
      .def_rw("history_link_relation", &HGraphEncoderEngine::Config::history_link_relation)
      .def_rw(
         "lgan_nn_edge_pos",
         &HGraphEncoderEngine::Config::lgan_nn_edge_pos,
         "lgan_nn_edge_pos"_a = defaults::lgan_nn_edge_pos
      );

   nb::class_< RelationDict >(m, "RelationDict")
      .def(nb::init<>())
      .def(
         "__init__",
         [](RelationDict* self, const std::map< std::string, int >& arity) {
            new(self) RelationDict();
            self->arity = arity;
         },
         "arity"_a
      )
      .def(
         "__init__",
         [](RelationDict* self,
            const std::map< std::string, int >& arity,
            int max_goal_level,
            bool support_literals,
            const std::set< GoalSatisfaction >& goal_satisfaction_derivations) {
            new(self) RelationDict();
            self->arity = arity;
            self->max_goal_level = max_goal_level;
            self->support_literals = support_literals;
            self->goal_satisfaction_derivations = goal_satisfaction_derivations;
         },
         "arity"_a,
         "max_goal_level"_a,
         "support_literals"_a,
         "goal_satisfaction_derivations"_a
      )
      .def_ro("arity", &RelationDict::arity)
      .def_ro("max_goal_level", &RelationDict::max_goal_level)
      .def_ro("support_literals", &RelationDict::support_literals)
      .def_ro("goal_satisfaction_derivations", &RelationDict::goal_satisfaction_derivations)
      .def("__len__", [](const RelationDict& self) { return self.arity.size(); })
      .def("__bool__", [](const RelationDict& self) { return not self.arity.empty(); })
      .def(
         "__contains__",
         [](const RelationDict& self, const std::string& key) {
            return self.arity.find(key) != self.arity.end();
         }
      )
      .def("__contains__", [](const RelationDict&, nb::handle) { return false; })
      .def(
         "__getitem__",
         [](const RelationDict& self, const std::string& key) {
            auto it = self.arity.find(key);
            if(it == self.arity.end()) {
               throw nb::key_error();
            }
            return it->second;
         }
      )
      .def(
         "__iter__",
         [](const RelationDict& self) {
            return nb::make_key_iterator(
               nb::type< RelationDict >(),
               "RelationDictKeyIterator",
               self.arity.begin(),
               self.arity.end()
            );
         },
         nb::keep_alive< 0, 1 >()
      )
      .def(
         "keys",
         [](const RelationDict& self) {
            nb::list out;
            for(const auto& [key, _] : self.arity) {
               out.append(key);
            }
            return out;
         }
      )
      .def(
         "values",
         [](const RelationDict& self) {
            nb::list out;
            for(const auto& [_, value] : self.arity) {
               out.append(value);
            }
            return out;
         }
      )
      .def(
         "items",
         [](const RelationDict& self) {
            nb::list out;
            for(const auto& [key, value] : self.arity) {
               out.append(nb::make_tuple(key, value));
            }
            return out;
         }
      )
      .def(
         "get",
         [](const RelationDict& self, const std::string& key, nb::handle default_value) {
            auto it = self.arity.find(key);
            if(it == self.arity.end()) {
               return nb::borrow< nb::object >(default_value);
            }
            return nb::cast(it->second);
         },
         "key"_a,
         "default_value"_a = nb::none()
      )
      .def("__reduce__", [](const RelationDict& self) {
         nb::tuple args = nb::make_tuple(
            self.arity,
            self.max_goal_level,
            self.support_literals,
            self.goal_satisfaction_derivations
         );
         return nb::make_tuple(nb::type< RelationDict >(), args);
      });

   nb::class_< HGraphEncoderEngine >(m, "HGraphEncoderEngine")
      .def(nb::init< const mimir::formalism::DomainImpl& >())
      .def(nb::init< const mimir::formalism::DomainImpl&, HGraphEncoderEngine::Config >())
      .def(nb::init< mimir::formalism::Domain >())
      .def(nb::init< mimir::formalism::Domain, HGraphEncoderEngine::Config >())
      .def_prop_ro("config", &HGraphEncoderEngine::get_config, nb::rv_policy::reference_internal)
      .def_prop_ro(
         "relation_dict", &HGraphEncoderEngine::get_relation_dict, nb::rv_policy::reference_internal
      )
      .def("update_relations", &HGraphEncoderEngine::update_relations, "relation_dict"_a)
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder, const mimir::search::State& state) {
            BatchBuilder builder;
            builder.set_graph_kind("hetero");
            encoder.encode(state, builder);
            return builder.build();
         },
         "state"_a
      )
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions) {
            BatchBuilder builder;
            builder.set_graph_kind("hetero");
            encoder.encode(state, goals, actions, builder);
            return builder.build();
         },
         "state"_a,
         "goals"_a,
         "actions"_a
      )
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions,
            const std::vector< HGraphEncoderEngine::HistorySubgoal >& history_subgoals,
            std::optional< int > history_max_steps) {
            BatchBuilder builder;
            builder.set_graph_kind("hetero");
            encoder.encode(state, goals, actions, history_subgoals, history_max_steps, builder);
            return builder.build();
         },
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt
      )
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder,
            const mimir::search::State& state,
            BatchBuilder& builder) { encoder.encode(state, builder); },
         "state"_a,
         "builder"_a
      )
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder,
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
         [](HGraphEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions,
            const std::vector< HGraphEncoderEngine::HistorySubgoal >& history_subgoals,
            std::optional< int > history_max_steps,
            BatchBuilder& builder) {
            encoder.encode(state, goals, actions, history_subgoals, history_max_steps, builder);
         },
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt,
         "builder"_a
      )
      .def(
         "encode_batch",
         [](HGraphEncoderEngine& encoder,
            nb::object states,
            nb::object goals,
            nb::object actions,
            nb::object subgoal_layers,
            nb::object history_subgoals,
            std::optional< int > history_max_steps) {
            auto parsed = batch_input::parse_hgraph_batch_inputs(
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

   nb::class_< HGraphMutableStreamEncoder >(m, "HGraphMutableStreamEncoder")
      .def(nb::init< HGraphEncoderEngine& >(), nb::keep_alive< 1, 2 >())
      .def(
         "append",
         nb::overload_cast< const mimir::search::State& >(&HGraphMutableStreamEncoder::append),
         "state"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast<
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >& >(
            &HGraphMutableStreamEncoder::append
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
            const std::vector< HGraphEncoderEngine::HistorySubgoal >&,
            std::optional< int > >(&HGraphMutableStreamEncoder::append),
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
            &HGraphMutableStreamEncoder::update
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
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >& >(
            &HGraphMutableStreamEncoder::update
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
            const std::vector< HGraphEncoderEngine::HistorySubgoal >&,
            std::optional< int > >(&HGraphMutableStreamEncoder::update),
         "id"_a,
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def("remove", &HGraphMutableStreamEncoder::remove, "id"_a)
      .def("set_reuse_removed", &HGraphMutableStreamEncoder::set_reuse_removed, "value"_a)
      .def("flush", &HGraphMutableStreamEncoder::flush)
      .def("flush_pyg", &HGraphMutableStreamEncoder::flush_pyg)
      .def("reset", &HGraphMutableStreamEncoder::reset);

   nb::class_< HGraphStreamEncoder >(m, "HGraphStreamEncoder")
      .def(nb::init< HGraphEncoderEngine& >(), nb::keep_alive< 1, 2 >())
      .def(
         "append",
         nb::overload_cast< const mimir::search::State& >(&HGraphStreamEncoder::append),
         "state"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast<
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >& >(&HGraphStreamEncoder::append),
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
            const std::vector< HGraphEncoderEngine::HistorySubgoal >&,
            std::optional< int > >(&HGraphStreamEncoder::append),
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def("flush", &HGraphStreamEncoder::flush)
      .def("flush_pyg", &HGraphStreamEncoder::flush_pyg)
      .def("reset", &HGraphStreamEncoder::reset);
}

}  // namespace mifrost
