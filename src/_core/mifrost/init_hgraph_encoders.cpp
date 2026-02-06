#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <mimir/formalism/problem.hpp>
#include <mimir/search/axiom_evaluators/grounded/grounded.hpp>
#include <mimir/search/axiom_evaluators/interface.hpp>
#include <mimir/search/grounders/lifted.hpp>
#include <mimir/search/state_repository.hpp>
#include <optional>

#include "mifrost/bindings.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/default_relations.hpp"
#include "mifrost/core/goal_inputs.hpp"
#include "mifrost/core/hgraph_stream_encoder.hpp"
#include "mifrost/core/horizon_hgraph_encoder.hpp"
#include "mifrost/core/nanobind_unordered_dense.hpp"
#include "mifrost/core/successor_hgraph_encoder.hpp"
#include "mifrost/core/transition_dag.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

void init_hgraph_encoders(nb::module_& m)
{
   m.attr("DEFAULT_SYMBOL_TYPE_ID") = defaults::symbol_type_id;
   m.attr("DEFAULT_LGAN_NN_EDGE_POS") = defaults::lgan_nn_edge_pos;
   m.attr("DEFAULT_PARENT_RELATION") = defaults::parent_relation;
   m.attr("DEFAULT_SIBLING_RELATION") = defaults::sibling_relation;
   m.attr("DEFAULT_COUSIN_RELATION") = defaults::cousin_relation;
   m.attr("DEFAULT_HISTORY_LINK_RELATION") = defaults::history_link_relation;

   nb::class_< BatchBuilder >(m, "BatchBuilder")
      .def(nb::init<>())
      .def(
         "add_node_features",
         [](BatchBuilder& builder,
            const std::string& node_type,
            const std::string& attr_name,
            nb::ndarray< nb::numpy, float > data) {
            if(data.ndim() != 1 and data.ndim() != 2) {
               throw std::invalid_argument("add_node_features expects a 1D/2D array");
            }
            const int feature_dim = data.ndim() == 2 ? static_cast< int >(data.shape(1)) : 1;
            const auto count = static_cast< size_t >(data.size());
            builder.add_node_features(
               node_type, attr_name, std::span< const float >(data.data(), count), feature_dim
            );
         }
      )
      .def(
         "add_edges",
         [](BatchBuilder& builder,
            const std::string& src_type,
            const std::string& rel_type,
            const std::string& dst_type,
            nb::ndarray< nb::numpy, int64_t > src,
            nb::ndarray< nb::numpy, int64_t > dst) {
            if(src.ndim() != 1 || dst.ndim() != 1) {
               throw std::invalid_argument("add_edges expects 1D arrays for src/dst indices");
            }
            if(src.size() != dst.size()) {
               throw std::invalid_argument("add_edges expects src/dst arrays of equal length");
            }
            builder.add_edges(
               src_type,
               rel_type,
               dst_type,
               std::span< const int64_t >(src.data(), src.size()),
               std::span< const int64_t >(dst.data(), dst.size())
            );
         }
      )
      .def(
         "add_edge_features",
         [](BatchBuilder& builder,
            const std::string& src_type,
            const std::string& rel_type,
            const std::string& dst_type,
            const std::string& attr_name,
            nb::ndarray< nb::numpy, float > data) {
            if(data.ndim() != 1 and data.ndim() != 2) {
               throw std::invalid_argument("add_edge_features expects a 1D/2D array");
            }
            const int feature_dim = data.ndim() == 2 ? static_cast< int >(data.shape(1)) : 1;
            const auto count = static_cast< size_t >(data.size());
            builder.add_edge_features(
               src_type,
               rel_type,
               dst_type,
               attr_name,
               std::span< const float >(data.data(), count),
               feature_dim
            );
         }
      )
      .def("add_nodes", &BatchBuilder::add_nodes)
      .def("set_node_names", &BatchBuilder::set_node_names)
      .def("set_object_names", &BatchBuilder::set_object_names)
      .def("build", &BatchBuilder::build)
      .def("build_parts", &BatchBuilder::build_parts)
      .def("next_graph", &BatchBuilder::next_graph)
      .def("set_graph_kind", &BatchBuilder::set_graph_kind, "kind"_a)
      .def("set_schema_flag", &BatchBuilder::set_schema_flag, "key"_a, "value"_a);

   nb::class_< GoalInputs >(m, "GoalInputs")
      .def(nb::init<>())
      .def(nb::init< const std::vector< GoalInputs::AnyGoalLiteral >& >(), "goals"_a)
      .def(
         nb::init< const std::vector< GoalInputs::AnyGoalLiteral >&, int >(), "goals"_a, "level"_a
      )
      .def_rw("static_goals", &GoalInputs::static_goals)
      .def_rw("fluent_goals", &GoalInputs::fluent_goals)
      .def_rw("derived_goals", &GoalInputs::derived_goals)
      .def_rw("static_goal_levels", &GoalInputs::static_goal_levels)
      .def_rw("fluent_goal_levels", &GoalInputs::fluent_goal_levels)
      .def_rw("derived_goal_levels", &GoalInputs::derived_goal_levels);

   nb::class_< HGraphEncoderEngine::Config >(m, "HGraphEncoderConfig")
      .def(nb::init<>())
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
      .def_rw("history_link_relation", &HGraphEncoderEngine::Config::history_link_relation)
      .def_rw(
         "lgan_nn_edge_pos",
         &HGraphEncoderEngine::Config::lgan_nn_edge_pos,
         "lgan_nn_edge_pos"_a = defaults::lgan_nn_edge_pos
      );

   nb::class_< HGraphEncoderEngine >(m, "HGraphEncoderEngine")
      .def(nb::init< const mimir::formalism::DomainImpl& >())
      .def(nb::init< const mimir::formalism::DomainImpl&, HGraphEncoderEngine::Config >())
      .def(nb::init< mimir::formalism::Domain >())
      .def(nb::init< mimir::formalism::Domain, HGraphEncoderEngine::Config >())
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder, const mimir::search::State& state) {
            BatchBuilder builder;
            builder.set_graph_kind("hetero");
            encoder.encode(state, builder);
            return builder.build_parts();
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
            return builder.build_parts();
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
            return builder.build_parts();
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
      );

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
      .def(
         "update",
         nb::overload_cast< int64_t, const mimir::search::State& >(&HGraphStreamEncoder::update),
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
            const std::vector< mimir::formalism::GroundAction >& >(&HGraphStreamEncoder::update),
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
            std::optional< int > >(&HGraphStreamEncoder::update),
         "id"_a,
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def("remove", &HGraphStreamEncoder::remove, "id"_a)
      .def("set_reuse_removed", &HGraphStreamEncoder::set_reuse_removed, "value"_a)
      .def("flush_parts", &HGraphStreamEncoder::flush_parts)
      .def("flush", &HGraphStreamEncoder::flush)
      .def("reset", &HGraphStreamEncoder::reset);

   nb::enum_< HorizonHGraphEncoderEngine::Mode >(m, "HorizonEncoderMode")
      .value("Full", HorizonHGraphEncoderEngine::Mode::Full)
      .value("Delta", HorizonHGraphEncoderEngine::Mode::Delta)
      .value("Action", HorizonHGraphEncoderEngine::Mode::Action);

   nb::class_< HorizonHGraphEncoderEngine::Config, HGraphEncoderEngine::Config >(
      m, "HorizonEncoderConfig"
   )
      .def(nb::init<>())
      .def_rw("transition_mode", &HorizonHGraphEncoderEngine::Config::transition_mode)
      .def_rw("target_symbol_prefix", &HorizonHGraphEncoderEngine::Config::target_symbol_prefix)
      .def_rw("parent_relation", &HorizonHGraphEncoderEngine::Config::parent_relation)
      .def_rw("sibling_relation", &HorizonHGraphEncoderEngine::Config::sibling_relation)
      .def_rw("cousin_relation", &HorizonHGraphEncoderEngine::Config::cousin_relation)
      .def_rw("enable_parent_relation", &HorizonHGraphEncoderEngine::Config::enable_parent_relation)
      .def_rw(
         "enable_sibling_relation", &HorizonHGraphEncoderEngine::Config::enable_sibling_relation
      )
      .def_rw("enable_cousin_relation", &HorizonHGraphEncoderEngine::Config::enable_cousin_relation)
      .def_rw(
         "exclude_root_candidate", &HorizonHGraphEncoderEngine::Config::exclude_root_candidate
      );

   nb::enum_< SuccessorHGraphEncoderEngine::Mode >(m, "SuccessorEncoderMode")
      .value("Full", SuccessorHGraphEncoderEngine::Mode::Full)
      .value("Delta", SuccessorHGraphEncoderEngine::Mode::Delta);

   nb::class_< SuccessorHGraphEncoderEngine::Config, HGraphEncoderEngine::Config >(
      m, "SuccessorEncoderConfig"
   )
      .def(nb::init<>())
      .def_rw("successor_mode", &SuccessorHGraphEncoderEngine::Config::successor_mode)
      .def_rw("successor_suffix", &SuccessorHGraphEncoderEngine::Config::successor_suffix)
      .def_rw(
         "include_successor_goal_satisfaction",
         &SuccessorHGraphEncoderEngine::Config::include_successor_goal_satisfaction
      );

   nb::class_< HorizonHGraphEncoderEngine, HGraphEncoderEngine >(m, "HorizonHGraphEncoderEngine")
      .def(nb::init< const mimir::formalism::DomainImpl& >())
      .def(nb::init< const mimir::formalism::DomainImpl&, HorizonHGraphEncoderEngine::Config >())
      .def(nb::init< mimir::formalism::Domain >())
      .def(nb::init< mimir::formalism::Domain, HorizonHGraphEncoderEngine::Config >())
      .def(
         "encode",
         [](HorizonHGraphEncoderEngine& encoder,
            const mimir::search::State& root,
            const TransitionDAG& dag,
            const GoalInputs& goals) {
            BatchBuilder builder;
            builder.set_graph_kind("hetero");
            encoder.encode(root, dag, goals, builder);
            return builder.build_parts();
         },
         "root"_a,
         "dag"_a,
         "goals"_a
      )
      .def(
         "encode",
         &HorizonHGraphEncoderEngine::encode,
         "root"_a,
         "dag"_a,
         "goals"_a,
         "builder"_a,
         nb::call_guard< nb::gil_scoped_release >()
      );

   nb::class_< HorizonStreamEncoder >(m, "HorizonStreamEncoder")
      .def(nb::init< HorizonHGraphEncoderEngine& >(), nb::keep_alive< 1, 2 >())
      .def(
         "append",
         nb::overload_cast< const mimir::search::State&, const TransitionDAG&, const GoalInputs& >(
            &HorizonStreamEncoder::append
         ),
         "root"_a,
         "dag"_a,
         "goals"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast< const mimir::search::State&, const GoalInputs& >(
            &HorizonStreamEncoder::append
         ),
         "root"_a,
         "goals"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "update",
         nb::overload_cast<
            int64_t,
            const mimir::search::State&,
            const TransitionDAG&,
            const GoalInputs& >(&HorizonStreamEncoder::update),
         "id"_a,
         "root"_a,
         "dag"_a,
         "goals"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "update",
         nb::overload_cast< int64_t, const mimir::search::State&, const GoalInputs& >(
            &HorizonStreamEncoder::update
         ),
         "id"_a,
         "root"_a,
         "goals"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def("remove", &HorizonStreamEncoder::remove, "id"_a)
      .def("set_reuse_removed", &HorizonStreamEncoder::set_reuse_removed, "value"_a)
      .def("flush_parts", &HorizonStreamEncoder::flush_parts)
      .def("flush", &HorizonStreamEncoder::flush)
      .def("reset", &HorizonStreamEncoder::reset);

   nb::class_< SuccessorHGraphEncoderEngine, HGraphEncoderEngine >(
      m, "SuccessorHGraphEncoderEngine"
   )
      .def(nb::init< const mimir::formalism::DomainImpl& >())
      .def(nb::init< const mimir::formalism::DomainImpl&, SuccessorHGraphEncoderEngine::Config >())
      .def(nb::init< mimir::formalism::Domain >())
      .def(nb::init< mimir::formalism::Domain, SuccessorHGraphEncoderEngine::Config >())
      .def(
         "encode",
         [](SuccessorHGraphEncoderEngine& encoder,
            const mimir::search::State& current,
            const mimir::search::State& successor,
            const GoalInputs& goals) {
            BatchBuilder builder;
            builder.set_graph_kind("hetero");
            encoder.encode(current, successor, goals, builder);
            return builder.build_parts();
         },
         "current"_a,
         "successor"_a,
         "goals"_a
      )
      .def(
         "encode",
         &SuccessorHGraphEncoderEngine::encode,
         "current"_a,
         "successor"_a,
         "goals"_a,
         "builder"_a,
         nb::call_guard< nb::gil_scoped_release >()
      );

   nb::class_< TransitionStreamEncoder >(m, "TransitionStreamEncoder")
      .def(nb::init< SuccessorHGraphEncoderEngine& >(), nb::keep_alive< 1, 2 >())
      .def(
         "append",
         nb::overload_cast<
            const mimir::search::State&,
            const mimir::search::State&,
            const GoalInputs& >(&TransitionStreamEncoder::append),
         "current"_a,
         "successor"_a,
         "goals"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "update",
         nb::overload_cast<
            int64_t,
            const mimir::search::State&,
            const mimir::search::State&,
            const GoalInputs& >(&TransitionStreamEncoder::update),
         "id"_a,
         "current"_a,
         "successor"_a,
         "goals"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def("remove", &TransitionStreamEncoder::remove, "id"_a)
      .def("set_reuse_removed", &TransitionStreamEncoder::set_reuse_removed, "value"_a)
      .def("flush_parts", &TransitionStreamEncoder::flush_parts)
      .def("flush", &TransitionStreamEncoder::flush)
      .def("reset", &TransitionStreamEncoder::reset);
}

}  // namespace mifrost
