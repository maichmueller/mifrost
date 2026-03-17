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

#include "mifrost/binding_kwargs.hpp"
#include "mifrost/bindings.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/encoders/common/default_relations.hpp"
#include "mifrost/core/encoders/common/goal_inputs.hpp"
#include "mifrost/core/encoders/common/transition_dag.hpp"
#include "mifrost/core/encoders/flat/flat_horizon_encoder.hpp"
#include "mifrost/core/encoders/hetero/hgraph_stream_encoder.hpp"
#include "mifrost/core/encoders/hetero/horizon_hgraph_encoder.hpp"
#include "mifrost/core/encoders/hetero/successor_hgraph_encoder.hpp"
#include "mifrost/core/nanobind_unordered_dense.hpp"
#include "mifrost/input_handling/batch_input_parser.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

namespace {

void apply_horizon_config_kwargs(
   HorizonHGraphEncoderEngine::Config& config,
   const nb::kwargs& kwargs
)
{
   apply_config_kwargs(config, kwargs, "HorizonEncoderConfig");
}

}  // namespace

void init_horizon_encoder(nb::module_& m)
{
   nb::enum_< RootPolicy >(m, "RootPolicy")
      .value("include", RootPolicy::include)
      .value("encode_only", RootPolicy::encode_only)
      .value("exclude", RootPolicy::exclude);
   nb::enum_< HorizonHGraphEncoderEngine::Mode >(m, "HorizonEncoderMode")
      .value("full", HorizonHGraphEncoderEngine::Mode::full)
      .value("delta", HorizonHGraphEncoderEngine::Mode::delta)
      .value("action", HorizonHGraphEncoderEngine::Mode::action);
   nb::enum_< FlatHorizonEncoderEngine::Mode >(m, "FlatHorizonEncoderMode")
      .value("full", FlatHorizonEncoderEngine::Mode::full)
      .value("delta", FlatHorizonEncoderEngine::Mode::delta)
      .value("action", FlatHorizonEncoderEngine::Mode::action);

   nb::class_< HorizonHGraphEncoderEngine::Config, HGraphEncoderEngine::Config >(
      m, "HorizonEncoderConfig"
   )
      .def(nb::init<>())
      .def(
         "__init__",
         [](HorizonHGraphEncoderEngine::Config* self, const nb::kwargs& kwargs) {
            new(self) HorizonHGraphEncoderEngine::Config();
            apply_horizon_config_kwargs(*self, kwargs);
         }
      )
      .def_rw("transition_mode", &HorizonHGraphEncoderEngine::Config::transition_mode)
      .def_rw("parent_relation", &HorizonHGraphEncoderEngine::Config::parent_relation)
      .def_rw("sibling_relation", &HorizonHGraphEncoderEngine::Config::sibling_relation)
      .def_rw("cousin_relation", &HorizonHGraphEncoderEngine::Config::cousin_relation)
      .def_rw("enable_parent_relation", &HorizonHGraphEncoderEngine::Config::enable_parent_relation)
      .def_rw(
         "enable_sibling_relation", &HorizonHGraphEncoderEngine::Config::enable_sibling_relation
      )
      .def_rw("enable_cousin_relation", &HorizonHGraphEncoderEngine::Config::enable_cousin_relation)
      .def_rw("root_policy", &HorizonHGraphEncoderEngine::Config::root_policy);

   nb::class_< HorizonHGraphEncoderEngine, HGraphEncoderEngine >(m, "HorizonHGraphEncoderEngine")
      .def(nb::init< const mimir::formalism::DomainImpl& >())
      .def(nb::init< const mimir::formalism::DomainImpl&, HorizonHGraphEncoderEngine::Config >())
      .def(nb::init< mimir::formalism::Domain >())
      .def(nb::init< mimir::formalism::Domain, HorizonHGraphEncoderEngine::Config >())
      .def_prop_ro(
         "config", &HorizonHGraphEncoderEngine::get_config, nb::rv_policy::reference_internal
      )
      .def("update_relations", &HorizonHGraphEncoderEngine::update_relations, "relation_dict"_a)
      .def(
         "encode",
         [](HorizonHGraphEncoderEngine& encoder,
            const mimir::search::State& root,
            const TransitionDAG& dag,
            const GoalInputs& goals) {
            BatchBuilder builder;
            builder.set_graph_kind("hetero");
            encoder.encode(root, dag, goals, builder);
            return builder.build();
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
      )
      .def(
         "encode_batch",
         [](HorizonHGraphEncoderEngine& encoder,
            nb::object roots,
            nb::object dags,
            nb::object goals,
            nb::object actions,
            nb::object subgoal_layers,
            nb::object history_subgoals,
            std::optional< int > history_max_steps) {
            auto parsed = batch_input::parse_horizon_batch_inputs(
               roots, dags, goals, actions, subgoal_layers, history_subgoals, history_max_steps
            );
            return encoder.encode_batch(parsed);
         },
         "roots"_a,
         "dags"_a = nb::none(),
         "goals"_a = nb::none(),
         "actions"_a = nb::none(),
         "subgoal_layers"_a = nb::none(),
         "history_subgoals"_a = nb::none(),
         "history_max_steps"_a = std::nullopt
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
      .def("flush", &HorizonStreamEncoder::flush)
      .def("flush_pyg", &HorizonStreamEncoder::flush_pyg)
      .def("reset", &HorizonStreamEncoder::reset);

   nb::class_< FlatHorizonEncoderEngine::Config >(m, "FlatHorizonEncoderConfig")
      .def(nb::init<>())
      .def(
         "__init__",
         [](FlatHorizonEncoderEngine::Config* self, const nb::kwargs& kwargs) {
            new(self) FlatHorizonEncoderEngine::Config();
            apply_config_kwargs(*self, kwargs, "FlatHorizonEncoderConfig");
         }
      )
      .def_rw("max_goal_level", &FlatHorizonEncoderEngine::Config::max_goal_level)
      .def_rw("support_literals", &FlatHorizonEncoderEngine::Config::support_literals)
      .def_rw("include_static", &FlatHorizonEncoderEngine::Config::include_static)
      .def_rw("export_node_names", &FlatHorizonEncoderEngine::Config::export_node_names)
      .def_rw(
         "ignore_zero_arity_relations",
         &FlatHorizonEncoderEngine::Config::ignore_zero_arity_relations
      )
      .def_rw("ignore_actions", &FlatHorizonEncoderEngine::Config::ignore_actions)
      .def_rw(
         "use_predicate_virtual_nodes",
         &FlatHorizonEncoderEngine::Config::use_predicate_virtual_nodes
      )
      .def_rw("include_lgan_edges", &FlatHorizonEncoderEngine::Config::include_lgan_edges)
      .def_rw("transition_mode", &FlatHorizonEncoderEngine::Config::transition_mode)
      .def_rw("target_symbol_prefix", &FlatHorizonEncoderEngine::Config::target_symbol_prefix)
      .def_rw("parent_relation", &FlatHorizonEncoderEngine::Config::parent_relation)
      .def_rw("sibling_relation", &FlatHorizonEncoderEngine::Config::sibling_relation)
      .def_rw("cousin_relation", &FlatHorizonEncoderEngine::Config::cousin_relation)
      .def_rw("lgan_tn_edge_pos", &FlatHorizonEncoderEngine::Config::lgan_tn_edge_pos)
      .def_rw("lgan_nn_edge_pos", &FlatHorizonEncoderEngine::Config::lgan_nn_edge_pos)
      .def_rw("lgan_rr_edge_pos", &FlatHorizonEncoderEngine::Config::lgan_rr_edge_pos)
      .def_rw("enable_parent_relation", &FlatHorizonEncoderEngine::Config::enable_parent_relation)
      .def_rw("enable_sibling_relation", &FlatHorizonEncoderEngine::Config::enable_sibling_relation)
      .def_rw("enable_cousin_relation", &FlatHorizonEncoderEngine::Config::enable_cousin_relation)
      .def_rw("root_policy", &FlatHorizonEncoderEngine::Config::root_policy)
      .def_rw("goal_derivations", &FlatHorizonEncoderEngine::Config::goal_derivations);

   nb::class_< FlatHorizonEncoderEngine >(m, "FlatHorizonEncoderEngine")
      .def(nb::init< const mimir::formalism::DomainImpl& >())
      .def(nb::init< const mimir::formalism::DomainImpl&, FlatHorizonEncoderEngine::Config >())
      .def(nb::init< mimir::formalism::Domain >())
      .def(nb::init< mimir::formalism::Domain, FlatHorizonEncoderEngine::Config >())
      .def_prop_ro(
         "config", &FlatHorizonEncoderEngine::get_config, nb::rv_policy::reference_internal
      )
      .def(
         "relation_dict",
         &FlatHorizonEncoderEngine::get_relation_dict,
         nb::rv_policy::reference_internal
      )
      .def_prop_ro("relation_names", &FlatHorizonEncoderEngine::get_relation_names)
      .def_prop_ro("relation_arities", &FlatHorizonEncoderEngine::get_relation_arities)
      .def_prop_ro("relation_sources", &FlatHorizonEncoderEngine::get_relation_sources)
      .def_prop_ro(
         "relation_logical_arities", &FlatHorizonEncoderEngine::get_relation_logical_arities
      )
      .def_prop_ro(
         "relation_encoded_arities", &FlatHorizonEncoderEngine::get_relation_encoded_arities
      )
      .def_prop_ro("relation_slot_roles", &FlatHorizonEncoderEngine::get_relation_slot_roles)
      .def_prop_ro(
         "relation_slot_role_offsets", &FlatHorizonEncoderEngine::get_relation_slot_role_offsets
      )
      .def_prop_ro("slot_role_names", &FlatHorizonEncoderEngine::get_slot_role_names)
      .def(
         "encode",
         [](FlatHorizonEncoderEngine& encoder,
            const mimir::search::State& root,
            const TransitionDAG& dag,
            const GoalInputs& goals) {
            BatchBuilder builder;
            builder.set_graph_kind("flat");
            encoder.encode(root, dag, goals, builder);
            return builder.build();
         },
         "root"_a,
         "dag"_a,
         "goals"_a
      )
      .def(
         "encode",
         &FlatHorizonEncoderEngine::encode,
         "root"_a,
         "dag"_a,
         "goals"_a,
         "builder"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "encode_batch",
         [](FlatHorizonEncoderEngine& encoder,
            nb::object roots,
            nb::object dags,
            nb::object goals,
            nb::object actions,
            nb::object subgoal_layers,
            nb::object history_subgoals,
            std::optional< int > history_max_steps) {
            auto parsed = batch_input::parse_horizon_batch_inputs(
               roots, dags, goals, actions, subgoal_layers, history_subgoals, history_max_steps
            );
            return encoder.encode_batch(parsed);
         },
         "roots"_a,
         "dags"_a = nb::none(),
         "goals"_a = nb::none(),
         "actions"_a = nb::none(),
         "subgoal_layers"_a = nb::none(),
         "history_subgoals"_a = nb::none(),
         "history_max_steps"_a = std::nullopt
      );

   nb::class_< FlatHorizonStreamEncoder >(m, "FlatHorizonStreamEncoder")
      .def(nb::init< FlatHorizonEncoderEngine& >(), nb::keep_alive< 1, 2 >())
      .def(
         "append",
         nb::overload_cast< const mimir::search::State&, const TransitionDAG&, const GoalInputs& >(
            &FlatHorizonStreamEncoder::append
         ),
         "root"_a,
         "dag"_a,
         "goals"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast< const mimir::search::State&, const GoalInputs& >(
            &FlatHorizonStreamEncoder::append
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
            const GoalInputs& >(&FlatHorizonStreamEncoder::update),
         "id"_a,
         "root"_a,
         "dag"_a,
         "goals"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "update",
         nb::overload_cast< int64_t, const mimir::search::State&, const GoalInputs& >(
            &FlatHorizonStreamEncoder::update
         ),
         "id"_a,
         "root"_a,
         "goals"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def("remove", &FlatHorizonStreamEncoder::remove, "id"_a)
      .def("set_reuse_removed", &FlatHorizonStreamEncoder::set_reuse_removed, "value"_a)
      .def("flush", &FlatHorizonStreamEncoder::flush)
      .def("flush_pyg", &FlatHorizonStreamEncoder::flush_pyg)
      .def("reset", &FlatHorizonStreamEncoder::reset);
}

}  // namespace mifrost
