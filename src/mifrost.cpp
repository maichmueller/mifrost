#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <filesystem>
#include <mimir/formalism/problem.hpp>
#include <mimir/search/axiom_evaluators/grounded/grounded.hpp>
#include <mimir/search/axiom_evaluators/interface.hpp>
#include <mimir/search/grounders/lifted.hpp>
#include <mimir/search/state_repository.hpp>
#include <new>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/goal_inputs.hpp"
#include "mifrost/core/hgraph_stream_encoder.hpp"
#include "mifrost/core/horizon_hgraph_encoder.hpp"
#include "mifrost/core/nanobind_unordered_dense.hpp"
#include "mifrost/core/transition_dag.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

NB_MODULE(_core, m)
{
   nb::class_< BatchBuilder >(m, "BatchBuilder")
      .def(nb::init<>())
      .def("build_parts", &BatchBuilder::build_parts);

   nb::class_< GoalInputs >(m, "GoalInputs")
      .def(nb::init< const std::vector< GoalInputs::AnyGoalLiteral >& >(), "goals"_a)
      .def(
         nb::init< const std::vector< GoalInputs::AnyGoalLiteral >&, int >(), "goals"_a, "level"_a
      )
      .def_rw("static_goals", &GoalInputs::static_goals)
      .def_rw("fluent_goals", &GoalInputs::fluent_goals)
      .def_rw("derived_goals", &GoalInputs::derived_goals);

   nb::class_< HGraphEncoderEngine::Config >(m, "HGraphEncoderConfig")
      .def(nb::init<>())
      .def_rw("symbol_type_id", &HGraphEncoderEngine::Config::symbol_type_id)
      .def_rw("nullary_object_name", &HGraphEncoderEngine::Config::nullary_object_name)
      .def_rw("max_goal_level", &HGraphEncoderEngine::Config::max_goal_level)
      .def_rw("support_literals", &HGraphEncoderEngine::Config::support_literals)
      .def_rw("add_nullary_predicates", &HGraphEncoderEngine::Config::add_nullary_predicates)
      .def_rw("ignore_actions", &HGraphEncoderEngine::Config::ignore_actions)
      .def_rw("include_static", &HGraphEncoderEngine::Config::include_static)
      .def_rw("include_lgan_edges", &HGraphEncoderEngine::Config::include_lgan_edges)
      .def_rw(
         "lgan_nn_edge_pos",
         &HGraphEncoderEngine::Config::lgan_nn_edge_pos,
         "lgan_nn_edge_pos"_a = "lgan_nn"
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
      );

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
      .def_rw("enable_parent_relation", &HorizonHGraphEncoderEngine::Config::enable_parent_relation)
      .def_rw(
         "enable_sibling_relation", &HorizonHGraphEncoderEngine::Config::enable_sibling_relation
      )
      .def_rw(
         "exclude_root_candidate", &HorizonHGraphEncoderEngine::Config::exclude_root_candidate
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
            encoder.encode(root, dag, goals, builder);
            return builder.build_parts();
         },
         "root"_a,
         "dag"_a,
         "goals"_a
      )
      .def(
         "encode", &HorizonHGraphEncoderEngine::encode, "root"_a, "dag"_a, "goals"_a, "builder"_a
      );

   nb::class_< TransitionDAG >(m, "TransitionDAG")
      .def(nb::init< mimir::search::State >(), "root"_a)
      .def(
         "register_transition",
         &TransitionDAG::register_transition,
         "parent"_a,
         "child"_a,
         "action"_a = std::nullopt,
         nb::rv_policy::copy
      )
      .def("index", &TransitionDAG::index, "state"_a)
      .def("depth", &TransitionDAG::depth, "idx"_a)
      .def("action", &TransitionDAG::action, "idx"_a)
      .def("state", &TransitionDAG::state, "idx"_a)
      .def("children", &TransitionDAG::children, "parent_idx"_a)
      .def("nodes", &TransitionDAG::nodes, nb::rv_policy::reference_internal)
      .def("successors", &TransitionDAG::successors)
      .def("transitions", &TransitionDAG::transitions)
      .def("root", &TransitionDAG::root)
      .def("root_index", &TransitionDAG::root_index)
      .def("contains", &TransitionDAG::contains, "state"_a);

   nb::class_< TransitionDAG::Node >(m, "TransitionNode")
      .def_ro("state", &TransitionDAG::Node::state)
      .def_ro("index", &TransitionDAG::Node::index)
      .def_ro("depth", &TransitionDAG::Node::depth)
      .def_ro("action", &TransitionDAG::Node::action);
}

}  // namespace mifrost
