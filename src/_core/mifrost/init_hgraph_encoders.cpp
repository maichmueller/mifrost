#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
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
   nb::class_< BatchBuilder >(m, "BatchBuilder")
      .def(nb::init<>())
      .def("build_parts", &BatchBuilder::build_parts)
      .def("next_graph", &BatchBuilder::next_graph);

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
      .def_rw("successor_suffix", &SuccessorHGraphEncoderEngine::Config::successor_suffix);

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
         "encode",
         &HorizonHGraphEncoderEngine::encode,
         "root"_a,
         "dag"_a,
         "goals"_a,
         "builder"_a,
         nb::call_guard< nb::gil_scoped_release >()
      );

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
}

}  // namespace mifrost
