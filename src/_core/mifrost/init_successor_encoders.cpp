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
#include "mifrost/core/batch_input_parser.hpp"
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

namespace {

void apply_successor_config_kwargs(
   SuccessorHGraphEncoderEngine::Config& config,
   const nb::kwargs& kwargs
)
{
   apply_config_kwargs(config, kwargs, "SuccessorEncoderConfig");
}

}  // namespace

void init_successor_encoders(nb::module_& m)
{
   nb::enum_< SuccessorHGraphEncoderEngine::Mode >(m, "SuccessorEncoderMode")
      .value("Full", SuccessorHGraphEncoderEngine::Mode::Full)
      .value("Delta", SuccessorHGraphEncoderEngine::Mode::Delta);

   nb::class_< SuccessorHGraphEncoderEngine::Config, HGraphEncoderEngine::Config >(
      m, "SuccessorEncoderConfig"
   )
      .def(nb::init<>())
      .def(
         "__init__",
         [](SuccessorHGraphEncoderEngine::Config* self, const nb::kwargs& kwargs) {
            new(self) SuccessorHGraphEncoderEngine::Config();
            apply_successor_config_kwargs(*self, kwargs);
         }
      )
      .def_rw("successor_mode", &SuccessorHGraphEncoderEngine::Config::successor_mode)
      .def_rw("successor_suffix", &SuccessorHGraphEncoderEngine::Config::successor_suffix)
      .def_rw(
         "include_successor_goal_satisfaction",
         &SuccessorHGraphEncoderEngine::Config::include_successor_goal_satisfaction
      );

   nb::class_< SuccessorHGraphEncoderEngine, HGraphEncoderEngine >(
      m, "SuccessorHGraphEncoderEngine"
   )
      .def(nb::init< const mimir::formalism::DomainImpl& >())
      .def(nb::init< const mimir::formalism::DomainImpl&, SuccessorHGraphEncoderEngine::Config >())
      .def(nb::init< mimir::formalism::Domain >())
      .def(nb::init< mimir::formalism::Domain, SuccessorHGraphEncoderEngine::Config >())
      .def_prop_ro(
         "config", &SuccessorHGraphEncoderEngine::get_config, nb::rv_policy::reference_internal
      )
      .def(
         "encode",
         [](SuccessorHGraphEncoderEngine& encoder,
            const mimir::search::State& current,
            const mimir::search::State& successor,
            const GoalInputs& goals) {
            BatchBuilder builder;
            builder.set_graph_kind("hetero");
            encoder.encode(current, successor, goals, builder);
            return builder.build();
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
      )
      .def(
         "encode_batch",
         [](SuccessorHGraphEncoderEngine& encoder,
            std::string encoder_name,
            nb::handle states,
            nb::handle successors,
            nb::handle goals,
            nb::handle actions,
            nb::handle subgoal_layers,
            nb::handle history_subgoals,
            std::optional< int > history_max_steps) {
            return batch_input::successor_encode_batch(
               encoder,
               encoder_name,
               states,
               successors,
               goals,
               actions,
               subgoal_layers,
               history_subgoals,
               history_max_steps
            );
         },
         "encoder_name"_a,
         "states"_a,
         "successors"_a,
         "goals"_a = nb::none(),
         "actions"_a = nb::none(),
         "subgoal_layers"_a = nb::none(),
         "history_subgoals"_a = nb::none(),
         "history_max_steps"_a = std::nullopt
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
      .def("flush", &TransitionStreamEncoder::flush)
      .def("flush_pyg", &TransitionStreamEncoder::flush_pyg)
      .def("reset", &TransitionStreamEncoder::reset);
}

}  // namespace mifrost
