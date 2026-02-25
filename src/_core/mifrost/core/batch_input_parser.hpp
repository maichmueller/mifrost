#pragma once

#include <nanobind/nanobind.h>

#include <mimir/formalism/ground_action.hpp>
#include <mimir/search/state.hpp>
#include <optional>
#include <string_view>
#include <vector>

#include "batch_builder.hpp"
#include "color_encoder.hpp"
#include "goal_inputs.hpp"
#include "hgraph_stream_encoder.hpp"
#include "horizon_hgraph_encoder.hpp"
#include "successor_hgraph_encoder.hpp"
#include "transition_dag.hpp"

namespace mifrost::batch_input {

namespace nb = nanobind;

template < typename T >
struct SharedOrPerState {
   std::optional< T > shared = std::nullopt;
   std::optional< std::vector< std::optional< T > > > per_state = std::nullopt;

   [[nodiscard]] const std::optional< T >& at(size_t idx) const
   {
      if(per_state.has_value()) {
         return per_state->at(idx);
      }
      return shared;
   }
};

struct ParsedStateEntry {
   nb::object source;
   mimir::search::State state;
   bool is_wrapper_state = false;
};

struct ParsedStateBatch {
   std::vector< ParsedStateEntry > states;
};

using ParsedGoalPayload = std::vector< LiteralVariant >;
using ParsedActionPayload = std::vector< mimir::formalism::GroundAction >;
using ParsedSubgoalLayersPayload = std::vector< std::vector< LiteralVariant > >;
using ParsedHistoryPayload = std::vector< HGraphEncoderEngine::HistorySubgoal >;

using ParsedGoalBatch = SharedOrPerState< ParsedGoalPayload >;
using ParsedActionBatch = SharedOrPerState< ParsedActionPayload >;
using ParsedSubgoalLayersBatch = SharedOrPerState< ParsedSubgoalLayersPayload >;
using ParsedHistoryBatch = SharedOrPerState< ParsedHistoryPayload >;
using ParsedDagBatch = SharedOrPerState< TransitionDAG >;

ParsedStateBatch
parse_states_batch_param(nb::handle states, std::string_view field_name = "states");

ParsedGoalBatch parse_goals_batch_param(nb::handle goals, size_t state_count);

ParsedActionBatch parse_actions_batch_param(nb::handle actions, size_t state_count);

ParsedSubgoalLayersBatch
parse_subgoal_layers_batch_param(nb::handle subgoal_layers, size_t state_count);

ParsedHistoryBatch
parse_history_subgoals_batch_param(nb::handle history_subgoals, size_t state_count);

std::vector< ParsedStateEntry >
parse_successors_batch_param(nb::handle successors, size_t state_count);

ParsedDagBatch parse_dags_batch_param(nb::handle dags, size_t state_count);

GoalInputs compose_goal_inputs(
   const ParsedGoalPayload& goals,
   const ParsedSubgoalLayersPayload* subgoal_layers = nullptr
);

GoalInputs default_goal_inputs_for_batch_state(const ParsedStateEntry& state_entry);

void reject_unsupported_batch_field(
   std::string_view encoder_name,
   std::string_view field_name,
   nb::handle value
);

BatchBuilder::BatchEncoding hgraph_encode_batch(
   HGraphEncoderEngine& encoder,
   nb::handle states,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers,
   nb::handle history_subgoals,
   std::optional< int > history_max_steps
);

BatchBuilder::BatchEncoding color_encode_batch(
   ColorEncoderEngine& encoder,
   std::string_view encoder_name,
   nb::handle states,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers
);

BatchBuilder::BatchEncoding successor_encode_batch(
   SuccessorHGraphEncoderEngine& encoder,
   std::string_view encoder_name,
   nb::handle states,
   nb::handle successors,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers,
   nb::handle history_subgoals,
   std::optional< int > history_max_steps
);

BatchBuilder::BatchEncoding horizon_encode_batch(
   HorizonHGraphEncoderEngine& encoder,
   std::string_view encoder_name,
   nb::handle roots,
   nb::handle dags,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers,
   nb::handle history_subgoals,
   std::optional< int > history_max_steps
);

nb::list parse_states_batch_python(nb::handle states);
nb::tuple parse_goals_batch_param_python(nb::handle goals, size_t state_count);
nb::tuple parse_actions_batch_param_python(nb::handle actions, size_t state_count);
nb::tuple parse_subgoal_layers_batch_param_python(nb::handle subgoal_layers, size_t state_count);
nb::tuple
parse_history_subgoals_batch_param_python(nb::handle history_subgoals, size_t state_count);
nb::list parse_successors_batch_param_python(nb::handle successors, size_t state_count);
nb::list parse_dags_batch_param_python(nb::handle dags, size_t state_count);
nb::tuple parse_ilg_batch_inputs_python(
   nb::handle states,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers
);

}  // namespace mifrost::batch_input
