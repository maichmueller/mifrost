#include "batch_input_parser.hpp"

namespace mifrost::batch_input {

parsed::FlatBatchInputs parse_flat_batch_inputs(
   nb::handle states,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers,
   nb::handle history_subgoals
)
{
   parsed::FlatBatchInputs out;
   out.states = parse_states_batch_param(states, "states");
   const size_t state_count = out.states.states.size();
   if(state_count == 0) {
      out.goals = parsed::GoalBatch::none();
      out.actions = parsed::ActionBatch::none();
      out.subgoal_layers = parsed::SubgoalLayersBatch::none();
      out.history_subgoals = parsed::HistoryBatch::none();
      return out;
   }

   out.goals = parse_goals_batch_param(goals, state_count);
   out.actions = parse_actions_batch_param(actions, state_count);
   out.subgoal_layers = parse_subgoal_layers_batch_param(subgoal_layers, state_count);
   out.history_subgoals = parse_history_subgoals_batch_param(history_subgoals, state_count);
   return out;
}

}  // namespace mifrost::batch_input
