#include "batch_input_parser.hpp"

namespace mifrost::batch_input {

parsed::SuccessorBatchInputs parse_successor_batch_inputs(
   nb::handle states,
   nb::handle successors,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers,
   nb::handle history_subgoals,
   std::optional< int > history_max_steps
)
{
   (void) actions;
   (void) history_subgoals;
   (void) history_max_steps;

   parsed::SuccessorBatchInputs out;
   out.states = parse_states_batch_param(states, "states");
   const size_t state_count = out.states.states.size();
   out.successors = parse_successors_batch_param(successors, state_count);
   out.goals = parse_goals_batch_param(goals, state_count);
   out.subgoal_layers = parse_subgoal_layers_batch_param(subgoal_layers, state_count);
   return out;
}

}  // namespace mifrost::batch_input
