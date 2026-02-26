#include "batch_input_parser.hpp"

namespace mifrost::batch_input {

parsed::ColorBatchInputs parse_color_batch_inputs(
   nb::handle states,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers
)
{
   (void) actions;

   parsed::ColorBatchInputs out;
   out.states = parse_states_batch_param(states, "states");
   const size_t state_count = out.states.states.size();
   out.goals = parse_goals_batch_param(goals, state_count);
   out.subgoal_layers = parse_subgoal_layers_batch_param(subgoal_layers, state_count);
   return out;
}

}  // namespace mifrost::batch_input
