#include "batch_input_parser.hpp"

namespace mifrost::batch_input {

parsed::HorizonBatchInputs parse_horizon_batch_inputs(
   nb::handle roots,
   nb::handle dags,
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

   parsed::HorizonBatchInputs out;
   out.roots = parse_states_batch_param(roots, "states");
   const size_t state_count = out.roots.states.size();
   out.dags = parse_dags_batch_param(dags, state_count);
   out.goals = parse_goals_batch_param(goals, state_count);
   out.subgoal_layers = parse_subgoal_layers_batch_param(subgoal_layers, state_count);
   return out;
}

}  // namespace mifrost::batch_input
